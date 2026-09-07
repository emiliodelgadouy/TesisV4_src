from __future__ import annotations
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterator, Literal

import numpy as np
import pandas as pd
import tensorflow as tf

from src.dataset.images import ImageDecoder
from src.dataset.mixup import PositiveMixup
from src.training.mode import TrainingMode

SplitName = Literal["train", "val", "test"]
PatchSampling = Literal["uniform", "normal"]
PatchCropStrategy = Literal["uniform", "roi", "normal", "avoid_roi"]
BagCanvasMode = Literal["resize", "pad"]

DEFAULT_PATCH_CROP_BY_LABEL: dict[float, PatchCropStrategy] = {
    0.0: "avoid_roi",
    1.0: "roi",
}


def _normalize_size(size: tuple[int, int] | int) -> tuple[int, int]:
    if isinstance(size, int):
        return (int(size), int(size))
    h, w = size
    return (int(h), int(w))


def _resize_preserving_dtype(img: tf.Tensor, size: tf.Tensor | list[int]) -> tf.Tensor:
    """tf.image.resize promueve a float32; restauramos el dtype de entrada."""
    return tf.cast(tf.image.resize(img, size), img.dtype)


def _ensure_min_spatial(img: tf.Tensor, min_h: int, min_w: int) -> tf.Tensor:
    """Escala la imagen solo si es mas chica que el parche objetivo."""
    h = tf.shape(img)[0]
    w = tf.shape(img)[1]
    needs_upscale = tf.logical_or(h < min_h, w < min_w)

    def _upscale() -> tf.Tensor:
        scale = tf.maximum(
            tf.cast(min_h, tf.float32) / tf.cast(h, tf.float32),
            tf.cast(min_w, tf.float32) / tf.cast(w, tf.float32),
        )
        new_h = tf.cast(tf.math.ceil(tf.cast(h, tf.float32) * scale), tf.int32)
        new_w = tf.cast(tf.math.ceil(tf.cast(w, tf.float32) * scale), tf.int32)
        return _resize_preserving_dtype(img, [new_h, new_w])

    return tf.cond(needs_upscale, _upscale, lambda: img)


def _random_patch_offset(
    max_offset: tf.Tensor,
    *,
    sampling: PatchSampling,
    mean_frac: float,
    sigma_frac: float,
) -> tf.Tensor:
    """Offset de crop en [0, max_offset]; normal truncada sesgada o uniforme."""
    zero = tf.constant(0, dtype=tf.int32)

    def _sample() -> tf.Tensor:
        max_f = tf.cast(max_offset, tf.float32)
        if sampling == "uniform":
            offset = tf.random.uniform(()) * max_f
        else:
            mean = tf.cast(mean_frac, tf.float32) * max_f
            std = tf.maximum(tf.cast(sigma_frac, tf.float32) * max_f, 1.0)
            offset = tf.random.normal(()) * std + mean
            offset = tf.clip_by_value(offset, 0.0, max_f)
        return tf.cast(tf.round(offset), tf.int32)

    return tf.cond(max_offset > 0, _sample, lambda: zero)


def _deterministic_patch_offset(max_offset: tf.Tensor, mean_frac: float) -> tf.Tensor:
    return tf.cast(
        tf.round(tf.cast(max_offset, tf.float32) * tf.cast(mean_frac, tf.float32)),
        tf.int32,
    )


def _uniform_patch_offsets(max_y: tf.Tensor, max_x: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    y0 = _random_patch_offset(
        max_y, sampling="uniform", mean_frac=0.5, sigma_frac=0.0
    )
    x0 = _random_patch_offset(
        max_x, sampling="uniform", mean_frac=0.5, sigma_frac=0.0
    )
    return y0, x0


def _roi_is_valid(
    roi_xmin: tf.Tensor,
    roi_ymin: tf.Tensor,
    roi_xmax: tf.Tensor,
    roi_ymax: tf.Tensor,
) -> tf.Tensor:
    return tf.logical_and(
        tf.less(roi_xmin, roi_xmax),
        tf.logical_and(
            tf.less(roi_ymin, roi_ymax),
            tf.math.is_finite(roi_xmin + roi_ymin + roi_xmax + roi_ymax),
        ),
    )


def _roi_box_pixels(
    roi_xmin: tf.Tensor,
    roi_ymin: tf.Tensor,
    roi_xmax: tf.Tensor,
    roi_ymax: tf.Tensor,
    h: tf.Tensor,
    w: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    h0 = tf.cast(h, tf.float32)
    w0 = tf.cast(w, tf.float32)
    return (
        roi_xmin * w0,
        roi_ymin * h0,
        roi_xmax * w0,
        roi_ymax * h0,
    )


def _offsets_avoiding_roi(
    max_y: tf.Tensor,
    max_x: tf.Tensor,
    ph: int,
    pw: int,
    roi_xmin: tf.Tensor,
    roi_ymin: tf.Tensor,
    roi_xmax: tf.Tensor,
    roi_ymax: tf.Tensor,
    h: tf.Tensor,
    w: tf.Tensor,
    *,
    random_patch: bool,
    max_attempts: int,
    path: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Muestrea offsets de parche que no intersectan la ROI (rejection sampling)."""
    max_y_i = tf.maximum(max_y, 0)
    max_x_i = tf.maximum(max_x, 0)
    n = max_attempts

    if random_patch:
        # Enteros en [0, max_*_i]; evitar uniform+round con maxval=max+1 (puede dar max+1).
        y0_cand_i = tf.random.uniform([n], maxval=max_y_i + 1, dtype=tf.int32)
        x0_cand_i = tf.random.uniform([n], maxval=max_x_i + 1, dtype=tf.int32)
    else:
        bucket = tf.cast(tf.strings.to_hash_bucket_fast(path, 2**31 - 1), tf.int32)
        idx = tf.cast(tf.range(n), tf.int32)
        seeds = bucket + idx * 9973
        y0_cand_i = seeds % tf.maximum(max_y_i + 1, 1)
        x0_cand_i = (seeds // 17) % tf.maximum(max_x_i + 1, 1)
    xmin, ymin, xmax, ymax = _roi_box_pixels(roi_xmin, roi_ymin, roi_xmax, roi_ymax, h, w)

    y0f = tf.cast(y0_cand_i, tf.float32)
    x0f = tf.cast(x0_cand_i, tf.float32)
    phf = tf.cast(ph, tf.float32)
    pwf = tf.cast(pw, tf.float32)
    intersects = tf.logical_not(
        tf.logical_or(
            tf.logical_or(y0f + phf <= ymin, y0f >= ymax),
            tf.logical_or(x0f + pwf <= xmin, x0f >= xmax),
        )
    )
    valid = tf.logical_not(intersects)
    has_valid = tf.reduce_any(valid)
    first_idx = tf.argmax(tf.cast(valid, tf.int32))
    y0_pick = y0_cand_i[first_idx]
    x0_pick = x0_cand_i[first_idx]
    y0_fb, x0_fb = _scan_disjoint_offset(
        max_y, max_x, ph, pw, xmin, ymin, xmax, ymax, stride=8
    )
    return (
        tf.cond(has_valid, lambda: y0_pick, lambda: y0_fb),
        tf.cond(has_valid, lambda: x0_pick, lambda: x0_fb),
    )


def _scan_disjoint_offset(
    max_y: tf.Tensor,
    max_x: tf.Tensor,
    ph: int,
    pw: int,
    roi_xmin_px: tf.Tensor,
    roi_ymin_px: tf.Tensor,
    roi_xmax_px: tf.Tensor,
    roi_ymax_px: tf.Tensor,
    *,
    stride: int = 8,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Busqueda en grilla (fallback) de un offset sin interseccion con la ROI."""
    max_y_i = tf.maximum(max_y, 0)
    max_x_i = tf.maximum(max_x, 0)
    y0s = tf.range(0, max_y_i + 1, stride)
    x0s = tf.range(0, max_x_i + 1, stride)
    y0_grid, x0_grid = tf.meshgrid(y0s, x0s, indexing="ij")
    y0_flat = tf.reshape(y0_grid, [-1])
    x0_flat = tf.reshape(x0_grid, [-1])
    y0f = tf.cast(y0_flat, tf.float32)
    x0f = tf.cast(x0_flat, tf.float32)
    phf = tf.cast(ph, tf.float32)
    pwf = tf.cast(pw, tf.float32)
    disjoint = tf.logical_or(
        tf.logical_or(y0f + phf <= roi_ymin_px, y0f >= roi_ymax_px),
        tf.logical_or(x0f + pwf <= roi_xmin_px, x0f >= roi_xmax_px),
    )
    valid = tf.reshape(disjoint, [-1])
    has_valid = tf.reduce_any(valid)
    first_idx = tf.argmax(tf.cast(valid, tf.int32))
    y0_pick = y0_flat[first_idx]
    x0_pick = x0_flat[first_idx]
    zero = tf.constant(0, dtype=tf.int32)
    return (
        tf.cond(has_valid, lambda: y0_pick, lambda: zero),
        tf.cond(has_valid, lambda: x0_pick, lambda: zero),
    )


def _flip_roi_norm_x(
    xmin: tf.Tensor,
    xmax: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    return 1.0 - xmax, 1.0 - xmin


def _offset_from_center(
    center: tf.Tensor,
    patch_size: int,
    max_offset: tf.Tensor,
    *,
    random_jitter: bool,
    sigma_frac: float,
) -> tf.Tensor:
    max_f = tf.cast(max_offset, tf.float32)
    ideal = tf.clip_by_value(
        center - tf.cast(patch_size, tf.float32) / 2.0,
        0.0,
        max_f,
    )
    if random_jitter:
        # Jitter relativo al tamaño del parche (no al rango de scroll de la imagen):
        # asi la lesion se descentra un poco pero sigue quedando dentro del parche.
        # Se acota a max_f para no proponer offsets fuera de la imagen.
        std = tf.minimum(
            tf.maximum(tf.cast(sigma_frac, tf.float32) * tf.cast(patch_size, tf.float32), 1.0),
            tf.maximum(max_f, 1.0),
        )
        offset = tf.random.normal([]) * std + ideal
        return tf.cast(tf.round(tf.clip_by_value(offset, 0.0, max_f)), tf.int32)
    return tf.cast(tf.round(ideal), tf.int32)


def _dataset_deterministic_options(ds: tf.data.Dataset) -> tf.data.Dataset:
    """Orden estable para eval y alineacion con tablas fuente."""
    opts = tf.data.Options()
    if hasattr(opts, "deterministic"):
        opts.deterministic = True
    elif hasattr(opts, "experimental_deterministic"):
        opts.experimental_deterministic = True
    return ds.with_options(opts)


def _dataset_perf_options(ds: tf.data.Dataset) -> tf.data.Dataset:
    opts = tf.data.Options()
    if hasattr(opts, "deterministic"):
        opts.deterministic = False
    elif hasattr(opts, "experimental_deterministic"):
        opts.experimental_deterministic = False
    return ds.with_options(opts)


@dataclass
class TfDatasetConfig:
    image_size: tuple[int, int]
    batch_size: int
    seed: int = 42
    path_column: str = "path"
    label_column: str = "cls"
    cache_dataset: bool = True
    cache_filename: str | Path | None = None
    use_clahe: bool = False
    clahe_clip_limit: float = 2.0
    clahe_tile_grid: int = 8
    lateralize: bool = False
    laterality_column: str = "laterality"
    patch_mode: bool = False
    patch_sampling: PatchSampling = "uniform"
    patch_bias_x: float = 0.75
    patch_bias_y: float = 0.5
    patch_bias_sigma: float = 0.2
    patch_crop_strategy: PatchCropStrategy | None = None
    patch_crop_by_label: dict[float, PatchCropStrategy] | None = None
    patch_roi_norm_columns: tuple[str, str, str, str] = (
        "pad_resized_xmin_norm",
        "pad_resized_ymin_norm",
        "pad_resized_xmax_norm",
        "pad_resized_ymax_norm",
    )
    patch_roi_sigma_frac: float = 0.1
    patch_avoid_roi_max_attempts: int = 64
    patch_align_to_bag_grid: bool = False
    # Reescala la imagen al canvas del bag (rows*ph x cols*pw) antes de recortar,
    # para que el parche vea la misma escala que los tiles de ABMIL, pero muestreando
    # libremente con las estrategias roi/avoid_roi (sin fijar a la grilla).
    # Default alineado con el notebook (PATCH.RESIZE_TO_BAG_CANVAS).
    patch_resize_to_bag_canvas: bool = True
    positive_mixup: bool = False
    positive_mixup_alpha: float = 0.1
    positive_mixup_probability: float = 0.5
    # Modo MIL (ABMIL): cada imagen es un "bag" troceado en parches.
    mode: str = "simple"
    bag_grid: tuple[int, int] = (3, 3)
    bag_keras_tiling: bool = False
    bag_canvas_mode: BagCanvasMode = "resize"

    def needs_roi_columns(self) -> bool:
        if TrainingMode.is_mil(self.mode):
            return False
        if not self.patch_mode:
            return False
        if self.patch_align_to_bag_grid:
            return True
        # Strategy global explicita: solo exige ROI si esa strategy la usa.
        if self.patch_crop_strategy is not None:
            return self.patch_crop_strategy in ("roi", "avoid_roi")
        by_label = self.patch_crop_by_label or DEFAULT_PATCH_CROP_BY_LABEL
        return by_label.get(1.0) == "roi" or by_label.get(0.0) == "avoid_roi"

    def __post_init__(self) -> None:
        self.image_size = _normalize_size(self.image_size)
        self.bag_grid = (int(self.bag_grid[0]), int(self.bag_grid[1]))
        self.mode = TrainingMode.parse(self.mode)
        self.patch_mode = TrainingMode.is_patch(self.mode)
        if self.bag_canvas_mode not in ("resize", "pad"):
            raise ValueError("bag_canvas_mode debe ser 'resize' o 'pad'")


@dataclass
class DatasetProviderConfig:
    """Configuracion de alto nivel; el notebook arma variantes con `with_overrides`."""

    image_size: tuple[int, int] | int
    batch_size: int
    seed: int = 42
    path_column: str = "path"
    label_column: str = "cls"
    cache_dataset: bool = True
    cache_filename: str | Path | None = None
    use_clahe: bool = False
    clahe_clip_limit: float = 2.0
    clahe_tile_grid: int = 8
    lateralize: bool = False
    laterality_column: str = "laterality"
    lateralize_flip_side: Literal["L", "R"] = "R"
    patch_mode: bool = False
    patch_sampling: PatchSampling = "uniform"
    patch_bias_x: float = 0.75
    patch_bias_y: float = 0.5
    patch_bias_sigma: float = 0.2
    patch_crop_strategy: PatchCropStrategy | None = None
    patch_crop_by_label: dict[float, PatchCropStrategy] | None = None
    patch_roi_norm_columns: tuple[str, str, str, str] = (
        "pad_resized_xmin_norm",
        "pad_resized_ymin_norm",
        "pad_resized_xmax_norm",
        "pad_resized_ymax_norm",
    )
    patch_roi_sigma_frac: float = 0.1
    patch_avoid_roi_max_attempts: int = 64
    patch_align_to_bag_grid: bool = False
    patch_resize_to_bag_canvas: bool = True
    positive_mixup: bool = False
    positive_mixup_alpha: float = 0.1
    positive_mixup_probability: float = 0.5
    mode: str = "simple"
    bag_grid: tuple[int, int] = (3, 3)
    bag_keras_tiling: bool = False
    bag_canvas_mode: BagCanvasMode = "resize"

    def __post_init__(self) -> None:
        self.image_size = _normalize_size(self.image_size)
        self.bag_grid = (int(self.bag_grid[0]), int(self.bag_grid[1]))
        self.mode = TrainingMode.parse(self.mode)
        # Sincroniza siempre: evita patch_mode sticky tras with_overrides(mode=...).
        self.patch_mode = TrainingMode.is_patch(self.mode)
        if self.bag_canvas_mode not in ("resize", "pad"):
            raise ValueError("bag_canvas_mode debe ser 'resize' o 'pad'")

    def with_overrides(self, **kwargs: Any) -> DatasetProviderConfig:
        return replace(self, **kwargs)

    def to_tf_config(self) -> TfDatasetConfig:
        return TfDatasetConfig(
            image_size=self.image_size,
            batch_size=self.batch_size,
            seed=self.seed,
            path_column=self.path_column,
            label_column=self.label_column,
            cache_dataset=self.cache_dataset,
            cache_filename=self.cache_filename,
            use_clahe=self.use_clahe,
            clahe_clip_limit=self.clahe_clip_limit,
            clahe_tile_grid=self.clahe_tile_grid,
            lateralize=self.lateralize,
            laterality_column=self.laterality_column,
            patch_mode=self.patch_mode,
            patch_sampling=self.patch_sampling,
            patch_bias_x=self.patch_bias_x,
            patch_bias_y=self.patch_bias_y,
            patch_bias_sigma=self.patch_bias_sigma,
            patch_crop_strategy=self.patch_crop_strategy,
            patch_crop_by_label=self.patch_crop_by_label,
            patch_roi_norm_columns=self.patch_roi_norm_columns,
            patch_roi_sigma_frac=self.patch_roi_sigma_frac,
            patch_avoid_roi_max_attempts=self.patch_avoid_roi_max_attempts,
            patch_align_to_bag_grid=self.patch_align_to_bag_grid,
            patch_resize_to_bag_canvas=self.patch_resize_to_bag_canvas,
            positive_mixup=self.positive_mixup,
            positive_mixup_alpha=self.positive_mixup_alpha,
            positive_mixup_probability=self.positive_mixup_probability,
            mode=self.mode,
            bag_grid=self.bag_grid,
            bag_keras_tiling=self.bag_keras_tiling,
            bag_canvas_mode=self.bag_canvas_mode,
        )


def hard_negatives_from_positives(
    pos_df: pd.DataFrame,
    *,
    label_column: str = "cls",
) -> pd.DataFrame:
    """Filas positivas reetiquetadas a 0 para muestrear hard negatives.

    Solo cambia la etiqueta. El crop fuera de ROI depende del provider
    (``patch_crop_by_label[0]=avoid_roi`` por defecto, o
    ``patch_align_to_bag_grid``). Un ``patch_crop_strategy="roi"`` global
    anularia esa semantica.
    """
    hard = pos_df[pos_df[label_column] >= 0.5].copy()
    hard[label_column] = 0.0
    return hard


class InspectDataset:
    """Proxy sobre `tf.data.Dataset` para unwrap/orden estable en Keras."""

    def __init__(
        self,
        dataset: tf.data.Dataset,
        *,
        name: str,
        config: TfDatasetConfig,
        row_count: int | None = None,
        ordered_dataset: tf.data.Dataset | None = None,
        source_table: pd.DataFrame | None = None,
        lateralize_flip_side: Literal["L", "R"] = "R",
    ):
        self._dataset = dataset
        self._ordered_dataset = ordered_dataset or dataset
        self.name = name
        self.config = config
        self.row_count = row_count
        self.source_table = source_table
        self.lateralize_flip_side = lateralize_flip_side

    @property
    def dataset(self) -> tf.data.Dataset:
        return self._dataset

    def ordered(self) -> tf.data.Dataset:
        """Misma pipeline sin shuffle; siempre el mismo orden de muestras."""
        return self._ordered_dataset

    def unwrap(self) -> tf.data.Dataset:
        return self._dataset

    def __iter__(self) -> Iterator:
        return iter(self._dataset)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._dataset, name)

    def __repr__(self) -> str:
        rows = f", rows={self.row_count}" if self.row_count is not None else ""
        return f"InspectDataset(name={self.name!r}{rows})"


def as_tf_dataset(dataset: tf.data.Dataset | InspectDataset) -> tf.data.Dataset:
    """Keras solo acepta `tf.data.Dataset`; desenvuelve `InspectDataset` si hace falta."""
    if isinstance(dataset, InspectDataset):
        return dataset.unwrap()
    if isinstance(dataset, DatasetSplits):
        raise TypeError(
            "as_tf_dataset recibio DatasetSplits; pasa splits.train / .val / .test"
        )
    return dataset


@dataclass
class DatasetSplits:
    train: InspectDataset
    val: InspectDataset
    test: InspectDataset

    def __iter__(self) -> Iterator[InspectDataset]:
        yield self.train
        yield self.val
        yield self.test

    def __getitem__(self, name: SplitName) -> InspectDataset:
        return getattr(self, name)

    def items(self) -> tuple[tuple[str, InspectDataset], ...]:
        return (("train", self.train), ("val", self.val), ("test", self.test))

    def unwrap(self) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        return self.train.unwrap(), self.val.unwrap(), self.test.unwrap()


class DatasetProvider:
    """Construye `tf.data.Dataset` a partir de tablas pandas con paths y etiquetas."""

    def __init__(
        self,
        config: TfDatasetConfig,
        *,
        lateralize_flip_side: Literal["L", "R"] = "R",
    ):
        self.config = config
        if self.config.bag_canvas_mode not in ("resize", "pad"):
            raise ValueError("bag_canvas_mode debe ser 'resize' o 'pad'")
        self.lateralize_flip_side = lateralize_flip_side
        self._height, self._width = _normalize_size(config.image_size)

    def _uniform_patch_offsets_branch(
        self,
        img: tf.Tensor,
        path: tf.Tensor,
        *,
        random_patch: bool,
        max_y: tf.Tensor,
        max_x: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        if random_patch:
            return _uniform_patch_offsets(max_y, max_x)
        return self._extract_patch_legacy(
            img, path, random_patch=False, max_y=max_y, max_x=max_x
        )

    def _extract_patch_legacy(
        self,
        img: tf.Tensor,
        path: tf.Tensor,
        *,
        random_patch: bool,
        max_y: tf.Tensor,
        max_x: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        if random_patch:
            y0 = _random_patch_offset(
                max_y,
                sampling=self.config.patch_sampling,
                mean_frac=self.config.patch_bias_y,
                sigma_frac=self.config.patch_bias_sigma,
            )
            x0 = _random_patch_offset(
                max_x,
                sampling=self.config.patch_sampling,
                mean_frac=self.config.patch_bias_x,
                sigma_frac=self.config.patch_bias_sigma,
            )
            return y0, x0
        if self.config.patch_sampling == "normal":
            return (
                _deterministic_patch_offset(max_y, self.config.patch_bias_y),
                _deterministic_patch_offset(max_x, self.config.patch_bias_x),
            )
        bucket = tf.cast(tf.strings.to_hash_bucket_fast(path, 2**31 - 1), tf.int32)
        return bucket % (max_y + 1), (bucket // 7) % (max_x + 1)

    def _offsets_for_label_strategy(
        self,
        strategy: PatchCropStrategy,
        img: tf.Tensor,
        path: tf.Tensor,
        label: tf.Tensor,
        roi_xmin: tf.Tensor,
        roi_ymin: tf.Tensor,
        roi_xmax: tf.Tensor,
        roi_ymax: tf.Tensor,
        *,
        random_patch: bool,
        max_y: tf.Tensor,
        max_x: tf.Tensor,
        ph: int,
        pw: int,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        if strategy == "roi":
            return self._extract_patch_roi(
                img,
                path,
                roi_xmin,
                roi_ymin,
                roi_xmax,
                roi_ymax,
                random_patch=random_patch,
                max_y=max_y,
                max_x=max_x,
                ph=ph,
                pw=pw,
            )
        if strategy == "avoid_roi":
            return self._extract_patch_avoid_roi(
                img,
                path,
                roi_xmin,
                roi_ymin,
                roi_xmax,
                roi_ymax,
                random_patch=random_patch,
                max_y=max_y,
                max_x=max_x,
                ph=ph,
                pw=pw,
            )
        if strategy == "normal":
            return self._extract_patch_legacy(
                img, path, random_patch=random_patch, max_y=max_y, max_x=max_x
            )
        return self._uniform_patch_offsets_branch(
            img, path, random_patch=random_patch, max_y=max_y, max_x=max_x
        )

    def _extract_patch_avoid_roi(
        self,
        img: tf.Tensor,
        path: tf.Tensor,
        roi_xmin: tf.Tensor,
        roi_ymin: tf.Tensor,
        roi_xmax: tf.Tensor,
        roi_ymax: tf.Tensor,
        *,
        random_patch: bool,
        max_y: tf.Tensor,
        max_x: tf.Tensor,
        ph: int,
        pw: int,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        h = tf.shape(img)[0]
        w = tf.shape(img)[1]
        valid = _roi_is_valid(roi_xmin, roi_ymin, roi_xmax, roi_ymax)

        def _sample_avoiding() -> tuple[tf.Tensor, tf.Tensor]:
            return _offsets_avoiding_roi(
                max_y,
                max_x,
                ph,
                pw,
                roi_xmin,
                roi_ymin,
                roi_xmax,
                roi_ymax,
                h,
                w,
                random_patch=random_patch,
                max_attempts=self.config.patch_avoid_roi_max_attempts,
                path=path,
            )

        return tf.cond(
            valid,
            _sample_avoiding,
            lambda: self._uniform_patch_offsets_branch(
                img, path, random_patch=random_patch, max_y=max_y, max_x=max_x
            ),
        )

    def _extract_patch_roi(
        self,
        img: tf.Tensor,
        path: tf.Tensor,
        roi_xmin: tf.Tensor,
        roi_ymin: tf.Tensor,
        roi_xmax: tf.Tensor,
        roi_ymax: tf.Tensor,
        *,
        random_patch: bool,
        max_y: tf.Tensor,
        max_x: tf.Tensor,
        ph: int,
        pw: int,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        valid = _roi_is_valid(roi_xmin, roi_ymin, roi_xmax, roi_ymax)

        def _sample_roi() -> tuple[tf.Tensor, tf.Tensor]:
            h0 = tf.cast(tf.shape(img)[0], tf.float32)
            w0 = tf.cast(tf.shape(img)[1], tf.float32)
            xmin = roi_xmin * w0
            ymin = roi_ymin * h0
            xmax = roi_xmax * w0
            ymax = roi_ymax * h0
            cx = (xmin + xmax) / 2.0
            cy = (ymin + ymax) / 2.0
            return (
                _offset_from_center(
                    cy,
                    ph,
                    max_y,
                    random_jitter=random_patch,
                    sigma_frac=self.config.patch_roi_sigma_frac,
                ),
                _offset_from_center(
                    cx,
                    pw,
                    max_x,
                    random_jitter=random_patch,
                    sigma_frac=self.config.patch_roi_sigma_frac,
                ),
            )

        return tf.cond(
            valid,
            _sample_roi,
            lambda: self._uniform_patch_offsets_branch(
                img, path, random_patch=random_patch, max_y=max_y, max_x=max_x
            ),
        )

    def _bag_aligned_tile_offsets(
        self,
        path: tf.Tensor,
        label: tf.Tensor,
        roi_xmin: tf.Tensor,
        roi_ymin: tf.Tensor,
        roi_xmax: tf.Tensor,
        roi_ymax: tf.Tensor,
        *,
        random_patch: bool,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """Elige un tile exacto de la grilla MIL, usando ROI solo como supervision."""
        rows, cols = self.config.bag_grid
        ph, pw = self._height, self._width
        n_tiles = rows * cols
        tile_indices = tf.range(n_tiles, dtype=tf.int32)
        tile_rows = tile_indices // cols
        tile_cols = tile_indices % cols

        tile_ymin = tf.cast(tile_rows * ph, tf.float32)
        tile_xmin = tf.cast(tile_cols * pw, tf.float32)
        tile_ymax = tile_ymin + tf.cast(ph, tf.float32)
        tile_xmax = tile_xmin + tf.cast(pw, tf.float32)
        canvas_h = tf.cast(rows * ph, tf.float32)
        canvas_w = tf.cast(cols * pw, tf.float32)
        xmin = tf.clip_by_value(roi_xmin * canvas_w, 0.0, canvas_w)
        ymin = tf.clip_by_value(roi_ymin * canvas_h, 0.0, canvas_h)
        xmax = tf.clip_by_value(roi_xmax * canvas_w, 0.0, canvas_w)
        ymax = tf.clip_by_value(roi_ymax * canvas_h, 0.0, canvas_h)
        overlap_w = tf.maximum(0.0, tf.minimum(tile_xmax, xmax) - tf.maximum(tile_xmin, xmin))
        overlap_h = tf.maximum(0.0, tf.minimum(tile_ymax, ymax) - tf.maximum(tile_ymin, ymin))
        overlap = overlap_w * overlap_h
        roi_valid = _roi_is_valid(roi_xmin, roi_ymin, roi_xmax, roi_ymax)

        def _sample_index(candidates: tf.Tensor) -> tf.Tensor:
            count = tf.shape(candidates)[0]

            def _random() -> tf.Tensor:
                position = tf.random.uniform([], maxval=count, dtype=tf.int32)
                return candidates[position]

            def _deterministic() -> tf.Tensor:
                bucket = tf.cast(
                    tf.strings.to_hash_bucket_fast(path, 2**31 - 1), tf.int32
                )
                return candidates[bucket % count]

            return _random() if random_patch else _deterministic()

        def _positive_index() -> tf.Tensor:
            return tf.cond(
                roi_valid,
                lambda: tf.argmax(overlap, output_type=tf.int32),
                lambda: _sample_index(tile_indices),
            )

        def _negative_index() -> tf.Tensor:
            non_roi_tiles = tf.boolean_mask(tile_indices, overlap <= 0.0)
            candidates = tf.cond(
                tf.logical_and(roi_valid, tf.size(non_roi_tiles) > 0),
                lambda: non_roi_tiles,
                lambda: tile_indices,
            )
            return _sample_index(candidates)

        tile_index = tf.cond(label >= 0.5, _positive_index, _negative_index)
        return (tile_index // cols) * ph, (tile_index % cols) * pw

    def _bag_canvas(self, img: tf.Tensor) -> tf.Tensor:
        rows, cols = self.config.bag_grid
        ph, pw = self._height, self._width
        canvas_h, canvas_w = rows * ph, cols * pw
        if self.config.bag_canvas_mode == "pad":
            h = tf.shape(img)[0]
            w = tf.shape(img)[1]
            checks = [
                tf.debugging.assert_less_equal(
                    h,
                    canvas_h,
                    message="La altura excede el canvas MIL; aumenta bag_grid.",
                ),
                tf.debugging.assert_less_equal(
                    w,
                    canvas_w,
                    message="El ancho excede el canvas MIL; aumenta bag_grid.",
                ),
            ]
            with tf.control_dependencies(checks):
                pad_top = (canvas_h - h) // 2
                pad_left = (canvas_w - w) // 2
                full = tf.image.pad_to_bounding_box(
                    img,
                    pad_top,
                    pad_left,
                    canvas_h,
                    canvas_w,
                )
        else:
            full = _resize_preserving_dtype(img, [canvas_h, canvas_w])
        if self.config.use_clahe:
            full = ImageDecoder.apply_clahe(
                full,
                self.config.clahe_clip_limit,
                self.config.clahe_tile_grid,
            )
        full.set_shape([canvas_h, canvas_w, 3])
        return full

    def _roi_norms_on_bag_canvas(
        self,
        img: tf.Tensor,
        roi_xmin: tf.Tensor,
        roi_ymin: tf.Tensor,
        roi_xmax: tf.Tensor,
        roi_ymax: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        if self.config.bag_canvas_mode != "pad":
            return roi_xmin, roi_ymin, roi_xmax, roi_ymax
        rows, cols = self.config.bag_grid
        ph, pw = self._height, self._width
        canvas_h = tf.cast(rows * ph, tf.float32)
        canvas_w = tf.cast(cols * pw, tf.float32)
        h = tf.cast(tf.shape(img)[0], tf.float32)
        w = tf.cast(tf.shape(img)[1], tf.float32)
        pad_top = tf.floor((canvas_h - h) / 2.0)
        pad_left = tf.floor((canvas_w - w) / 2.0)
        return (
            (roi_xmin * w + pad_left) / canvas_w,
            (roi_ymin * h + pad_top) / canvas_h,
            (roi_xmax * w + pad_left) / canvas_w,
            (roi_ymax * h + pad_top) / canvas_h,
        )

    def _extract_patch(
        self,
        img: tf.Tensor,
        path: tf.Tensor,
        label: tf.Tensor,
        roi_xmin: tf.Tensor,
        roi_ymin: tf.Tensor,
        roi_xmax: tf.Tensor,
        roi_ymax: tf.Tensor,
        *,
        random_patch: bool,
    ) -> tf.Tensor:
        ph, pw = self._height, self._width
        if self.config.patch_align_to_bag_grid or self.config.patch_resize_to_bag_canvas:
            roi_xmin, roi_ymin, roi_xmax, roi_ymax = self._roi_norms_on_bag_canvas(
                img,
                roi_xmin,
                roi_ymin,
                roi_xmax,
                roi_ymax,
            )
            img = self._bag_canvas(img)
        else:
            img = _ensure_min_spatial(img, ph, pw)
        h = tf.shape(img)[0]
        w = tf.shape(img)[1]
        max_y = tf.maximum(h - ph, 0)
        max_x = tf.maximum(w - pw, 0)
        strategy = self.config.patch_crop_strategy

        if self.config.patch_align_to_bag_grid:
            y0, x0 = self._bag_aligned_tile_offsets(
                path,
                label,
                roi_xmin,
                roi_ymin,
                roi_xmax,
                roi_ymax,
                random_patch=random_patch,
            )
        elif strategy == "roi":
            y0, x0 = self._extract_patch_roi(
                img,
                path,
                roi_xmin,
                roi_ymin,
                roi_xmax,
                roi_ymax,
                random_patch=random_patch,
                max_y=max_y,
                max_x=max_x,
                ph=ph,
                pw=pw,
            )
        elif strategy == "uniform":
            y0, x0 = self._uniform_patch_offsets_branch(
                img, path, random_patch=random_patch, max_y=max_y, max_x=max_x
            )
        elif strategy == "avoid_roi":
            y0, x0 = self._extract_patch_avoid_roi(
                img,
                path,
                roi_xmin,
                roi_ymin,
                roi_xmax,
                roi_ymax,
                random_patch=random_patch,
                max_y=max_y,
                max_x=max_x,
                ph=ph,
                pw=pw,
            )
        elif strategy == "normal":
            y0, x0 = self._extract_patch_legacy(
                img, path, random_patch=random_patch, max_y=max_y, max_x=max_x
            )
        else:
            by_label = self.config.patch_crop_by_label or DEFAULT_PATCH_CROP_BY_LABEL
            is_positive = label >= 0.5

            def _positive_offsets() -> tuple[tf.Tensor, tf.Tensor]:
                return self._offsets_for_label_strategy(
                    by_label.get(1.0, "uniform"),
                    img,
                    path,
                    label,
                    roi_xmin,
                    roi_ymin,
                    roi_xmax,
                    roi_ymax,
                    random_patch=random_patch,
                    max_y=max_y,
                    max_x=max_x,
                    ph=ph,
                    pw=pw,
                )

            def _negative_offsets() -> tuple[tf.Tensor, tf.Tensor]:
                return self._offsets_for_label_strategy(
                    by_label.get(0.0, "uniform"),
                    img,
                    path,
                    label,
                    roi_xmin,
                    roi_ymin,
                    roi_xmax,
                    roi_ymax,
                    random_patch=random_patch,
                    max_y=max_y,
                    max_x=max_x,
                    ph=ph,
                    pw=pw,
                )

            y0, x0 = tf.cond(is_positive, _positive_offsets, _negative_offsets)
        y0 = tf.minimum(y0, max_y)
        x0 = tf.minimum(x0, max_x)
        img = tf.image.crop_to_bounding_box(img, y0, x0, ph, pw)
        img.set_shape([ph, pw, 3])
        return img

    def _make_bag(self, img: tf.Tensor) -> tf.Tensor:
        """Lleva la imagen al canvas y, opcionalmente, la trocea en una grilla.

        Con bag_keras_tiling=False (defecto): devuelve (K, ph, pw, 3) —
        el tiling ocurre aqui en tf.data, la augmentacion se aplica por tile.

        Con bag_keras_tiling=True: devuelve (rows*ph, cols*pw, 3) —
        el tiling lo realiza la capa BagTiling dentro del modelo Keras,
        permitiendo que la augmentacion se aplique sobre la imagen completa (1x).
        """
        rows, cols = self.config.bag_grid
        ph, pw = self._height, self._width
        full = self._bag_canvas(img)
        if self.config.bag_keras_tiling:
            return full
        tiles = tf.reshape(full, [rows, ph, cols, pw, 3])
        tiles = tf.transpose(tiles, [0, 2, 1, 3, 4])
        bag = tf.reshape(tiles, [rows * cols, ph, pw, 3])
        bag.set_shape([rows * cols, ph, pw, 3])
        return bag

    def _roi_norm_tensors(self, tbl: pd.DataFrame) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        xmin_c, ymin_c, xmax_c, ymax_c = self.config.patch_roi_norm_columns
        missing = [
            column
            for column in self.config.patch_roi_norm_columns
            if column not in tbl.columns
        ]
        if missing:
            raise KeyError(
                "Las columnas ROI son necesarias para crop en hallazgos: "
                f"{missing}"
            )
        return (
            tf.constant(tbl[xmin_c].astype(np.float32).values),
            tf.constant(tbl[ymin_c].astype(np.float32).values),
            tf.constant(tbl[xmax_c].astype(np.float32).values),
            tf.constant(tbl[ymax_c].astype(np.float32).values),
        )

    def _process(
        self,
        path: tf.Tensor,
        label: tf.Tensor,
        roi_xmin: tf.Tensor,
        roi_ymin: tf.Tensor,
        roi_xmax: tf.Tensor,
        roi_ymax: tf.Tensor,
        flip_lateral: tf.Tensor | None = None,
        *,
        random_patch: bool = False,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        img = ImageDecoder.decode(path)
        if flip_lateral is not None:
            flip = flip_lateral

            def _flip_image() -> tf.Tensor:
                return tf.image.flip_left_right(img)

            img = tf.cond(flip, _flip_image, lambda: img)
            if self.config.needs_roi_columns():
                flipped_xmin, flipped_xmax = _flip_roi_norm_x(roi_xmin, roi_xmax)
                roi_xmin = tf.where(flip, flipped_xmin, roi_xmin)
                roi_xmax = tf.where(flip, flipped_xmax, roi_xmax)
        if TrainingMode.is_mil(self.config.mode):
            return self._make_bag(img), label
        if self.config.patch_mode:
            img = self._extract_patch(
                img,
                path,
                label,
                roi_xmin,
                roi_ymin,
                roi_xmax,
                roi_ymax,
                random_patch=random_patch,
            )
        else:
            img = _resize_preserving_dtype(img, self.config.image_size)
        if self.config.use_clahe and not (
            self.config.patch_mode
            and (self.config.patch_align_to_bag_grid or self.config.patch_resize_to_bag_canvas)
        ):
            img = ImageDecoder.apply_clahe(
                img,
                self.config.clahe_clip_limit,
                self.config.clahe_tile_grid,
            )
            img.set_shape([self._height, self._width, 3])
        return img, label

    def _base_dataset(self, tbl: pd.DataFrame) -> tf.data.Dataset:
        paths = tf.constant(tbl[self.config.path_column].values)
        labels = tf.constant(tbl[self.config.label_column].values.astype(np.float32))
        roi_tensors: tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor] | tuple[()] = ()
        if self.config.needs_roi_columns():
            roi_tensors = self._roi_norm_tensors(tbl)
        elements: tuple[Any, ...] = (paths, labels, *roi_tensors)
        if not self.config.lateralize:
            return tf.data.Dataset.from_tensor_slices(elements)

        flip_lateral = tf.constant(
            tbl[self.config.laterality_column].astype(str).values
            == self.lateralize_flip_side,
            dtype=tf.bool,
        )
        return tf.data.Dataset.from_tensor_slices((*elements, flip_lateral))

    def _wrap(
        self,
        dataset: tf.data.Dataset,
        *,
        name: str,
        row_count: int,
        ordered_dataset: tf.data.Dataset | None = None,
        source_table: pd.DataFrame | None = None,
    ) -> InspectDataset:
        return InspectDataset(
            dataset,
            name=name,
            config=self.config,
            row_count=row_count,
            ordered_dataset=ordered_dataset,
            source_table=source_table,
            lateralize_flip_side=self.lateralize_flip_side,
        )

    def _make_process_fn(self, *, random_patch: bool):
        with_roi = self.config.needs_roi_columns()

        if self.config.lateralize:

            def process_with_laterality(
                path: tf.Tensor,
                label: tf.Tensor,
                *extra: tf.Tensor,
            ) -> tuple[tf.Tensor, tf.Tensor]:
                if with_roi:
                    roi_xmin, roi_ymin, roi_xmax, roi_ymax = extra[:4]
                    flip_lateral = extra[4]
                else:
                    roi_xmin = roi_ymin = roi_xmax = roi_ymax = tf.constant(0.0, tf.float32)
                    flip_lateral = extra[0]
                return self._process(
                    path,
                    label,
                    roi_xmin,
                    roi_ymin,
                    roi_xmax,
                    roi_ymax,
                    flip_lateral=flip_lateral,
                    random_patch=random_patch,
                )

            return process_with_laterality

        def process(path: tf.Tensor, label: tf.Tensor, *extra: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
            if with_roi:
                roi_xmin, roi_ymin, roi_xmax, roi_ymax = extra[:4]
            else:
                roi_xmin = roi_ymin = roi_xmax = roi_ymax = tf.constant(0.0, tf.float32)
            return self._process(
                path,
                label,
                roi_xmin,
                roi_ymin,
                roi_xmax,
                roi_ymax,
                random_patch=random_patch,
            )

        return process

    def build(
        self,
        tbl: pd.DataFrame,
        *,
        shuffle: bool,
        random_patch: bool | None = None,
        name: str = "dataset",
    ) -> InspectDataset:
        if len(tbl) == 0:
            raise ValueError(f"No se puede construir el dataset {name!r}: tabla vacia")
        if random_patch is None:
            random_patch = shuffle

        base = self._base_dataset(tbl)
        processed = base.map(
            self._make_process_fn(random_patch=random_patch),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

        # En patch + crop aleatorio, cache() congelaria el primer crop por epoca.
        cache_random_crops = self.config.patch_mode and random_patch
        if not cache_random_crops:
            if self.config.cache_filename is not None:
                cache_path = Path(self.config.cache_filename)
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                processed = processed.cache(str(cache_path))
            elif self.config.cache_dataset:
                processed = processed.cache()

        if self.config.patch_mode and shuffle:
            ordered_processed = base.map(
                self._make_process_fn(random_patch=False),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
            if self.config.cache_filename is not None:
                ordered_cache = Path(str(self.config.cache_filename) + ".ordered")
                ordered_cache.parent.mkdir(parents=True, exist_ok=True)
                ordered_processed = ordered_processed.cache(str(ordered_cache))
            elif self.config.cache_dataset:
                ordered_processed = ordered_processed.cache()
        else:
            ordered_processed = processed

        ordered_batched = _dataset_deterministic_options(
            ordered_processed.batch(self.config.batch_size).prefetch(tf.data.AUTOTUNE)
        )

        if shuffle:
            processed = processed.shuffle(
                len(tbl),
                seed=self.config.seed,
                reshuffle_each_iteration=True,
            )
            batched = _dataset_perf_options(
                processed.batch(self.config.batch_size).prefetch(tf.data.AUTOTUNE)
            )
        else:
            batched = ordered_batched

        if shuffle and self.config.positive_mixup:
            alpha = float(self.config.positive_mixup_alpha)
            if alpha <= 0:
                raise ValueError(
                    f"positive_mixup_alpha debe ser > 0; recibido {alpha!r}"
                )
            probability = float(self.config.positive_mixup_probability)

            def _mixup_map(images: tf.Tensor, labels: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
                return PositiveMixup.apply(
                    images,
                    labels,
                    alpha=alpha,
                    probability=probability,
                )

            batched = batched.map(_mixup_map, num_parallel_calls=tf.data.AUTOTUNE)
            batched = _dataset_perf_options(batched.prefetch(tf.data.AUTOTUNE))

        return self._wrap(
            batched,
            name=name,
            row_count=len(tbl),
            ordered_dataset=ordered_batched,
            source_table=tbl,
        )

    def build_train(self, tbl: pd.DataFrame) -> InspectDataset:
        return self.build(tbl, shuffle=True, name="train")

    def build_eval(self, tbl: pd.DataFrame, *, name: str = "eval") -> InspectDataset:
        return self.build(tbl, shuffle=False, name=name)

    def build_splits(
        self,
        train: pd.DataFrame,
        val: pd.DataFrame,
        test: pd.DataFrame,
    ) -> DatasetSplits:
        return DatasetSplits(
            train=self.build_train(train),
            val=self.build_eval(val, name="val"),
            test=self.build_eval(test, name="test"),
        )


def build_dataset_provider(
    config: dict[str, Any] | None = None,
    image_size: tuple[int, int] | int | None = None,
    batch_size: int | None = None,
    *,
    provider_config: DatasetProviderConfig | None = None,
    **kwargs: Any,
) -> DatasetProvider:
    """Crea un DatasetProvider.

    Formas de uso:
    - ``build_dataset_provider(provider_config=DatasetProviderConfig(...))``: bajo nivel.
    - ``build_dataset_provider(CONFIG, image_size, batch_size, mode=..., ...)``: toma
      seed/use_clahe/bag_* del CONFIG del notebook; los kwargs explicitos los pisan.
    - ``build_dataset_provider(None, image_size, batch_size, seed=..., ...)``: sin CONFIG,
      todo por kwargs.
    """
    if provider_config is None:
        if image_size is None or batch_size is None:
            raise ValueError(
                "Indica `provider_config=DatasetProviderConfig(...)` o los argumentos "
                "image_size y batch_size."
            )
        if config is not None:
            general = config["GENERAL"]
            mil = config["MIL"]
            full_cfg = config.get("FULL") or {}
            patch_cfg = config.get("PATCH") or {}
            defaults = {
                "seed": general["RANDOM_SEED"],
                "use_clahe": general["USE_CLAHE"],
                "bag_grid": full_cfg.get("BAG_GRID", (3, 3)),
                "bag_keras_tiling": mil["BAG_KERAS_TILING"],
                "bag_canvas_mode": full_cfg.get("BAG_CANVAS_MODE", "resize"),
                "patch_resize_to_bag_canvas": patch_cfg.get("RESIZE_TO_BAG_CANVAS", True),
            }
            # Los kwargs explicitos pisan los defaults del CONFIG.
            kwargs = {**defaults, **kwargs}
        if "mode" in kwargs:
            kwargs["mode"] = TrainingMode.parse(kwargs["mode"])
        provider_config = DatasetProviderConfig(
            image_size=image_size,
            batch_size=batch_size,
            **kwargs,
        )
    elif kwargs or config is not None:
        raise ValueError("Pasa solo `provider_config` o (config + kwargs sueltos), no ambos.")
    return DatasetProvider(
        provider_config.to_tf_config(),
        lateralize_flip_side=provider_config.lateralize_flip_side,
    )
