from __future__ import annotations

from typing import Literal

import pandas as pd

from src.dataset.config import (
    ALL_FINDING_COLUMNS,
    DEFAULT_CLS_POSITIVE_COLUMNS,
    DEFAULT_ROI_NORM_COLUMNS,
)
from src.dataset.repository import DatasetRepository
from src.dataset.target import TargetMode


class DatasetPreparer:
    """Descarga, etiqueta binaria segun TARGET_MODE y deduplica por imagen."""

    def prepare(self, config) -> pd.DataFrame:
        """Une descarga + columna ``cls`` con el recorte positivo/negativo del objetivo."""
        data = DatasetRepository.download_and_build_dataset(config)
        return self.build_positive_negative(config, data["ds"])

    def build_positive_negative(self, config, ds: pd.DataFrame) -> pd.DataFrame:
        """Arma el dataset binario positivos/negativos segun ``TARGET_MODE``.

        - "mass": positivos = imagenes con ``Mass==1`` (tarea "deteccion de masas").
          Las imagenes con otro hallazgo pero sin masa se DESCARTAN (no son ni
          positivo ni negativo limpio).
        - "full": cualquier hallazgo cuenta como positivo (``cls==1``); no se descarta
          ninguna fila (positivos U negativos = dataset completo).

        Deduplica por imagen, fuerza ``cls`` a 1.0/0.0 y quita de los negativos las
        imagenes que ya aparecen como positivas (mismo patient_id/image_id).
        """
        target = TargetMode.from_config(config)
        positive_mask, negative_mask = target.masks(ds)

        ds_finding = self.deduplicate_images(ds[positive_mask].copy())
        ds_no_finding = self.deduplicate_images(ds[negative_mask].copy())
        ds_finding["cls"] = 1.0
        ds_no_finding["cls"] = 0.0

        key = ["patient_id", "image_id"]
        ds_no_finding = (
            ds_no_finding.merge(ds_finding[key], on=key, how="left", indicator=True)
            .query("_merge == 'left_only'")
            .drop(columns="_merge")
        )
        ds = pd.concat([ds_finding, ds_no_finding], ignore_index=True)
        print(
            f"TARGET_MODE={target}: {len(ds_finding)} positivos / "
            f"{len(ds_no_finding)} negativos ({len(ds)} total)"
        )
        return ds

    @staticmethod
    def deduplicate_images(
        df: pd.DataFrame,
        *,
        patient_id_column: str = "patient_id",
        image_id_column: str = "image_id",
        path_column: str = "path",
        label_column: str = "cls",
        flag_columns: tuple[str, ...] | None = None,
        roi_norm_columns: tuple[str, str, str, str] = DEFAULT_ROI_NORM_COLUMNS,
        keep: Literal["first", "last"] = "first",
    ) -> pd.DataFrame:
        """Una fila por imagen: colapsa hallazgos/cajas multiples del CSV.

        Las columnas ROI se agregan como union del bounding box (min en mins, max
        en maxs) para que avoid_roi evite todas las anotaciones, no solo la primera.
        """
        if df.empty:
            return df.copy()

        key_cols = [patient_id_column, image_id_column]
        missing = [c for c in key_cols if c not in df.columns]
        if missing:
            raise KeyError(f"Missing columns for deduplication: {missing}")

        out = df.copy()
        n_before = len(out)

        if flag_columns is None:
            flag_columns = tuple(
                c
                for c in (*ALL_FINDING_COLUMNS, *DEFAULT_CLS_POSITIVE_COLUMNS)
                if c in out.columns
            )

        if label_column in out.columns:
            conflicts = (
                out.groupby(key_cols, dropna=False)[label_column].nunique().gt(1).sum()
            )
            if conflicts:
                raise ValueError(
                    f"{conflicts} images have inconsistent {label_column} before deduplication."
                )

        agg: dict[str, str] = {
            patient_id_column: "first",
            image_id_column: "first",
        }
        if path_column in out.columns:
            agg[path_column] = "first"
        if label_column in out.columns:
            agg[label_column] = "max"
        for col in flag_columns:
            if col in out.columns:
                agg[col] = "max"

        xmin_c, ymin_c, xmax_c, ymax_c = roi_norm_columns
        if xmin_c in out.columns:
            agg[xmin_c] = "min"
        if ymin_c in out.columns:
            agg[ymin_c] = "min"
        if xmax_c in out.columns:
            agg[xmax_c] = "max"
        if ymax_c in out.columns:
            agg[ymax_c] = "max"

        for col in out.columns:
            if col not in agg:
                agg[col] = keep

        out = (
            out.groupby(key_cols, dropna=False, as_index=False)
            .agg(agg)
            .reset_index(drop=True)
        )

        n_removed = n_before - len(out)
        if n_removed:
            print(
                f"deduplicate_images: {n_before} -> {len(out)} rows "
                f"({n_removed} duplicates removed)"
            )

        return out
