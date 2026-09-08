from __future__ import annotations

from typing import Callable, ClassVar

from tensorflow import keras

ModelFactory = Callable[..., keras.Model]
PreprocessFunction = Callable
InputSize = tuple[int, int]

DEFAULT_WEIGHTS = object()
_REGISTRY: dict[str, Backbone] = {}
BACKBONES = _REGISTRY

# A100 80GB + mixed_float16, etapa 3 (backbone entero). Los valores son el
# maximo que entra sin OOM / sin dejar el GPU idle en la geometria de
# referencia: SIMPLE/PATCH a S nativo, FULL a 672, ABMIL a grilla 3×3.
# ``resolve_batch_size`` escala si FULL o BAG_GRID cambian.
_DEFAULT_BATCH_SIZE = {
    "simple": 256,
    "full": 128,
    "patch": 256,
    "patch_hardneg": 256,
    "abmil": 64,
    "abmil_patch_hardneg": 64,
}


def mode_batch_sizes(
    *,
    simple: int,
    full: int,
    abmil: int,
    patch: int | None = None,
    patch_hardneg: int | None = None,
    abmil_patch_hardneg: int | None = None,
) -> dict[str, int]:
    """Batch de referencia por modo (A100 80GB; SIMPLE a S, FULL a 672, ABMIL 3×3).

    PATCH sigue a SIMPLE y ABMIL_PATCH_HARDNEG a ABMIL si no se pasan.
    """
    patch_bs = simple if patch is None else patch
    return {
        "simple": int(simple),
        "full": int(full),
        "patch": int(patch_bs),
        "patch_hardneg": int(patch_bs if patch_hardneg is None else patch_hardneg),
        "abmil": int(abmil),
        "abmil_patch_hardneg": int(abmil if abmil_patch_hardneg is None else abmil_patch_hardneg),
    }


class Backbone:
    key: ClassVar[str]
    input_size: ClassVar[InputSize]
    default_weights: ClassVar[str | None] = None
    application: ClassVar[ModelFactory]
    preprocess_fn: ClassVar[PreprocessFunction]
    batch_size: ClassVar[dict[str, int]] = dict(_DEFAULT_BATCH_SIZE)
    # "spatial": activaciones ~ H×W (CNN, Swin). "attention": ~ (H×W)^2 (ViT FULL).
    batch_memory_scale: ClassVar[str] = "spatial"

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        if "key" in cls.__dict__:
            _REGISTRY[cls.key] = cls()

    def batch_size_for(self, mode: str) -> int | None:
        """Batch calibrado para ``mode``, o None si el provider no lo declara."""
        sizes = self.batch_size or {}
        compact = mode.strip().lower().replace("-", "").replace("_", "")
        for name, value in sizes.items():
            if str(name).strip().lower().replace("-", "").replace("_", "") == compact:
                return int(value)
        return None

    def preprocess_input(self, x):
        return self.__class__.preprocess_fn(x)

    def build(self, *, weights=DEFAULT_WEIGHTS, include_top: bool = False, input_shape: tuple[int, int, int] | None = None, **kwargs) -> keras.Model:
        return self.__class__.application(
            weights=self.coalesce_weights(weights),
            include_top=include_top,
            input_shape=self.input_shape_or_default(input_shape),
            **kwargs,
        )

    def resolve(self, input_size: InputSize | None = None) -> tuple[keras.Model, PreprocessFunction, InputSize]:
        size = input_size or self.input_size
        h, w = size
        return self.build(input_shape=(h, w, 3)), self.preprocess_input, size

    def coalesce_weights(self, weights) -> str | None:
        return self.default_weights if weights is DEFAULT_WEIGHTS else weights

    def input_shape_or_default(self, shape: tuple[int, int, int] | None) -> tuple[int, int, int]:
        h, w = self.input_size
        return shape or (h, w, 3)


class ImagenetBackbone(Backbone):
    default_weights = "imagenet"


def get_backbone(name: str) -> Backbone:
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(sorted(_REGISTRY))
        raise ValueError(
            f"Backbone {name!r} no disponible. Opciones: {available}"
        ) from exc


def resolve_backbone(name: str, input_size: InputSize | None = None) -> tuple[keras.Model, PreprocessFunction, InputSize]:
    return get_backbone(name).resolve(input_size)
