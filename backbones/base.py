from __future__ import annotations

from typing import Callable, ClassVar

from tensorflow import keras

ModelFactory = Callable[..., keras.Model]
PreprocessFunction = Callable
InputSize = tuple[int, int]

DEFAULT_WEIGHTS = object()
_REGISTRY: dict[str, Backbone] = {}
BACKBONES = _REGISTRY


class Backbone:
    key: ClassVar[str]
    input_size: ClassVar[InputSize]
    default_weights: ClassVar[str | None] = None
    application: ClassVar[ModelFactory]
    preprocess_fn: ClassVar[PreprocessFunction]

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        if "key" in cls.__dict__:
            _REGISTRY[cls.key] = cls()

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
