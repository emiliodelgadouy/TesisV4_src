from typing import override

from src.model_builder.image import ImageClassifierBuilder
from src.training.mode import (
    TrainingMode,
    resolve_resized_batch_size,
    resolve_resized_input_size,
    resolve_resized_input_sizes,
    resized_size_label,
)


class ResizedModelBuilder(ImageClassifierBuilder):
    # Mismo grafo que STANDARD, pero el canvas sale de la lista discreta
    # CONFIG["RESIZED"]["INPUT_SIZES"] (o 3× nativo si no hay lista).
    model_name = "resized"
    cache_by_default = False

    @classmethod
    @override
    def resolve_input_size(cls, config, *, native_size, requested=None, mode_name_raw=None):
        named = TrainingMode.parse_resized_size(mode_name_raw) if mode_name_raw else None
        if named is not None and requested is not None:
            chosen = resolve_resized_input_size(config, native_size=native_size, requested=requested)
            if chosen != named:
                raise ValueError(f"Conflicto RESIZED: el modo {mode_name_raw!r} pide {named} pero input_size={chosen}")
            return chosen
        return resolve_resized_input_size(config, native_size=native_size, requested=requested or named)

    @classmethod
    @override
    def experiment_name(cls, backbone_name, *, mode, input_size, suffix=None) -> str:
        del mode
        name = f"resized_{resized_size_label(input_size)}_{backbone_name}"
        if suffix:
            return f"{name}_{suffix}"
        return name

    @classmethod
    @override
    def extra_run_config(cls, config, *, native_size, input_size) -> dict:
        sizes = resolve_resized_input_sizes(config, native_size=native_size)
        extra = {
            "NATIVE_INPUT_SIZE": list(native_size),
            "RESIZED_INPUT_SIZES": [list(size) for size in sizes],
        }
        if dict(config.get("RESIZED") or {}).get("BATCH_SIZES") is not None:
            extra["RESIZED_BATCH_SIZES"] = {
                resized_size_label(size): resolve_resized_batch_size(config, size) for size in sizes
            }
            extra["RESIZED_BATCH_SIZE"] = resolve_resized_batch_size(config, input_size)
        return extra

    @override
    def keras_model_name(self) -> str:
        return f"resized_{resized_size_label(self.IMG_SIZE)}"
