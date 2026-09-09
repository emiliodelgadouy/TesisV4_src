from typing import override

from src.model_builder.image import ImageClassifierBuilder


class StandardModelBuilder(ImageClassifierBuilder):
    # Clasificador a resolucion nativa del backbone. No interpola canvas:
    # IMG_SIZE == backbone.input_size.
    model_name = "standard"
    cache_by_default = True

    @classmethod
    @override
    def resolve_input_size(cls, config, *, native_size, requested=None, mode_name_raw=None):
        if requested is not None:
            raise ValueError("input_size solo aplica al modo resized, no a 'standard'")
        return super().resolve_input_size(config, native_size=native_size, requested=None, mode_name_raw=mode_name_raw)
