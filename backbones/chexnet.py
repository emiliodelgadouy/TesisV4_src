from __future__ import annotations

from pathlib import Path

import gdown
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess

from .base import Backbone, DEFAULT_WEIGHTS
from .medical_weights import medical_weights_dir

def download_chexnet_weights() -> Path:
    cache_dir = medical_weights_dir() / "chexnet"
    cache_dir.mkdir(parents=True, exist_ok=True)
    weights_path = cache_dir / "CheXNet_weights.h5"
    if not weights_path.is_file():
        gdown.download(id="19BllaOvs2x5PLV_vlWMy4i8LapLb2j6b", output=str(weights_path), quiet=False)
    return weights_path


class CheXNetBackbone(Backbone):
    key = "chexnet"
    input_size = (224, 224)
    default_weights = "chexnet"

    def preprocess_input(self, x):
        return densenet_preprocess(x)

    def build(self, *, weights=DEFAULT_WEIGHTS, include_top: bool = False, input_shape: tuple[int, int, int] | None = None, **kwargs) -> keras.Model:
        if self.coalesce_weights(weights) != "chexnet":
            raise ValueError("CheXNet solo admite weights='chexnet'")
        shape = self.input_shape_or_default(input_shape)
        backbone = DenseNet121(weights=None, include_top=False, input_shape=shape, **kwargs)
        weights_path = download_chexnet_weights()
        gap = layers.GlobalAveragePooling2D(name="avg_pool")(backbone.output)
        predictions = layers.Dense(14, activation="sigmoid", name="predictions")(gap)
        full_model = keras.Model(inputs=backbone.input, outputs=predictions, name="chexnet_full")
        full_model.load_weights(str(weights_path))

        backbone._name = self.key
        return backbone
