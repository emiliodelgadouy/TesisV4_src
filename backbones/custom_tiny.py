from __future__ import annotations

from tensorflow import keras
from tensorflow.keras import layers

from .base import Backbone, DEFAULT_WEIGHTS


class CustomTinyBackbone(Backbone):
    key = "customtiny"
    input_size = (64, 64)
    default_weights = None

    def preprocess_input(self, x):
        # identidad, el rescale va en el modelo
        return x

    def build(self, *, weights=DEFAULT_WEIGHTS, include_top: bool = False, input_shape: tuple[int, int, int] | None = None, **kwargs) -> keras.Model:
        # cnn minima (2 convs) para medir overhead del pipeline, sin transfer learning
        inputs = keras.Input(shape=self.input_shape_or_default(input_shape), name="input")
        x = layers.Rescaling(1.0 / 255.0, name="rescale")(inputs)
        x = layers.Conv2D(8, 3, strides=2, padding="same", activation="relu", name="conv_1")(x)
        x = layers.Conv2D(16, 3, strides=2, padding="same", activation="relu", name="conv_2")(x)
        if include_top:
            x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
            x = layers.Dense(kwargs["classes"], activation="softmax", name="predictions")(x)
        return keras.Model(inputs, x, name=self.key)
