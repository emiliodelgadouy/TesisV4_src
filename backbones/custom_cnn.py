from __future__ import annotations

from typing import override

from tensorflow import keras
from tensorflow.keras import layers

from .base import Backbone, DEFAULT_WEIGHTS, mode_batch_sizes


def conv_bn_relu(x, filters: int, kernel_size: int, *, strides: int = 1, name: str):
    # bloque basico: conv + batchnorm + relu
    x = layers.Conv2D(filters, kernel_size, strides=strides, padding="same", use_bias=False, name=f"{name}_conv")(x)
    x = layers.BatchNormalization(name=f"{name}_bn")(x)
    return layers.Activation("relu", name=f"{name}_relu")(x)


def residual_block(x, filters: int, name: str, *, strides: int = 1, dropout: float = 0.0):
    # bloque residual chico (tipo resnet pero mas simple)
    shortcut = x
    x = conv_bn_relu(x, filters, 3, strides=strides, name=f"{name}_a")
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"{name}_b_conv")(x)
    x = layers.BatchNormalization(name=f"{name}_b_bn")(x)
    if strides != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, strides=strides, padding="same", use_bias=False, name=f"{name}_proj_conv")(shortcut)
        shortcut = layers.BatchNormalization(name=f"{name}_proj_bn")(shortcut)
    x = layers.Add(name=f"{name}_add")([x, shortcut])
    x = layers.Activation("relu", name=f"{name}_relu")(x)
    if dropout > 0:
        x = layers.Dropout(dropout, name=f"{name}_drop")(x)
    return x


def residual_stage(x, filters: int, blocks: int, name: str, *, first_stride: int = 1, dropout: float = 0.15):
    # una etapa con varios bloques residuales seguidos
    x = residual_block(x, filters, f"{name}_block0", strides=first_stride, dropout=dropout)
    for i in range(1, blocks):
        x = residual_block(x, filters, f"{name}_block{i}", dropout=dropout)
    return x


class CustomCnnBackbone(Backbone):
    """CNN residual hecha a mano, entrenada desde cero (sin pesos preentrenados)."""

    key = "customcnn"
    input_size = (224, 224)
    default_weights = None
    batch_size = mode_batch_sizes(standard=256, resized=128, abmil=64)

    @override
    def preprocess_input(self, x):
        # identidad, el rescale va en el modelo
        return x

    @override
    def build(self, *, weights=DEFAULT_WEIGHTS, include_top: bool = False, input_shape: tuple[int, int, int] | None = None, **kwargs) -> keras.Model:
        if self.coalesce_weights(weights) is not None:
            raise ValueError("customcnn solo admite weights=None; se entrena desde cero")
        # cnn custom con bloques residuales
        inputs = keras.Input(shape=self.input_shape_or_default(input_shape), name="input")
        x = layers.Rescaling(1.0 / 255.0, name="rescale")(inputs)
        x = conv_bn_relu(x, 64, 7, strides=2, name="stem")
        x = layers.MaxPooling2D(3, strides=2, padding="same", name="stem_pool")(x)
        x = residual_stage(x, 64, 2, "stage1", dropout=0.05)
        x = residual_stage(x, 128, 2, "stage2", first_stride=2, dropout=0.10)
        x = residual_stage(x, 256, 2, "stage3", first_stride=2, dropout=0.15)
        x = residual_stage(x, 512, 2, "stage4", first_stride=2, dropout=0.20)
        if include_top:
            x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
            x = layers.Dense(kwargs["classes"], activation="softmax", name="predictions")(x)
        return keras.Model(inputs, x, name=self.key)
