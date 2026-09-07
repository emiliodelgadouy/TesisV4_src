from tensorflow.keras.applications import (
    EfficientNetV2B0,
    EfficientNetV2B1,
    EfficientNetV2B2,
    EfficientNetV2B3,
    EfficientNetV2L,
    EfficientNetV2M,
    EfficientNetV2S,
)
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input

from .base import ImagenetBackbone


class EfficientNetV2B0Backbone(ImagenetBackbone):
    key = "efficientnetv2b0"
    application = EfficientNetV2B0
    preprocess_fn = preprocess_input
    input_size = (224, 224)


class EfficientNetV2B1Backbone(ImagenetBackbone):
    key = "efficientnetv2b1"
    application = EfficientNetV2B1
    preprocess_fn = preprocess_input
    input_size = (240, 240)


class EfficientNetV2B2Backbone(ImagenetBackbone):
    key = "efficientnetv2b2"
    application = EfficientNetV2B2
    preprocess_fn = preprocess_input
    input_size = (260, 260)


class EfficientNetV2B3Backbone(ImagenetBackbone):
    key = "efficientnetv2b3"
    application = EfficientNetV2B3
    preprocess_fn = preprocess_input
    input_size = (300, 300)


class EfficientNetV2SBackbone(ImagenetBackbone):
    key = "efficientnetv2s"
    application = EfficientNetV2S
    preprocess_fn = preprocess_input
    input_size = (384, 384)


class EfficientNetV2MBackbone(ImagenetBackbone):
    key = "efficientnetv2m"
    application = EfficientNetV2M
    preprocess_fn = preprocess_input
    input_size = (480, 480)


class EfficientNetV2LBackbone(ImagenetBackbone):
    key = "efficientnetv2l"
    application = EfficientNetV2L
    preprocess_fn = preprocess_input
    input_size = (480, 480)
