from tensorflow.keras.applications import (
    EfficientNetB0,
    EfficientNetB1,
    EfficientNetB2,
    EfficientNetB3,
    EfficientNetB4,
    EfficientNetB5,
    EfficientNetB6,
    EfficientNetB7,
)
from tensorflow.keras.applications.efficientnet import preprocess_input

from .base import ImagenetBackbone, mode_batch_sizes


class EfficientNetB0Backbone(ImagenetBackbone):
    key = "efficientnetb0"
    application = EfficientNetB0
    preprocess_fn = preprocess_input
    input_size = (224, 224)
    batch_size = mode_batch_sizes(simple=256, full=128, abmil=64)


class EfficientNetB1Backbone(ImagenetBackbone):
    key = "efficientnetb1"
    application = EfficientNetB1
    preprocess_fn = preprocess_input
    input_size = (240, 240)
    batch_size = mode_batch_sizes(simple=256, full=128, abmil=64)


class EfficientNetB2Backbone(ImagenetBackbone):
    key = "efficientnetb2"
    application = EfficientNetB2
    preprocess_fn = preprocess_input
    input_size = (260, 260)
    batch_size = mode_batch_sizes(simple=256, full=64, abmil=64)


class EfficientNetB3Backbone(ImagenetBackbone):
    key = "efficientnetb3"
    application = EfficientNetB3
    preprocess_fn = preprocess_input
    input_size = (300, 300)
    batch_size = mode_batch_sizes(simple=128, full=64, abmil=32)


class EfficientNetB4Backbone(ImagenetBackbone):
    key = "efficientnetb4"
    application = EfficientNetB4
    preprocess_fn = preprocess_input
    input_size = (380, 380)
    batch_size = mode_batch_sizes(simple=128, full=32, abmil=32)


class EfficientNetB5Backbone(ImagenetBackbone):
    key = "efficientnetb5"
    application = EfficientNetB5
    preprocess_fn = preprocess_input
    input_size = (456, 456)
    batch_size = mode_batch_sizes(simple=64, full=32, abmil=16)


class EfficientNetB6Backbone(ImagenetBackbone):
    key = "efficientnetb6"
    application = EfficientNetB6
    preprocess_fn = preprocess_input
    input_size = (528, 528)
    batch_size = mode_batch_sizes(simple=64, full=16, abmil=16)


class EfficientNetB7Backbone(ImagenetBackbone):
    key = "efficientnetb7"
    application = EfficientNetB7
    preprocess_fn = preprocess_input
    input_size = (600, 600)
    batch_size = mode_batch_sizes(simple=32, full=16, abmil=8)
