from typing import override

from tensorflow import keras

from src.model_builder.base import BaseModelBuilder


class SimpleModelBuilder(BaseModelBuilder):
    # Clasificador de imagen completa:
    #
    #   +-------------+    +--------------+    +------------+    +----------+    +-----+    +------------+    +---------+    +-----------------+
    #   | image HxWx3 |--->| augmentation |--->| preprocess |--->| backbone |--->| GAP |--->| Dense+ReLU |--->| Dropout |--->| Dense(1) logits |
    #   +-------------+    +--------------+    +------------+    +----------+    +-----+    +------------+    +---------+    +-----------------+
    model_name = "simple"

    @override
    def inputs(self):
        return keras.Input(shape=(self.IMG_SIZE[0], self.IMG_SIZE[1], 3), name="image")

    @override
    def encode_features(self, x):
        x = self.augmentation(x)
        x = self.preprocess(x)
        return self.backbone(x)

    @override
    def aggregate(self, x):
        return self.head(x)
