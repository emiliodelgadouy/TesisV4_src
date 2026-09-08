from tensorflow import keras

from src.model_builder.base import BaseModelBuilder


class SimpleModelBuilder(BaseModelBuilder):
    # Clasificador de imagen completa:
    #
    #   +-------------+    +--------------+    +------------+    +----------+    +-----+    +------------+    +---------+    +-----------------+
    #   | image HxWx3 |--->| augmentation |--->| preprocess |--->| backbone |--->| GAP |--->| Dense+ReLU |--->| Dropout |--->| Dense(1) logits |
    #   +-------------+    +--------------+    +------------+    +----------+    +-----+    +------------+    +---------+    +-----------------+
    model_name = "simple"

    def build(self):
        inputs = self.inputs()
        x = self.augmentation(inputs)
        x = self.preprocess(x)
        x = self.backbone(x)
        x = self.head(x)
        outputs = self.output(x)
        self.model = keras.Model(inputs, outputs, name=self.model_name)
        return self.compile()
