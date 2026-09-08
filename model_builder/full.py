from src.model_builder.simple import SimpleModelBuilder


class FullModelBuilder(SimpleModelBuilder):
    # Igual que SIMPLE, con IMG_SIZE de CONFIG["FULL"]["INPUT_SIZE"] (sin grilla ni bags).
    #
    #   +-------------+    +--------------+    +------------+    +----------+    +-----+    +------------+    +---------+    +-----------------+
    #   | image HxWx3 |--->| augmentation |--->| preprocess |--->| backbone |--->| GAP |--->| Dense+ReLU |--->| Dropout |--->| Dense(1) logits |
    #   +-------------+    +--------------+    +------------+    +----------+    +-----+    +------------+    +---------+    +-----------------+
    model_name = "full"
