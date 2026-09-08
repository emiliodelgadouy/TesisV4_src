from src.model_builder.simple import SimpleModelBuilder


class PatchModelBuilder(SimpleModelBuilder):
    # Clasificador de parches: mismo grafo Keras que SIMPLE; el crop lo hace el dataset.
    #
    #   +-------------+    +--------------+    +------------+    +----------+    +-----+    +------------+    +---------+    +-----------------+
    #   | patch HxWx3 |--->| augmentation |--->| preprocess |--->| backbone |--->| GAP |--->| Dense+ReLU |--->| Dropout |--->| Dense(1) logits |
    #   +-------------+    +--------------+    +------------+    +----------+    +-----+    +------------+    +---------+    +-----------------+
    model_name = "patch"


class PatchHardnegModelBuilder(PatchModelBuilder):
    # Igual que PATCH; el dataset suma parches negativos fuera del ROI en mamografias positivas.
    #
    #   +-------------+    +--------------+    +------------+    +----------+    +-----+    +------------+    +---------+    +-----------------+
    #   | patch HxWx3 |--->| augmentation |--->| preprocess |--->| backbone |--->| GAP |--->| Dense+ReLU |--->| Dropout |--->| Dense(1) logits |
    #   +-------------+    +--------------+    +------------+    +----------+    +-----+    +------------+    +---------+    +-----------------+
    model_name = "patch_hardneg"
