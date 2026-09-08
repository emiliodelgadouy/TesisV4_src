from src.model_builder.layers import GatedAttentionPooling
from src.model_builder.mil_base import MilModelBuilderBase


class AbmilModelBuilder(MilModelBuilderBase):
    # Attention-based MIL (Ilse et al.). El tiling es en dataset o en Keras segun BAG_KERAS_TILING.
    #
    # dataset (bag_keras_tiling=False):
    #   +-------------+    +------------+    +---------------+    +-------------+    +--------+    +---------------+    +------------+    +------------+    +-------------+    +-----------------+
    #   | bag KxHxWx3 |--->| TD augment |--->| TD preprocess |--->| TD backbone |--->| TD GAP |--->| TD Dense+ReLU |--->| TD Dropout |--->| gated attn |--->| bag dropout |--->| Dense(1) logits |
    #   +-------------+    +------------+    +---------------+    +-------------+    +--------+    +---------------+    +------------+    +------------+    +-------------+    +-----------------+
    #
    # Keras (bag_keras_tiling=True):
    #   +---------------------+    +--------------+    +------------+    +-----------+    +-------------+    +--------+    +---------------+    +------------+    +------------+    +-------------+    +-----------------+
    #   | image (R*H)x(C*W)x3 |--->| augmentation |--->| preprocess |--->| BagTiling |--->| TD backbone |--->| TD GAP |--->| TD Dense+ReLU |--->| TD Dropout |--->| gated attn |--->| bag dropout |--->| Dense(1) logits |
    #   +---------------------+    +--------------+    +------------+    +-----------+    +-------------+    +--------+    +---------------+    +------------+    +------------+    +-------------+    +-----------------+
    model_name = "abmil"

    def pool_instances(self, x):
        # atencion gated sobre instancias (Ilse et al.)
        return GatedAttentionPooling(attention_dim=self.attention_dim, gated=self.attention_gated, name="attention_pooling")(x)

    def build_keras_tiling(self):
        # Conserva la identidad del submodo (p.ej. abmil_patch_hardneg).
        self.model_name = f"{type(self).model_name}_keras_tiling"
        return super().build_keras_tiling()


class AbmilPatchHardnegModelBuilder(AbmilModelBuilder):
    # Mismo grafo que ABMIL. Transfiere backbone + Dense de instancia desde patch_hardneg;
    # el clasificador de bag se inicializa nuevo.
    #
    # dataset (bag_keras_tiling=False):
    #   +-------------+    +------------+    +---------------+    +-------------+    +--------+    +---------------+    +------------+    +------------+    +-------------+    +-----------------+
    #   | bag KxHxWx3 |--->| TD augment |--->| TD preprocess |--->| TD backbone |--->| TD GAP |--->| TD Dense+ReLU |--->| TD Dropout |--->| gated attn |--->| bag dropout |--->| Dense(1) logits |
    #   +-------------+    +------------+    +---------------+    +-------------+    +--------+    +---------------+    +------------+    +------------+    +-------------+    +-----------------+
    #
    # Keras (bag_keras_tiling=True):
    #   +---------------------+    +--------------+    +------------+    +-----------+    +-------------+    +--------+    +---------------+    +------------+    +------------+    +-------------+    +-----------------+
    #   | image (R*H)x(C*W)x3 |--->| augmentation |--->| preprocess |--->| BagTiling |--->| TD backbone |--->| TD GAP |--->| TD Dense+ReLU |--->| TD Dropout |--->| gated attn |--->| bag dropout |--->| Dense(1) logits |
    #   +---------------------+    +--------------+    +------------+    +-----------+    +-------------+    +--------+    +---------------+    +------------+    +------------+    +-------------+    +-----------------+
    model_name = "abmil_patch_hardneg"

    def build(self):
        if self.pretrained_builder is None:
            raise ValueError("abmil_patch_hardneg requiere pretrained_builder entrenado en patch_hardneg")
        return super().build()
