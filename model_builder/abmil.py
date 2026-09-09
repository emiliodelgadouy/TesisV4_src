from typing import override

from src.model_builder.layers import GatedAttentionPooling
from src.model_builder.mil_base import MilModelBuilderBase


class AbmilModelBuilder(MilModelBuilderBase):
    # Attention-based MIL (Ilse et al.). El tiling es en dataset o en Keras segun BAG_KERAS_TILING.
    # abmil_patch_hardneg usa este mismo grafo: transfiere backbone + Dense de instancia
    # desde patch_hardneg y deja nuevo el clasificador de bag.
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

    @override
    def pool_instances(self, x):
        # atencion gated sobre instancias (Ilse et al.)
        return GatedAttentionPooling(attention_dim=self.attention_dim, gated=self.attention_gated, name="attention_pooling")(x)

    @override
    def keras_model_name(self) -> str:
        if self.bag_keras_tiling:
            return f"{self.model_name}_keras_tiling"
        return self.model_name
