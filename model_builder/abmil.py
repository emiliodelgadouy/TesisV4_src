from src.model_builder.layers import GatedAttentionPooling
from src.model_builder.mil_base import MilModelBuilderBase


class AbmilModelBuilder(MilModelBuilderBase):
    model_name = "abmil"

    def pool_instances(self, x):
        # atencion gated sobre instancias (Ilse et al.)
        return GatedAttentionPooling(attention_dim=self.attention_dim, gated=self.attention_gated, name="attention_pooling")(x)

    def build_keras_tiling(self):
        # Conserva la identidad del submodo (p.ej. abmil_patch_hardneg).
        self.model_name = f"{type(self).model_name}_keras_tiling"
        return super().build_keras_tiling()


class AbmilPatchHardnegModelBuilder(AbmilModelBuilder):
    model_name = "abmil_patch_hardneg"

    def build(self):
        if self.pretrained_builder is None:
            raise ValueError("abmil_patch_hardneg requiere pretrained_builder entrenado en patch_hardneg")
        return super().build()
