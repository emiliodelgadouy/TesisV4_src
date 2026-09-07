from src.model_builder.simple import SimpleModelBuilder


class PatchModelBuilder(SimpleModelBuilder):
    # clasificador de parches: mismo grafo que simple; el crop lo hace el dataset
    model_name = "patch"


class PatchHardnegModelBuilder(PatchModelBuilder):
    model_name = "patch_hardneg"
