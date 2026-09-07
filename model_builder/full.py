from src.model_builder.simple import SimpleModelBuilder


class FullModelBuilder(SimpleModelBuilder):
    # Igual que SIMPLE, pero con IMG_SIZE tomado de CONFIG["FULL"]["INPUT_SIZE"].
    model_name = "full"
