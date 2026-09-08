from src.model_builder.simple import SimpleModelBuilder


class FullModelBuilder(SimpleModelBuilder):
    # Igual que SIMPLE, pero con IMG_SIZE tomado de CONFIG["FULL"]["INPUT_SIZE"].
    # No usa grilla ni bags: es un clasificador de imagen completa a mayor resolucion.
    model_name = "full"
