from tensorflow import keras
from tensorflow.keras import layers

from src.model_builder.base import BaseModelBuilder
from src.model_builder.layers import BagTiling


class MilModelBuilderBase(BaseModelBuilder):
    def __init__(self, *args, bag_size=None, attention_dim=128, attention_gated=True, bag_grid=(3, 3), bag_keras_tiling=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.bag_grid = (int(bag_grid[0]), int(bag_grid[1]))
        expected_bag_size = self.bag_grid[0] * self.bag_grid[1]
        if bag_size is not None and int(bag_size) != expected_bag_size:
            raise ValueError(
                f"bag_size={bag_size} no coincide con bag_grid={self.bag_grid} "
                f"(esperado {expected_bag_size})"
            )
        self.bag_size = expected_bag_size
        self.attention_dim = attention_dim
        self.attention_gated = attention_gated
        self.bag_keras_tiling = bag_keras_tiling

    def augmentation_seq(self):
        # Augmentacion moderada: menos jitter que aggressive_augmentation porque
        # el tiling fijo ya mueve el contenido entre tiles en cada transformacion.
        layers_list: list[layers.Layer] = []
        if not self.lateralized_inputs:
            layers_list.append(layers.RandomFlip("horizontal", name="aug_flip_h"))
        layers_list.extend([
            layers.RandomRotation(0.03, fill_mode="reflect", name="aug_rot"),
            layers.RandomZoom(height_factor=(0.0, 0.10), width_factor=(0.0, 0.10), fill_mode="reflect", name="aug_zoom"),
            layers.RandomTranslation(height_factor=0.10, width_factor=0.10, fill_mode="reflect", name="aug_translate"),
            layers.RandomContrast(0.15, name="aug_contrast"),
            layers.RandomBrightness(0.15, value_range=(0.0, 255.0), name="aug_brightness"),
        ])
        return keras.Sequential(layers_list, name="augmentation_mil")

    def mil_inputs(self):
        # bag de K parches (K, H, W, 3) — el dataset ya arma los tiles
        return keras.Input(shape=(self.bag_size, self.IMG_SIZE[0], self.IMG_SIZE[1], 3), name="bag")

    def mil_inputs_full(self):
        # imagen completa q despues se parte en tiles adentro del modelo
        rows, cols = self.bag_grid
        h, w = self.IMG_SIZE
        return keras.Input(shape=(rows * h, cols * w, 3), name="bag_full_image")

    def encode_instance_features(self, inputs):
        """Codifica cada instancia sin aplicar dropout."""
        x = layers.TimeDistributed(self.augmentation_seq(), name="td_augmentation")(inputs)
        x = layers.TimeDistributed(layers.Lambda(self.preprocess_input, name="preprocess_input"), name="td_preprocess")(x)
        x = layers.TimeDistributed(self.backbone, name="td_backbone")(x)
        x = layers.TimeDistributed(layers.GlobalAveragePooling2D(name="gap"), name="td_gap")(x)
        return layers.TimeDistributed(layers.Dense(self.top_dense, activation="relu", name="instance_dense"), name="td_instance_dense")(x)

    def encode_instance_features_keras_tiling(self, inputs):
        """Augmenta la imagen completa, genera tiles y devuelve sus embeddings."""
        x = self.augmentation(inputs)
        x = self.preprocess(x)
        x = BagTiling(self.bag_grid, name="bag_tiling")(x)
        x = layers.TimeDistributed(self.backbone, name="td_backbone")(x)
        x = layers.TimeDistributed(layers.GlobalAveragePooling2D(name="gap"), name="td_gap")(x)
        return layers.TimeDistributed(layers.Dense(self.top_dense, activation="relu", name="instance_dense"), name="td_instance_dense")(x)

    def instance_dropout(self, x):
        return layers.TimeDistributed(layers.Dropout(self.dropout, name="instance_dropout"), name="td_instance_dropout")(x)

    def encode_instances(self, inputs):
        # corre backbone en cada instancia del bag con TimeDistributed
        return self.instance_dropout(self.encode_instance_features(inputs))

    def encode_instances_keras_tiling(self, inputs):
        return self.instance_dropout(self.encode_instance_features_keras_tiling(inputs))

    def bag_dropout(self, x):
        return layers.Dropout(self.dropout, dtype="float32", name="bag_dropout")(x)

    def pool_instances(self, x):
        raise NotImplementedError

    def wrap_model(self, inputs, outputs):
        return keras.Model(inputs, outputs, name=self.model_name)

    def build_td_tiling(self):
        inputs = self.mil_inputs()
        x = self.encode_instances(inputs)
        x = self.pool_instances(x)
        x = self.bag_dropout(x)
        outputs = self.output(x)
        self.model = self.wrap_model(inputs, outputs)
        return self.compile()

    def build_keras_tiling(self):
        inputs = self.mil_inputs_full()
        x = self.encode_instances_keras_tiling(inputs)
        x = self.pool_instances(x)
        x = self.bag_dropout(x)
        outputs = self.output(x)
        self.model = self.wrap_model(inputs, outputs)
        return self.compile()

    def _transfer_pretrained_patch_layers(self) -> None:
        # El backbone ya se comparte. La cabeza final de bag siempre queda nueva:
        # un clasificador de instancia y uno de bag no tienen la misma semantica.
        source = self.pretrained_builder
        if source is None:
            return
        if source.model is None or self.model is None:
            raise RuntimeError("No se puede transferir el encoder patch: modelo fuente/destino ausente")
        try:
            dense_weights = source.model.get_layer("dense").get_weights()
            target_dense = self.model.get_layer("td_instance_dense").layer
        except ValueError as exc:
            raise RuntimeError("No se encontro la proyeccion densa del modelo patch") from exc
        target_dense.set_weights(dense_weights)
        target_dense.trainable = False
        print("Modelo patch transferido: backbone + dense congelados para etapa 1; output de bag nuevo")

    def _unfreeze_transferred_patch_layers(self) -> None:
        if self.pretrained_builder is not None and self.model is not None:
            self.model.get_layer("td_instance_dense").layer.trainable = True

    def make_backbone_partially_trainable(
        self,
        trainable_fraction=0.30,
        learning_rate=None,
        train_batch_norm=False,
    ):
        self._unfreeze_transferred_patch_layers()
        return super().make_backbone_partially_trainable(
            trainable_fraction=trainable_fraction,
            learning_rate=learning_rate,
            train_batch_norm=train_batch_norm,
        )

    def make_backbone_trainable(
        self,
        trainable=True,
        learning_rate=None,
        train_batch_norm=False,
    ):
        if trainable:
            self._unfreeze_transferred_patch_layers()
        return super().make_backbone_trainable(
            trainable=trainable,
            learning_rate=learning_rate,
            train_batch_norm=train_batch_norm,
        )

    def build(self):
        result = self.build_keras_tiling() if self.bag_keras_tiling else self.build_td_tiling()
        self._transfer_pretrained_patch_layers()
        if self.pretrained_builder is not None:
            # Keras captura trainable_weights al compilar; recompilar aplica el freeze.
            result = self.compile()
        return result
