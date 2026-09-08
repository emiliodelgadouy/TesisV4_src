from __future__ import annotations

from abc import ABC, abstractmethod

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.training.model_trainer import ModelTrainer


class BaseModelBuilder(ABC):
    """Arma el grafo Keras. El loop de fit/checkpoint vive en ``ModelTrainer``.

    Template method ``build``: ``inputs`` -> ``encode_features`` -> ``aggregate``
    -> ``regularize`` -> ``output``. Las subclases concretas tienen que
    implementar los tres pasos marcados abstractos.
    """

    model_name = "model"

    def __init__(self, IMG_SIZE, backbone, preprocess_input, backbone_trainable=False, top_dense=256, dropout=0.4, learning_rate=1e-3, focal_alpha=0.90, focal_gamma=2.0, metric_to_maximize="pr_auc", checkpoint_monitor=None, monitor_mode="max", early_stopping_patience=8, reduce_lr_patience=4, reduce_lr_factor=0.5, min_lr=1e-7, aggressive_augmentation=False, initial_bias=None, pretrained_builder=None, jit_compile=True, steps_per_execution=32, checkpoint_prefix=None, lateralized_inputs=False):
        self.pretrained_builder = pretrained_builder
        if pretrained_builder is not None:
            # Evita recargar si el caller (p.ej. experiment) ya cargo el mejor global.
            if not getattr(pretrained_builder, "_global_checkpoint_loaded", False):
                pretrained_builder.load_best_global_checkpoint()
            IMG_SIZE = pretrained_builder.IMG_SIZE
            backbone = pretrained_builder.backbone
            preprocess_input = pretrained_builder.preprocess_input
        self.IMG_SIZE = IMG_SIZE
        self.backbone = backbone
        self.preprocess_input = preprocess_input
        self.backbone.trainable = backbone_trainable
        self.aggressive_augmentation = aggressive_augmentation
        self.top_dense = top_dense
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        monitor = checkpoint_monitor or metric_to_maximize
        if not str(monitor).startswith("val_"):
            monitor = f"val_{monitor}"
        self.checkpoint_monitor = monitor
        self.metric_to_maximize = monitor.removeprefix("val_")
        self.monitor_mode = monitor_mode
        self.early_stopping_patience = early_stopping_patience
        self.reduce_lr_patience = reduce_lr_patience
        self.reduce_lr_factor = reduce_lr_factor
        self.min_lr = min_lr
        self.initial_bias = initial_bias
        self.jit_compile = jit_compile
        self.steps_per_execution = steps_per_execution
        self.lateralized_inputs = lateralized_inputs
        self.loss_from_logits = True
        self.model = None
        self.trainer = ModelTrainer(self, checkpoint_prefix=checkpoint_prefix)

    @property
    def checkpoint_prefix(self):
        return self.trainer.checkpoint_prefix

    @property
    def best_checkpoints(self):
        return self.trainer.best_checkpoints

    @property
    def _global_checkpoint_loaded(self) -> bool:
        return self.trainer.global_checkpoint_loaded

    @abstractmethod
    def inputs(self):
        """Tensor de entrada del grafo (imagen, bag o canvas)."""

    @abstractmethod
    def encode_features(self, x):
        """Representacion previa al agregado (mapa espacial o instancias)."""

    @abstractmethod
    def aggregate(self, x):
        """Reduce el encode a un vector de bag/imagen."""

    def regularize(self, x):
        return x

    def keras_model_name(self) -> str:
        return self.model_name

    def after_build(self):
        """Hook post-grafo (p.ej. transferir pesos) antes de ``compile``."""
        return None

    def build(self):
        inputs = self.inputs()
        x = self.encode_features(inputs)
        x = self.aggregate(x)
        x = self.regularize(x)
        self.model = keras.Model(inputs, self.output(x), name=self.keras_model_name())
        self.after_build()
        return self.compile()

    def top_mlp(self, x):
        # capas densas de la cabeza (relu + dropout)
        x = layers.Dense(self.top_dense, activation="relu", name="dense")(x)
        return layers.Dropout(self.dropout, name="dropout")(x)

    def head(self, x):
        # gap + mlp, tipico para clasificacion de imagen completa
        return self.top_mlp(layers.GlobalAveragePooling2D(name="gap")(x))

    def augmentation_seq(self):
        # augmentacion de entrenamiento, agresiva o suave segun config
        layers_list: list[layers.Layer] = []
        if not self.lateralized_inputs:
            layers_list.append(layers.RandomFlip("horizontal", name="aug_flip_h"))
        if self.aggressive_augmentation:
            layers_list.extend(
                [
                    layers.RandomRotation(0.14, fill_mode="reflect", name="aug_rot"),
                    layers.RandomZoom(height_factor=(0.0, 0.22), width_factor=(0.0, 0.22), fill_mode="reflect", name="aug_zoom"),
                    layers.RandomTranslation(height_factor=0.14, width_factor=0.14, fill_mode="reflect", name="aug_translate"),
                    layers.RandomContrast(0.25, name="aug_contrast"),
                    layers.RandomBrightness(0.25, value_range=(0.0, 255.0), name="aug_brightness"),
                ]
            )
            return keras.Sequential(layers_list, name="augmentation_aggressive")
        layers_list.extend(
            [
                layers.RandomContrast(0.08),
                layers.RandomBrightness(0.08, value_range=(0.0, 255.0)),
            ]
        )
        return keras.Sequential(layers_list, name="augmentation")

    def augmentation(self, x):
        return self.augmentation_seq()(x)

    def output(self, x):
        # salida binaria en logits (sigmoid va en la loss)
        bias_init = tf.keras.initializers.Constant(self.initial_bias) if self.initial_bias is not None else "zeros"
        x = layers.Dense(1, dtype="float32", bias_initializer=bias_init, name="output")(x)
        if self.loss_from_logits:
            return x
        return layers.Activation("sigmoid", dtype="float32", name="output_sigmoid")(x)

    def preprocess(self, x):
        return layers.Lambda(self.preprocess_input, name="preprocess_input")(x)

    def optimizer(self):
        return keras.optimizers.Adam(learning_rate=self.learning_rate)

    def focal_loss(self):
        # apply_class_balancing=True es imprescindible: sin el, Keras ignora alpha
        # y la clase positiva pierde ponderacion (el modelo colapsa a negativo).
        # alpha pesa la clase 1 (positiva) y 1-alpha la clase 0; con
        # focal_alpha=frac_negativos, los positivos quedan upweighted para
        # compensar el desbalance residual tras el undersample/resample.
        return keras.losses.BinaryFocalCrossentropy(
            apply_class_balancing=True,
            alpha=self.focal_alpha,
            gamma=self.focal_gamma,
            from_logits=self.loss_from_logits,
        )

    def metrics(self):
        threshold = 0.0 if self.loss_from_logits else 0.5
        return [
            keras.metrics.BinaryAccuracy(name="accuracy", threshold=threshold),
            keras.metrics.AUC(name="auc", from_logits=self.loss_from_logits),
            keras.metrics.AUC(curve="PR", name="pr_auc", from_logits=self.loss_from_logits),
            keras.metrics.Precision(name="precision", thresholds=threshold),
            keras.metrics.Recall(name="recall", thresholds=threshold),
        ]

    def keep_batch_norm_frozen(self):
        # congela batchnorm del backbone cuando hacemos fine-tuning parcial
        for layer in self.backbone.layers:
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False

    def make_backbone_trainable(self, trainable=True, learning_rate=None, train_batch_norm=False):
        self.backbone.trainable = trainable
        if trainable and not train_batch_norm:
            self.keep_batch_norm_frozen()
        if learning_rate is not None:
            self.learning_rate = learning_rate
        return self.compile()

    def make_backbone_partially_trainable(self, trainable_fraction=0.30, learning_rate=None, train_batch_norm=False):
        # descongela solo el ultimo % de capas del backbone
        total = len(self.backbone.layers)
        freeze_until = total - max(1, round(total * trainable_fraction))
        self.backbone.trainable = True
        for layer in self.backbone.layers[:freeze_until]:
            layer.trainable = False
        for layer in self.backbone.layers[freeze_until:]:
            layer.trainable = True
        if not train_batch_norm:
            self.keep_batch_norm_frozen()
        if learning_rate is not None:
            self.learning_rate = learning_rate
        return self.compile()

    def compile(self):
        return self.trainer.compile()

    def summary(self):
        return self.model.summary()

    def callbacks(self, training_timer=None):
        return self.trainer.callbacks(training_timer=training_timer)

    def fit(self, train_ds, val_ds, epochs=5, callbacks=None, training_timer=None, stage=None):
        return self.trainer.fit(train_ds, val_ds, epochs=epochs, callbacks=callbacks, training_timer=training_timer, stage=stage)

    def load_best_checkpoint(self):
        return self.trainer.load_best_checkpoint()

    def load_best_global_checkpoint(self):
        return self.trainer.load_best_global_checkpoint()

    def evaluate(self, test_ds, return_dict=True):
        return self.trainer.evaluate(test_ds, return_dict=return_dict)

    def predict(self, test_ds, verbose=1):
        return self.trainer.predict(test_ds, verbose=verbose)
