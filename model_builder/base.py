from pathlib import Path
import math
import re

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.dataset.provider import as_tf_dataset
from src.training.timer import EpochTimer, MemoryEpochLogger


def _sanitize_checkpoint_prefix(name: str) -> str:
    slug = re.sub(r"[^\w.\-]+", "_", str(name).strip())
    return slug.strip("_") or "run"


def _is_finite_number(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


class BaseModelBuilder:
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
        self.checkpoint_dir = Path("checkpoints")
        self.checkpoint_path = self.checkpoint_dir / "best_checkpoint.weights.h5"
        self.fit_number = 0
        self.best_checkpoints = []
        self._global_checkpoint_loaded = False
        self.initial_bias = initial_bias
        self.jit_compile = jit_compile
        self.steps_per_execution = steps_per_execution
        self.checkpoint_prefix = (
            _sanitize_checkpoint_prefix(checkpoint_prefix) if checkpoint_prefix else None
        )
        self.lateralized_inputs = lateralized_inputs
        self.loss_from_logits = True
        self.model = None

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
            layers_list.extend([
                layers.RandomRotation(0.14, fill_mode="reflect", name="aug_rot"),
                layers.RandomZoom(height_factor=(0.0, 0.22), width_factor=(0.0, 0.22), fill_mode="reflect", name="aug_zoom"),
                layers.RandomTranslation(height_factor=0.14, width_factor=0.14, fill_mode="reflect", name="aug_translate"),
                layers.RandomContrast(0.25, name="aug_contrast"),
                layers.RandomBrightness(0.25, value_range=(0.0, 255.0), name="aug_brightness"),
            ])
            return keras.Sequential(layers_list, name="augmentation_aggressive")
        layers_list.extend([
            layers.RandomContrast(0.08),
            layers.RandomBrightness(0.08, value_range=(0.0, 255.0)),
        ])
        return keras.Sequential(layers_list, name="augmentation")

    def augmentation(self, x):
        return self.augmentation_seq()(x)

    def inputs(self):
        return keras.Input(shape=(self.IMG_SIZE[0], self.IMG_SIZE[1], 3), name="image")

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
        self.model.compile(
            optimizer=self.optimizer(),
            loss=self.focal_loss(),
            metrics=self.metrics(),
            jit_compile=self.jit_compile,
            steps_per_execution=self.steps_per_execution,
        )
        return self

    def build(self):
        raise NotImplementedError

    def summary(self):
        return self.model.summary()

    def checkpoint_filepath(self, epoch):
        # path del checkpoint por epoca
        path = Path(self.checkpoint_path)
        stem = path.name.removesuffix(".weights.h5")
        return path.with_name(f"{stem}_epoch{epoch:02d}.weights.h5")

    def checkpoint_files(self):
        path = Path(self.checkpoint_path)
        stem = path.name.removesuffix(".weights.h5")
        return path.parent.glob(f"{stem}_epoch*.weights.h5")

    def monitor_improved(self, current, best):
        # NaN/Inf nunca mejoran; un best no finito se reemplaza por el primer valor finito.
        if not _is_finite_number(current):
            return False
        if best is None or not _is_finite_number(best):
            return True
        return current < best if self.monitor_mode == "min" else current > best

    def checkpoint_callback(self):
        # guarda pesos cuando mejora la metrica monitoreada
        monitor = self.checkpoint_monitor
        stage = self.fit_number
        best_value = {monitor: None}

        def on_epoch_end(epoch, logs):
            logs = logs or {}
            if monitor not in logs:
                print(f"\nEpoch {epoch + 1}: {monitor} ausente en logs; se omite checkpoint")
                return
            current = float(logs[monitor])
            if not self.monitor_improved(current, best_value[monitor]):
                return
            checkpoint_path = self.checkpoint_filepath(epoch + 1)
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            self.model.save_weights(str(checkpoint_path))
            for old_path in self.checkpoint_files():
                if old_path != checkpoint_path:
                    old_path.unlink()
            best_value[monitor] = current
            self.best_checkpoints = [info for info in self.best_checkpoints if info["stage"] != stage]
            self.best_checkpoints.append({"stage": stage, "epoch": epoch + 1, "monitor": monitor, "value": current, "path": checkpoint_path})
            print(f"\nEpoch {epoch + 1}: {monitor} improved to {current:.4f}. Saved {checkpoint_path}")

        return keras.callbacks.LambdaCallback(on_epoch_end=on_epoch_end)

    def early_stopping_callback(self):
        return keras.callbacks.EarlyStopping(monitor=self.checkpoint_monitor, mode=self.monitor_mode, patience=self.early_stopping_patience, restore_best_weights=True, verbose=1)

    def reduce_lr_callback(self):
        return keras.callbacks.ReduceLROnPlateau(monitor=self.checkpoint_monitor, mode=self.monitor_mode, factor=self.reduce_lr_factor, patience=self.reduce_lr_patience, min_lr=self.min_lr, verbose=1)

    def callbacks(self, training_timer=None):
        return [
            self.checkpoint_callback(),
            self.early_stopping_callback(),
            self.reduce_lr_callback(),
            EpochTimer(training_timer=training_timer),
            MemoryEpochLogger(),
        ]

    def fit(self, train_ds, val_ds, epochs=5, callbacks=None, training_timer=None, stage=None):
        # entrena una etapa (frozen / partial / full) y trackea checkpoints por stage.
        # ``stage`` explicito alinea nombres on-disk con Comet aunque se omitan etapas.
        if stage is not None:
            self.fit_number = int(stage)
        else:
            self.fit_number += 1
        stage_stem = (
            f"{self.checkpoint_prefix}_stage_{self.fit_number}"
            if self.checkpoint_prefix
            else f"stage_{self.fit_number}"
        )
        self.checkpoint_path = self.checkpoint_dir / f"{stage_stem}.weights.h5"
        return self.model.fit(as_tf_dataset(train_ds), validation_data=as_tf_dataset(val_ds), epochs=epochs, callbacks=self.callbacks(training_timer=training_timer) + list(callbacks or []))

    def load_best_checkpoint(self):
        # carga el mejor checkpoint de la etapa actual; si no hay, conserva pesos en memoria.
        info = next((item for item in self.best_checkpoints if item["stage"] == self.fit_number), None)
        if info is None:
            print(
                f"Advertencia: no hay checkpoint para stage {self.fit_number}; "
                "se mantienen los pesos actuales del modelo"
            )
            return None
        self.model.load_weights(str(info["path"]))
        return info["epoch"]

    def load_best_global_checkpoint(self):
        # carga el mejor checkpoint de toda la corrida (entre stages)
        if not self.best_checkpoints:
            raise RuntimeError(
                "No hay checkpoints guardados para cargar "
                f"(model_name={self.model_name!r}, prefix={self.checkpoint_prefix!r})"
            )
        finite = [item for item in self.best_checkpoints if _is_finite_number(item.get("value"))]
        pool = finite or self.best_checkpoints
        pick = min if self.monitor_mode == "min" else max
        info = pick(pool, key=lambda i: i["value"])
        self.model.load_weights(str(info["path"]))
        self._global_checkpoint_loaded = True
        return info

    def evaluate(self, test_ds, return_dict=True):
        return self.model.evaluate(as_tf_dataset(test_ds), return_dict=return_dict)

    def predict(self, test_ds, verbose=1):
        return self.model.predict(as_tf_dataset(test_ds), verbose=verbose)
