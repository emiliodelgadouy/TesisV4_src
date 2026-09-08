"""Fit, checkpoints y evaluate: el loop de entrenamiento, no el grafo Keras."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import TYPE_CHECKING

from tensorflow import keras

from src.training.timer import EpochTimer, MemoryEpochLogger

if TYPE_CHECKING:
    from src.model_builder.base import BaseModelBuilder


def _sanitize_checkpoint_prefix(name: str) -> str:
    slug = re.sub(r"[^\w.\-]+", "_", str(name).strip())
    return slug.strip("_") or "run"


def _is_finite_number(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


class ModelTrainer:
    """Compila, entrena y checkpointea el ``builder.model``.

    El builder define loss/metricas/optimizer y el grafo; este objeto posee el
    estado de stages y checkpoints.
    """

    def __init__(self, builder: BaseModelBuilder, checkpoint_prefix: str | None = None) -> None:
        self.builder = builder
        self.checkpoint_dir = Path("checkpoints")
        self.checkpoint_path = self.checkpoint_dir / "best_checkpoint.weights.h5"
        self.checkpoint_prefix = _sanitize_checkpoint_prefix(checkpoint_prefix) if checkpoint_prefix else None
        self.fit_number = 0
        self.epoch_offset = 0
        self.best_checkpoints: list[dict] = []
        self.global_checkpoint_loaded = False

    def compile(self):
        builder = self.builder
        builder.model.compile(
            optimizer=builder.optimizer(),
            loss=builder.focal_loss(),
            metrics=builder.metrics(),
            jit_compile=builder.jit_compile,
            steps_per_execution=builder.steps_per_execution,
        )
        return builder

    def checkpoint_filepath(self, epoch):
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
        return current < best if self.builder.monitor_mode == "min" else current > best

    def checkpoint_callback(self):
        monitor = self.builder.checkpoint_monitor
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
            self.builder.model.save_weights(str(checkpoint_path))
            for old_path in self.checkpoint_files():
                if old_path != checkpoint_path:
                    old_path.unlink()
            best_value[monitor] = current
            self.best_checkpoints = [info for info in self.best_checkpoints if info["stage"] != stage]
            local_epoch = epoch + 1
            self.best_checkpoints.append(
                {
                    "stage": stage,
                    "epoch": local_epoch,
                    "global_epoch": self.epoch_offset + local_epoch,
                    "monitor": monitor,
                    "value": current,
                    "path": checkpoint_path,
                }
            )
            print(f"\nEpoch {epoch + 1}: {monitor} improved to {current:.4f}. Saved {checkpoint_path}")

        return keras.callbacks.LambdaCallback(on_epoch_end=on_epoch_end)

    def early_stopping_callback(self):
        builder = self.builder
        return keras.callbacks.EarlyStopping(monitor=builder.checkpoint_monitor, mode=builder.monitor_mode, patience=builder.early_stopping_patience, restore_best_weights=True, verbose=1)

    def reduce_lr_callback(self):
        builder = self.builder
        return keras.callbacks.ReduceLROnPlateau(monitor=builder.checkpoint_monitor, mode=builder.monitor_mode, factor=builder.reduce_lr_factor, patience=builder.reduce_lr_patience, min_lr=builder.min_lr, verbose=1)

    def callbacks(self, training_timer=None):
        return [
            self.checkpoint_callback(),
            self.early_stopping_callback(),
            self.reduce_lr_callback(),
            EpochTimer(training_timer=training_timer),
            MemoryEpochLogger(),
        ]

    def fit(self, train_ds, val_ds, epochs=5, callbacks=None, training_timer=None, stage=None, epoch_offset=0):
        # ``stage`` explicito alinea nombres on-disk con Comet aunque se omitan etapas.
        # ``epoch_offset`` no cambia el conteo local de Keras; solo indexa Comet y ``global_epoch``.
        if stage is not None:
            self.fit_number = int(stage)
        else:
            self.fit_number += 1
        self.epoch_offset = int(epoch_offset)
        stage_stem = f"{self.checkpoint_prefix}_stage_{self.fit_number}" if self.checkpoint_prefix else f"stage_{self.fit_number}"
        self.checkpoint_path = self.checkpoint_dir / f"{stage_stem}.weights.h5"
        from src.dataset.provider import as_tf_dataset

        return self.builder.model.fit(as_tf_dataset(train_ds), validation_data=as_tf_dataset(val_ds), epochs=epochs, callbacks=self.callbacks(training_timer=training_timer) + list(callbacks or []))

    def load_best_checkpoint(self):
        info = next((item for item in self.best_checkpoints if item["stage"] == self.fit_number), None)
        if info is None:
            print(f"Advertencia: no hay checkpoint para stage {self.fit_number}; se mantienen los pesos actuales del modelo")
            return None
        self.builder.model.load_weights(str(info["path"]))
        return info

    def load_best_global_checkpoint(self):
        builder = self.builder
        if not self.best_checkpoints:
            raise RuntimeError(f"No hay checkpoints guardados para cargar (model_name={builder.model_name!r}, prefix={self.checkpoint_prefix!r})")
        finite = [item for item in self.best_checkpoints if _is_finite_number(item.get("value"))]
        pool = finite or self.best_checkpoints
        pick = min if builder.monitor_mode == "min" else max
        info = pick(pool, key=lambda item: item["value"])
        builder.model.load_weights(str(info["path"]))
        self.global_checkpoint_loaded = True
        return info

    def evaluate(self, test_ds, return_dict=True):
        from src.dataset.provider import as_tf_dataset

        return self.builder.model.evaluate(as_tf_dataset(test_ds), return_dict=return_dict)

    def predict(self, test_ds, verbose=1):
        from src.dataset.provider import as_tf_dataset

        return self.builder.model.predict(as_tf_dataset(test_ds), verbose=verbose)
