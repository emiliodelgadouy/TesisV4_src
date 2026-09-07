from __future__ import annotations

import time

from tensorflow import keras


class TrainingTimer:
    """Wall-clock compartido entre las fases de entrenamiento multi-etapa."""

    def __init__(self) -> None:
        self._training_start: float | None = None
        self._stage_start: float | None = None
        self.current_stage: int | None = None
        self.stage_summaries: dict[int, dict[str, float]] = {}

    def start_training(self) -> None:
        self._training_start = time.perf_counter()

    def start_stage(self, stage: int) -> None:
        self.current_stage = stage
        self._stage_start = time.perf_counter()

    def elapsed_since_training_start(self) -> float:
        if self._training_start is None:
            return 0.0
        return time.perf_counter() - self._training_start

    def elapsed_since_stage_start(self) -> float:
        if self._stage_start is None:
            return 0.0
        return time.perf_counter() - self._stage_start

    def record_stage_summary(
        self,
        stage: int,
        *,
        setup_seconds: float,
        warmup_seconds: float,
        fit_seconds: float,
        checkpoint_seconds: float,
    ) -> dict[str, float]:
        summary = {
            "setup_seconds": setup_seconds,
            "warmup_seconds": warmup_seconds,
            "fit_seconds": fit_seconds,
            "checkpoint_seconds": checkpoint_seconds,
            "stage_wall_seconds": self.elapsed_since_stage_start(),
        }
        self.stage_summaries[stage] = summary
        return summary


def sample_memory_usage() -> dict[str, float]:
    """RAM del proceso y VRAM de GPU:0 en GB. Omite metricas no disponibles."""
    import tensorflow as tf

    metrics: dict[str, float] = {}

    try:
        import psutil

        metrics["ram_rss_gb"] = psutil.Process().memory_info().rss / (1024**3)
    except Exception:
        pass

    try:
        if tf.config.list_physical_devices("GPU"):
            info = tf.config.experimental.get_memory_info("GPU:0")
            metrics["vram_current_gb"] = info["current"] / (1024**3)
            metrics["vram_peak_gb"] = info["peak"] / (1024**3)
    except Exception:
        pass

    return metrics


class MemoryEpochLogger(keras.callbacks.Callback):
    """Anade uso de RAM/VRAM a logs al final de cada epoca (Comet los recoge via CometEpochLogger)."""

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            return
        logs.update(sample_memory_usage())


class EpochTimer(keras.callbacks.Callback):
    def __init__(self, training_timer: TrainingTimer | None = None):
        super().__init__()
        self.training_timer = training_timer

    def on_train_begin(self, logs=None):
        self.epoch_times = []
        self.fit_elapsed_times = []
        self._fit_start = time.perf_counter()

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        epoch_wall = time.perf_counter() - self.epoch_start_time
        fit_elapsed = time.perf_counter() - self._fit_start
        self.epoch_times.append(epoch_wall)
        self.fit_elapsed_times.append(fit_elapsed)
        if logs is None:
            return

        logs["epoch_wall_seconds"] = epoch_wall
        logs["epoch_time_seconds"] = epoch_wall
        logs["fit_elapsed_seconds"] = fit_elapsed
        logs["total_elapsed_seconds"] = fit_elapsed
        if self.training_timer is not None:
            logs["global_elapsed_seconds"] = self.training_timer.elapsed_since_training_start()
            logs["stage_elapsed_seconds"] = self.training_timer.elapsed_since_stage_start()
