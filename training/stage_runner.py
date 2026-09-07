from __future__ import annotations

import time

from tensorflow import keras

from src.dataset.provider import as_tf_dataset
from src.training.timer import TrainingTimer


_STAGE_EPOCH_KEYS = {
    1: "EPOCHS_FROZEN_BACKBONE",
    2: "EPOCHS_PARTIAL_BACKBONE",
    3: "EPOCHS_FULL_FINETUNE",
}


class TrainingStageRunner:
    """Ejecuta una etapa de congelamiento con metricas de tiempo reales (wall-clock)."""

    @staticmethod
    def warmup(model, train_ds) -> float:
        """Un paso forward-only para compilar el grafo antes de model.fit() (sin actualizar pesos)."""
        t0 = time.perf_counter()
        try:
            xb, _ = next(iter(as_tf_dataset(train_ds)))
        except StopIteration as exc:
            raise ValueError("warmup: el dataset de train no produjo batches") from exc
        model.model.predict_on_batch(xb)
        elapsed = time.perf_counter() - t0
        print(f"  Warm-up completado en {elapsed:.1f}s")
        return elapsed

    def run(
        self,
        config,
        model,
        train_ds,
        val_ds,
        *,
        stage: int,
        training_timer: TrainingTimer,
        epoch_offset: int = 0,
        experiment=None,
        extra_callbacks=None,
        setup_fn=None,
    ):
        """Las epocas de cada etapa (1=frozen, 2=partial, 3=full) salen de ``config["TRAINING"]``."""
        from src.tracking.comet import CometEpochLogger, CometTracker

        if stage not in _STAGE_EPOCH_KEYS:
            raise ValueError(f"stage invalido: {stage!r} (esperado 1, 2 o 3)")
        epochs = config["TRAINING"][_STAGE_EPOCH_KEYS[stage]]

        # El setup de freeze/unfreeze corre aunque se omita el fit, para no saltar fases.
        setup_seconds = 0.0
        if setup_fn is not None:
            t0 = time.perf_counter()
            setup_fn()
            setup_seconds = time.perf_counter() - t0

        if epochs <= 0:
            empty_history = keras.callbacks.History()
            empty_history.history = {}
            print(f"  Etapa {stage}: omitida (epochs=0)")
            return empty_history, None, None

        training_timer.start_stage(stage)

        warmup_seconds = self.warmup(model, train_ds)

        fit_t0 = time.perf_counter()
        callbacks = list(extra_callbacks or [])
        if experiment is not None:
            callbacks.insert(
                0,
                CometEpochLogger(experiment, epoch_offset=epoch_offset, stage=stage),
            )
        history = model.fit(
            train_ds,
            val_ds,
            epochs=epochs,
            callbacks=callbacks,
            training_timer=training_timer,
            stage=stage,
        )
        fit_seconds = time.perf_counter() - fit_t0

        ckpt_t0 = time.perf_counter()
        best_epoch = model.load_best_checkpoint()
        checkpoint_seconds = time.perf_counter() - ckpt_t0

        summary = training_timer.record_stage_summary(
            stage,
            setup_seconds=setup_seconds,
            warmup_seconds=warmup_seconds,
            fit_seconds=fit_seconds,
            checkpoint_seconds=checkpoint_seconds,
        )
        epochs_completed = len(history.history.get("loss", []))
        global_epoch = epoch_offset + epochs_completed
        if experiment is not None:
            CometTracker.log_stage_timing(experiment, stage, summary, step=global_epoch)

        print(
            f"  Etapa {stage}: {summary['stage_wall_seconds']:.1f}s total "
            f"(setup={setup_seconds:.1f}s, warmup={warmup_seconds:.1f}s, "
            f"fit={fit_seconds:.1f}s, checkpoint={checkpoint_seconds:.1f}s)"
        )
        return history, best_epoch, summary
