from __future__ import annotations

import time

from src.dataset.provider import as_tf_dataset
from src.training.resume import epochs_completed, history_with_epochs
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
        resume=None,
    ):
        """Las epocas de cada etapa (1=frozen, 2=partial, 3=full) salen de ``config["TRAINING"]``.

        Con ``resume`` (``TrainingResume``) saltea etapas ya completas, retoma una
        etapa a medias desde la ultima epoca guardada y registra cada epoca.
        """
        from src.tracking.comet import CometEpochLogger, CometTracker

        if stage not in _STAGE_EPOCH_KEYS:
            raise ValueError(f"stage invalido: {stage!r} (esperado 1, 2 o 3)")
        epochs = config["TRAINING"][_STAGE_EPOCH_KEYS[stage]]

        if resume is not None and resume.stage_finished(stage):
            done = resume.stage_epochs_done(stage)
            print(f"  Etapa {stage}: ya completada en una sesion anterior ({done} epocas); se saltea")
            return history_with_epochs(done), None, None

        # El setup de freeze/unfreeze corre aunque se omita el fit, para no saltar fases.
        setup_seconds = 0.0
        if setup_fn is not None:
            t0 = time.perf_counter()
            setup_fn()
            setup_seconds = time.perf_counter() - t0

        if epochs <= 0:
            print(f"  Etapa {stage}: omitida (epochs=0)")
            if resume is not None:
                resume.finish_stage(stage, 0, summary=None, elapsed_seconds=training_timer.elapsed_since_training_start())
            return history_with_epochs(0), None, None

        initial_epoch = resume.prepare_stage(stage) if resume is not None else 0
        training_timer.start_stage(stage)

        warmup_seconds = self.warmup(model, train_ds)

        fit_t0 = time.perf_counter()
        callbacks = list(extra_callbacks or [])
        if experiment is not None:
            callbacks.insert(
                0,
                CometEpochLogger(
                    experiment,
                    epoch_offset=epoch_offset,
                    stage=stage,
                    monitor=getattr(model, "checkpoint_monitor", None),
                ),
            )
        if resume is not None:
            callbacks.append(resume.callback(stage, training_timer=training_timer))
        if initial_epoch < epochs:
            history = model.fit(
                train_ds,
                val_ds,
                epochs=epochs,
                callbacks=callbacks,
                training_timer=training_timer,
                stage=stage,
                epoch_offset=epoch_offset,
                initial_epoch=initial_epoch,
            )
        else:
            # Se cortó despues de la ultima epoca y antes de cerrar la etapa.
            model.trainer.fit_number = int(stage)
            model.trainer.epoch_offset = int(epoch_offset)
            history = history_with_epochs(initial_epoch)
        fit_seconds = time.perf_counter() - fit_t0
        stage_epochs = max(epochs_completed(history), initial_epoch)

        ckpt_t0 = time.perf_counter()
        restore_info = model.load_best_checkpoint()
        checkpoint_seconds = time.perf_counter() - ckpt_t0

        summary = training_timer.record_stage_summary(
            stage,
            setup_seconds=setup_seconds,
            warmup_seconds=warmup_seconds,
            fit_seconds=fit_seconds,
            checkpoint_seconds=checkpoint_seconds,
        )
        if resume is not None:
            resume.finish_stage(
                stage,
                stage_epochs,
                summary=summary,
                elapsed_seconds=training_timer.elapsed_since_training_start(),
            )
        global_epoch = epoch_offset + stage_epochs
        if experiment is not None:
            CometTracker.log_stage_timing(experiment, stage, summary, step=global_epoch)
            if restore_info is not None:
                CometTracker.log_checkpoint_restore(
                    experiment,
                    stage=stage,
                    global_epoch=int(restore_info["global_epoch"]),
                )

        print(
            f"  Etapa {stage}: {summary['stage_wall_seconds']:.1f}s total "
            f"(setup={setup_seconds:.1f}s, warmup={warmup_seconds:.1f}s, "
            f"fit={fit_seconds:.1f}s, checkpoint={checkpoint_seconds:.1f}s)"
        )
        if restore_info is not None:
            print(
                f"  Restore etapa {stage}: epoca global {restore_info['global_epoch']} "
                f"(epoca {restore_info['epoch']} de la etapa)"
            )
        return history, restore_info, summary
