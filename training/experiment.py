"""Corrida completa de entrenamiento: etapas, evaluacion, umbrales y limpieza."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.backbones import get_backbone, resolve_backbone
from src.dataset.provider import build_dataset_provider
from src.dataset.splits import SplitManager
from src.model_builder import ModelBuilder
from src.tracking.comet import CometTracker
from src.training.evaluator import Predictor, ThresholdSelector
from src.training.mode import TrainingMode, resolve_abmil_config, resolve_batch_size
from src.training.resources import GpuResources
from src.training.stage_runner import TrainingStageRunner
from src.training.timer import TrainingTimer, sample_memory_usage


class TrainingExperiment:
    """Cuerpo de una corrida ``mode`` × ``backbone`` sobre splits ya construidos."""

    def __init__(
        self,
        config,
        mode,
        backbone_name,
        train_df,
        val_df,
        test_df,
        *,
        pretrained_builder=None,
        return_builder=False,
        return_summary=False,
        experiment_suffix=None,
        dispose_pretrained_builder=True,
    ) -> None:
        self.config = config
        self.mode = TrainingMode.parse(mode)
        self.backbone_name = backbone_name
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.pretrained_builder = pretrained_builder
        self.return_builder = return_builder
        self.return_summary = return_summary
        self.experiment_suffix = experiment_suffix
        self.dispose_pretrained_builder = dispose_pretrained_builder

    def run(self):
        config = self.config
        mode = self.mode
        backbone_name = self.backbone_name
        train_df = self.train_df
        val_df = self.val_df
        test_df = self.test_df
        pretrained_builder = self.pretrained_builder

        general = config["GENERAL"]
        training = config["TRAINING"]
        abmil_cfg = resolve_abmil_config(config)
        full_cfg = config.get("FULL") or {}
        patch_cfg = config.get("PATCH") or {}
        patch_hardneg_cfg = config.get("PATCH_HARDNEG") or {}

        bag_grid = abmil_cfg["BAG_GRID"]
        bag_canvas_mode = abmil_cfg["BAG_CANVAS_MODE"]
        bag_keras_tiling = abmil_cfg["BAG_KERAS_TILING"]
        attention_dim = abmil_cfg["ATTENTION_DIM"]
        attention_gated = abmil_cfg["ATTENTION_GATED"]
        patch_resize_to_bag_canvas = patch_cfg.get("RESIZE_TO_BAG_CANVAS", True)
        patch_align_to_bag_grid = (
            patch_hardneg_cfg.get("ALIGN_TO_BAG_GRID", False)
            if mode == "patch_hardneg"
            else False
        )
        # FULL usa su propio INPUT_SIZE. Si falta, 3× el nativo del backbone
        # (default historico, ya no lee la grilla de ABMIL).
        native_size = get_backbone(backbone_name).input_size
        if mode == "full":
            full_override = full_cfg.get("INPUT_SIZE")
            if full_override is not None:
                input_size = tuple(full_override)
            else:
                native_h, native_w = native_size
                input_size = (3 * native_h, 3 * native_w)
        else:
            input_size = native_size

        experiment = model = builder = backbone = dataset_provider = train_ds = val_ds = (
            ds_test
        ) = result_builder = summary = pretrain_best = None
        GpuResources.release(clear_keras_session=pretrained_builder is None)

        exp_name = f"{mode}_{backbone_name}"
        if self.experiment_suffix:
            exp_name = f"{exp_name}_{self.experiment_suffix}"
        is_mil_run = TrainingMode.is_mil(mode)
        batch_size, batch_size_source, batch_size_base = resolve_batch_size(
            config,
            mode,
            backbone_name,
            input_size=input_size,
            bag_grid=bag_grid,
        )
        print(
            f"batch_size={batch_size} (source={batch_size_source}, base={batch_size_base})"
        )
        # Default: cache on en simple/patch; off en full/abmil (canvases grandes).
        cache_dataset = general.get("CACHE_DATASET", not is_mil_run and mode != "full")

        if mode in ("patch", "patch_hardneg"):
            patch_ratio = (
                config["PATCH_HARDNEG"]
                if mode == "patch_hardneg"
                else config["PATCH"]
            )
            train_df, focal_alpha, bias = SplitManager.resample_train_for_patch(
                train_df, patch_ratio, general["RANDOM_SEED"]
            )
        else:
            if len(train_df) == 0:
                raise ValueError("train_df esta vacio; no se puede calcular focal_alpha/bias")
            n_pos = int((train_df["cls"] == 1).sum())
            n_neg = int((train_df["cls"] == 0).sum())
            if n_pos == 0 or n_neg == 0:
                raise ValueError(
                    f"train_df necesita ambas clases; recibido pos={n_pos}, neg={n_neg}"
                )
            focal_alpha = n_neg / len(train_df)
            bias = Predictor.logit_initial_bias(n_pos, n_neg)

        if training.get("FOCAL_ALPHA") is not None:
            focal_alpha = float(training["FOCAL_ALPHA"])
        if training.get("INITIAL_BIAS") is not None:
            bias = float(training["INITIAL_BIAS"])

        try:
            if pretrained_builder is not None:
                input_size = pretrained_builder.IMG_SIZE
                backbone = pretrained_builder.backbone
                preprocess_input = pretrained_builder.preprocess_input
            else:
                backbone, preprocess_input, input_size = resolve_backbone(
                    backbone_name, input_size=input_size
                )

            dataset_provider = build_dataset_provider(
                config,
                input_size,
                batch_size,
                lateralize=True,
                mode=mode,
                patch_align_to_bag_grid=patch_align_to_bag_grid,
                cache_dataset=cache_dataset,
            )
            train_ds, val_ds, ds_test = dataset_provider.build_splits(
                train_df, val_df, test_df
            )

            run_config = {
                **general,
                **training,
                "BACKBONE_ARCHITECTURE": backbone_name,
                "MODE": mode,
                "INPUT_SIZE": list(input_size),
                "TRAIN_ROWS": len(train_df),
                "BATCH_SIZE": batch_size,
                "BATCH_SIZE_BASE": batch_size_base,
                "BATCH_SIZE_SOURCE": batch_size_source,
                "USE_CUSTOM_BATCH_SIZE": bool(general.get("USE_CUSTOM_BATCH_SIZE", False)),
                "CACHE_DATASET": cache_dataset,
                "JIT_COMPILE": True,
                "FOCAL_ALPHA_EFFECTIVE": focal_alpha,
                "INITIAL_BIAS_EFFECTIVE": bias,
            }
            if mode in ("patch", "patch_hardneg"):
                run_config["PATCH_ALIGN_TO_BAG_GRID"] = patch_align_to_bag_grid
                run_config["PATCH_RESIZE_TO_BAG_CANVAS"] = patch_resize_to_bag_canvas
                run_config["PATCH_EVAL_USES_ROI_ORACLE"] = True
                if patch_align_to_bag_grid or patch_resize_to_bag_canvas:
                    run_config["BAG_GRID"] = list(bag_grid)
                    run_config["BAG_CANVAS_MODE"] = bag_canvas_mode
            if is_mil_run:
                run_config["BAG_GRID"] = list(bag_grid)
                run_config["BAG_KERAS_TILING"] = bag_keras_tiling
                run_config["BAG_CANVAS_MODE"] = bag_canvas_mode
                run_config["BAG_CANVAS_SIZE"] = [
                    bag_grid[0] * input_size[0],
                    bag_grid[1] * input_size[1],
                ]
                run_config["BAG_INSTANCES"] = bag_grid[0] * bag_grid[1]
                run_config["ATTENTION_DIM"] = attention_dim
                run_config["ATTENTION_GATED"] = attention_gated
                run_config["BAG_SIZE"] = bag_grid[0] * bag_grid[1]
            if pretrained_builder is not None:
                pretrain_best = pretrained_builder.load_best_global_checkpoint()
                pretraining_mode = getattr(
                    pretrained_builder, "pretraining_mode", pretrained_builder.model_name
                )
                run_config["PRETRAINED_FROM"] = f"{pretraining_mode}_{backbone_name}"
                run_config["PRETRAINED_BEST_VAL_METRIC"] = pretrain_best["value"]
                run_config["PRETRAINED_FROZEN_STAGE1"] = ["backbone", "dense"]
                run_config["BAG_OUTPUT_INITIALIZATION"] = "new"

            steps_per_execution = Predictor.resolve_steps_per_execution(
                len(train_df),
                batch_size,
                max_steps=32,
            )
            run_config["STEPS_PER_EXECUTION"] = steps_per_execution

            builder = ModelBuilder(
                config,
                input_size,
                backbone,
                preprocess_input,
                mode=mode,
                initial_bias=bias,
                focal_alpha=focal_alpha,
                pretrained_builder=pretrained_builder,
                checkpoint_prefix=exp_name,
                lateralized_inputs=True,
                steps_per_execution=steps_per_execution,
            )
            model = builder.build()

            tracker = CometTracker.start(
                config, experiment_name=exp_name, run_config=run_config, model=model.model
            )
            experiment = tracker.experiment
            tracker.log_dataset_class_counts(train=train_df, val=val_df, test=test_df)
            tracker.log_deterministic_input_samples(
                config, train_ds, experiment_name=exp_name, split_name="train"
            )

            training_timer = TrainingTimer()
            training_timer.start_training()
            epoch_offset = 0
            stage_runner = TrainingStageRunner()

            history_frozen, _, _ = stage_runner.run(
                config,
                model,
                train_ds,
                val_ds,
                stage=1,
                training_timer=training_timer,
                epoch_offset=epoch_offset,
                experiment=experiment,
            )
            epoch_offset += len(history_frozen.history.get("loss", []))

            history_partial, _, _ = stage_runner.run(
                config,
                model,
                train_ds,
                val_ds,
                stage=2,
                training_timer=training_timer,
                epoch_offset=epoch_offset,
                experiment=experiment,
                setup_fn=lambda: model.make_backbone_partially_trainable(
                    trainable_fraction=training["BACKBONE_TRAINABLE_FRACTION"],
                    learning_rate=1e-4,
                ),
            )
            epoch_offset += len(history_partial.history.get("loss", []))

            stage_runner.run(
                config,
                model,
                train_ds,
                val_ds,
                stage=3,
                training_timer=training_timer,
                epoch_offset=epoch_offset,
                experiment=experiment,
                setup_fn=lambda: model.make_backbone_trainable(
                    trainable=True, learning_rate=1e-5
                ),
            )
            tracker.log_training_timing_summary(training_timer)
            best_global_checkpoint = model.load_best_global_checkpoint()

            tracker.log_keras_eval_metrics(
                model,
                train_ds=train_ds,
                val_ds=val_ds,
                test_ds=ds_test,
            )

            y_train_true, y_train_prob = Predictor.predict_probs_and_labels(
                model, train_ds.ordered()
            )
            y_val_true, y_val_prob = Predictor.predict_probs_and_labels(model, val_ds)
            y_test_true, y_test_prob = Predictor.predict_probs_and_labels(model, ds_test)

            evaluation_cfg = config.get("EVALUATION") or {}
            prediction_paths = {}
            if evaluation_cfg.get("EXPORT_PREDICTIONS", False):
                prediction_paths["train"] = self._export_predictions(
                    exp_name, "train", train_df, y_train_true, y_train_prob
                )
                prediction_paths["val"] = self._export_predictions(
                    exp_name, "val", val_df, y_val_true, y_val_prob
                )
                prediction_paths["test"] = self._export_predictions(
                    exp_name, "test", test_df, y_test_true, y_test_prob
                )
            bootstrap_intervals = self._bootstrap_auc_intervals(
                y_test_true,
                y_test_prob,
                samples=int(evaluation_cfg.get("BOOTSTRAP_SAMPLES", 0)),
                seed=int(general["RANDOM_SEED"]),
            )
            if bootstrap_intervals:
                experiment.log_metrics(bootstrap_intervals)

            thr_youden = ThresholdSelector.youden_j(y_val_true, y_val_prob)
            thr_recall90 = ThresholdSelector.recall_target(
                y_val_true, y_val_prob, target_recall=0.90
            )

            y_train_pred_default = ThresholdSelector.apply(
                y_train_prob, general["PROBABILITY_THRESHOLD"]
            )
            y_train_pred_youden = ThresholdSelector.apply(y_train_prob, thr_youden)
            y_val_pred_default = ThresholdSelector.apply(
                y_val_prob, general["PROBABILITY_THRESHOLD"]
            )
            y_val_pred_youden = ThresholdSelector.apply(y_val_prob, thr_youden)
            y_pred_default = ThresholdSelector.apply(
                y_test_prob, general["PROBABILITY_THRESHOLD"]
            )
            y_pred_youden = ThresholdSelector.apply(y_test_prob, thr_youden)

            comet_url = tracker.log_test_results(
                config,
                backbone_name=exp_name,
                y_test_true=y_test_true,
                y_test_prob=y_test_prob,
                y_pred_default=y_pred_default,
                y_pred_youden=y_pred_youden,
                y_train_true=y_train_true,
                y_train_prob=y_train_prob,
                y_train_pred_default=y_train_pred_default,
                y_train_pred_youden=y_train_pred_youden,
                y_val_true=y_val_true,
                y_val_prob=y_val_prob,
                y_val_pred_default=y_val_pred_default,
                y_val_pred_youden=y_val_pred_youden,
                thr_youden=thr_youden,
                thr_recall90=thr_recall90,
                best_val_metric=best_global_checkpoint["value"],
                final_weights_path=best_global_checkpoint["path"],
                show_plots=False,
            )
            experiment = None  # ya cerrado por log_test_results; evita doble end() en finally

            if self.return_summary:
                from sklearn.metrics import average_precision_score, roc_auc_score

                summary = {
                    "experiment": exp_name,
                    "mode": mode,
                    "backbone": backbone_name,
                    "input_size": list(input_size),
                    "val_best_metric_name": str(best_global_checkpoint["monitor"]),
                    "val_best_metric": float(best_global_checkpoint["value"]),
                    "final_weights_file": str(best_global_checkpoint["path"]),
                    "test_roc_auc": float(roc_auc_score(y_test_true, y_test_prob)),
                    "test_pr_auc": float(average_precision_score(y_test_true, y_test_prob)),
                    "thr_youden": float(thr_youden),
                    "training_wall_seconds": float(
                        training_timer.elapsed_since_training_start()
                    ),
                    "comet_url": comet_url,
                    **bootstrap_intervals,
                    **sample_memory_usage(),
                }
                for stage, stage_summary in training_timer.stage_summaries.items():
                    for key, value in stage_summary.items():
                        summary[f"stage_{stage}_{key}"] = float(value)
                monitor_name = str(best_global_checkpoint["monitor"])
                if monitor_name == "val_auc":
                    summary["val_best_auc"] = float(best_global_checkpoint["value"])
                elif monitor_name == "val_pr_auc":
                    summary["val_best_pr_auc"] = float(best_global_checkpoint["value"])
                if prediction_paths:
                    summary["train_predictions_file"] = str(prediction_paths["train"])
                    summary["val_predictions_file"] = str(prediction_paths["val"])
                    summary["test_predictions_file"] = str(prediction_paths["test"])
                    summary["predictions_file"] = str(prediction_paths["test"])
                if pretrain_best is not None:
                    summary["pretrained_checkpoint"] = str(pretrain_best["path"])
                    summary["pretrained_best_metric"] = float(pretrain_best["value"])

            if self.return_builder:
                model.pretraining_mode = mode
                result_builder = model
        except Exception as exc:
            import traceback

            print(f"\n[FALLO] {backbone_name}/{mode}: {exc}")
            traceback.print_exc()
            if experiment is not None:
                try:
                    experiment.end()
                except Exception:
                    pass
            raise
        finally:
            keep_builder = self.return_builder and result_builder is not None
            keep_pretrained_builder = pretrained_builder is not None and not self.dispose_pretrained_builder
            del dataset_provider, train_ds, val_ds, ds_test

            if keep_builder or keep_pretrained_builder:
                if not keep_builder:
                    del model, builder, backbone
                GpuResources.release(clear_keras_session=False)
            else:
                del model, builder, backbone
                GpuResources.dispose_model_builder(pretrained_builder)
                GpuResources.release(clear_keras_session=True)

        if self.return_summary:
            return summary
        return result_builder

    @staticmethod
    def _bootstrap_auc_intervals(
        y_true,
        y_prob,
        *,
        samples: int,
        seed: int,
    ) -> dict[str, float]:
        """Intervalos bootstrap estratificados para ROC-AUC y PR-AUC."""
        if samples <= 0:
            return {}
        from sklearn.metrics import average_precision_score, roc_auc_score

        y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
        y_prob = np.asarray(y_prob, dtype=np.float64).reshape(-1)
        positive = np.flatnonzero(y_true == 1)
        negative = np.flatnonzero(y_true == 0)
        if len(positive) == 0 or len(negative) == 0:
            return {}
        rng = np.random.default_rng(seed)
        roc_values = np.empty(samples, dtype=np.float64)
        pr_values = np.empty(samples, dtype=np.float64)
        for index in range(samples):
            sampled = np.concatenate(
                [
                    rng.choice(positive, size=len(positive), replace=True),
                    rng.choice(negative, size=len(negative), replace=True),
                ]
            )
            roc_values[index] = roc_auc_score(y_true[sampled], y_prob[sampled])
            pr_values[index] = average_precision_score(y_true[sampled], y_prob[sampled])
        return {
            "test_roc_auc_ci_low": float(np.quantile(roc_values, 0.025)),
            "test_roc_auc_ci_high": float(np.quantile(roc_values, 0.975)),
            "test_pr_auc_ci_low": float(np.quantile(pr_values, 0.025)),
            "test_pr_auc_ci_high": float(np.quantile(pr_values, 0.975)),
        }

    @staticmethod
    def _export_predictions(
        exp_name: str,
        split_name: str,
        table,
        y_true,
        y_prob,
    ) -> Path:
        """Guarda predicciones alineadas para comparaciones pareadas sin reentrenar."""
        export_dir = Path("exports")
        export_dir.mkdir(parents=True, exist_ok=True)
        columns = [
            column
            for column in (
                "patient_id",
                "image_id",
                "path",
                "view",
                "laterality",
                "breast_birads",
                "birads",
                "breast_density",
                "density",
                "finding_categories",
                "Mass",
                "Suspicious_Lymph_Node",
                "Nipple_Retraction",
                "Skin_Retraction",
                "Skin_Thickening",
                "Suspicious_Calcification",
                "Architectural_Distortion",
                "Asymmetry",
                "Focal_Asymmetry",
                "Global_Asymmetry",
                "No_Finding",
                "cls",
            )
            if column in table.columns
        ]
        predictions = table[columns].reset_index(drop=True).copy()
        if len(predictions) != len(y_true):
            raise RuntimeError(
                f"No se pueden exportar predicciones de {exp_name}: "
                f"tabla={len(predictions)} vs predicciones={len(y_true)}"
            )
        predictions["y_true"] = np.asarray(y_true, dtype=np.int64)
        predictions["y_prob"] = np.asarray(y_prob, dtype=np.float64)
        path = export_dir / f"{exp_name}_{split_name}_predictions.csv"
        predictions.to_csv(path, index=False)
        return path
