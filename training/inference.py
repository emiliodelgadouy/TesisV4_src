"""Modelos ya entrenados: reconstruir el grafo, cargar pesos y predecir sin reentrenar."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.backbones import resolve_backbone
from src.dataset.provider import build_dataset_provider
from src.model_builder import ModelBuilder
from src.training.evaluator import Predictor
from src.training.experiment import resolve_experiment_name
from src.training.mode import TrainingMode, resolve_abmil_config, resolve_batch_size
from src.training.resources import GpuResources


def build_trained_model(
    config,
    mode,
    backbone_name,
    weights_path,
    *,
    input_size=None,
    best_value=None,
    pretrained_builder=None,
):
    """Arma el builder de ``mode × backbone`` y carga ``weights_path``.

    El builder queda listo para ``predict`` y tambien para transferirse como
    ``pretrained_builder`` (p. ej. patch_hardneg → abmil_patch_hardneg): su
    ``best_checkpoints`` apunta a esos pesos.
    """
    mode = TrainingMode.parse(mode)
    exp_name, input_size = resolve_experiment_name(config, mode, backbone_name, input_size=input_size)
    if pretrained_builder is not None:
        input_size = pretrained_builder.IMG_SIZE
        backbone = pretrained_builder.backbone
        preprocess_input = pretrained_builder.preprocess_input
    else:
        backbone, preprocess_input, input_size = resolve_backbone(backbone_name, input_size=input_size)
    builder = ModelBuilder(
        config,
        input_size,
        backbone,
        preprocess_input,
        mode=mode,
        pretrained_builder=pretrained_builder,
        checkpoint_prefix=exp_name,
        lateralized_inputs=True,
        steps_per_execution=1,
    )
    model = builder.build()
    weights_path = Path(weights_path)
    model.model.load_weights(str(weights_path))
    model.trainer.best_checkpoints = [
        {
            "stage": 0,
            "epoch": 0,
            "global_epoch": 0,
            "monitor": model.checkpoint_monitor,
            "value": float(best_value) if best_value is not None else 0.0,
            "path": weights_path,
        }
    ]
    model.pretraining_mode = mode
    return model


def predict_dataframe(config, model, df, *, mode, backbone_name) -> np.ndarray:
    """Probabilidades alineadas con las filas de ``df`` (mismo pipeline de evaluacion)."""
    mode = TrainingMode.parse(mode)
    input_size = tuple(model.IMG_SIZE)
    batch_size, _, _ = resolve_batch_size(
        config,
        mode,
        backbone_name,
        input_size=input_size,
        bag_grid=resolve_abmil_config(config)["BAG_GRID"],
    )
    provider = build_dataset_provider(
        config,
        input_size,
        batch_size,
        lateralize=True,
        mode=mode,
        cache_dataset=False,
    )
    dataset = provider.build_eval(df, name="predict")
    _, y_prob = Predictor.predict_probs_and_labels(model, dataset)
    if len(y_prob) != len(df):
        raise RuntimeError(f"predicciones={len(y_prob)} vs filas={len(df)}")
    return y_prob


def predict_with_weights(config, mode, backbone_name, weights_path, df, *, input_size=None) -> np.ndarray:
    """Construye, predice y libera la GPU."""
    model = build_trained_model(config, mode, backbone_name, weights_path, input_size=input_size)
    try:
        return predict_dataframe(config, model, df, mode=mode, backbone_name=backbone_name)
    finally:
        GpuResources.dispose_model_builder(model)
        GpuResources.release(clear_keras_session=True)
