"""Fachada de imports para ``multirun.ipynb``.

Reexporta en un unico lugar lo que el notebook usa de `src`:

    from src.notebook_api import *

Llamar a ``configure_notebook()`` antes de importar este modulo (TensorFlow se
carga al importar la fachada). Comet ML se importa aqui por si la fachada se
usa sin pasar por ``configure_notebook()``.
"""

from __future__ import annotations

import comet_ml  # noqa: F401 - antes de TensorFlow

import numpy as np
import pandas as pd
import tensorflow as tf

from src.dataset.preparer import DatasetPreparer
from src.dataset.provider import build_dataset_provider
from src.dataset.splits import SplitManager
from src.tracking.comet import CometTracker
from src.training.evaluator import Predictor, ThresholdSelector
from src.training.experiment import TrainingExperiment
from src.training.resources import RandomSeeds

def login_comet(config) -> None:
    CometTracker.login(config)


def set_random_seeds(config) -> None:
    RandomSeeds.set(config)


def predict_probs_and_labels(model, dataset):
    return Predictor.predict_probs_and_labels(model, dataset)


def threshold_youden_j(y_true, y_prob, *, default: float = 0.5) -> float:
    return ThresholdSelector.youden_j(y_true, y_prob, default=default)


def apply_probability_threshold(y_prob, threshold: float):
    return ThresholdSelector.apply(y_prob, threshold)


def prepare_dataset(config):
    return DatasetPreparer().prepare(config)


def get_dataset_splits(config, ds):
    return SplitManager().get(config, ds)


def run_training_experiment(
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
):
    return TrainingExperiment(
        config,
        mode,
        backbone_name,
        train_df,
        val_df,
        test_df,
        pretrained_builder=pretrained_builder,
        return_builder=return_builder,
        return_summary=return_summary,
        experiment_suffix=experiment_suffix,
        dispose_pretrained_builder=dispose_pretrained_builder,
    ).run()


__all__ = [
    "np",
    "pd",
    "tf",
    "apply_probability_threshold",
    "build_dataset_provider",
    "get_dataset_splits",
    "login_comet",
    "predict_probs_and_labels",
    "prepare_dataset",
    "run_training_experiment",
    "set_random_seeds",
    "threshold_youden_j",
]
