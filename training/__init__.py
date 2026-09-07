"""Paquete de entrenamiento: modos, etapas, evaluacion y recursos GPU."""

from src.training.mode import TrainingMode

__all__ = [
    "EpochTimer",
    "GpuResources",
    "MemoryEpochLogger",
    "Predictor",
    "RandomSeeds",
    "ThresholdSelector",
    "TrainingExperiment",
    "TrainingMode",
    "TrainingStageRunner",
    "TrainingTimer",
]

_LAZY = {
    "EpochTimer": ("src.training.timer", "EpochTimer"),
    "MemoryEpochLogger": ("src.training.timer", "MemoryEpochLogger"),
    "TrainingTimer": ("src.training.timer", "TrainingTimer"),
    "GpuResources": ("src.training.resources", "GpuResources"),
    "RandomSeeds": ("src.training.resources", "RandomSeeds"),
    "Predictor": ("src.training.evaluator", "Predictor"),
    "ThresholdSelector": ("src.training.evaluator", "ThresholdSelector"),
    "TrainingExperiment": ("src.training.experiment", "TrainingExperiment"),
    "TrainingStageRunner": ("src.training.stage_runner", "TrainingStageRunner"),
}


def __getattr__(name: str):
    spec = _LAZY.get(name)
    if spec is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr = spec
    import importlib

    return getattr(importlib.import_module(module_name), attr)
