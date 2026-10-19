"""Paquete de entrenamiento: modos, etapas, evaluacion y recursos GPU."""

from src.training.mode import TrainingMode, resolve_abmil_config, resolve_batch_size, scale_batch_size, resolve_resized_input_sizes, resolve_resized_input_size, resolve_resized_batch_size, resolve_resized_cache, resized_size_label

__all__ = [
    "EpochTimer",
    "ExperimentPlan",
    "GpuResources",
    "MemoryEpochLogger",
    "ModelTrainer",
    "PairedBootstrap",
    "Predictor",
    "RandomSeeds",
    "RunSpec",
    "RunStore",
    "TrainingResume",
    "ThresholdSelector",
    "TrainingExperiment",
    "TrainingMode",
    "resolve_abmil_config",
    "resolve_batch_size",
    "resolve_resized_batch_size",
    "resolve_resized_cache",
    "resolve_resized_input_size",
    "resolve_resized_input_sizes",
    "resized_size_label",
    "scale_batch_size",
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
    "PairedBootstrap": ("src.training.evaluator", "PairedBootstrap"),
    "ExperimentPlan": ("src.training.plan", "ExperimentPlan"),
    "RunSpec": ("src.training.plan", "RunSpec"),
    "RunStore": ("src.training.resume", "RunStore"),
    "TrainingResume": ("src.training.resume", "TrainingResume"),
    "ThresholdSelector": ("src.training.evaluator", "ThresholdSelector"),
    "TrainingExperiment": ("src.training.experiment", "TrainingExperiment"),
    "TrainingStageRunner": ("src.training.stage_runner", "TrainingStageRunner"),
    "ModelTrainer": ("src.training.model_trainer", "ModelTrainer"),
}


def __getattr__(name: str):
    spec = _LAZY.get(name)
    if spec is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr = spec
    import importlib

    return getattr(importlib.import_module(module_name), attr)
