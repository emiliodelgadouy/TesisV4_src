"""Paquete de dataset: I/O, objetivo binario, splits y pipeline tf.data."""

from src.dataset.config import (
    ALL_FINDING_COLUMNS,
    DEFAULT_CLS_POSITIVE_COLUMNS,
    DEFAULT_FILTER_COLUMNS,
    DEFAULT_ROI_NORM_COLUMNS,
    DatasetConfig,
)

__all__ = [
    "ALL_FINDING_COLUMNS",
    "DEFAULT_CLS_POSITIVE_COLUMNS",
    "DEFAULT_FILTER_COLUMNS",
    "DEFAULT_ROI_NORM_COLUMNS",
    "DatasetConfig",
    "DatasetPreparer",
    "DatasetProvider",
    "DatasetProviderConfig",
    "DatasetRepository",
    "DatasetSplits",
    "ImageDecoder",
    "InspectDataset",
    "PositiveMixup",
    "SplitManager",
    "TargetMode",
    "TfDatasetConfig",
    "as_tf_dataset",
    "build_dataset_provider",
    "hard_negatives_from_positives",
]

_LAZY = {
    "DatasetPreparer": ("src.dataset.preparer", "DatasetPreparer"),
    "DatasetProvider": ("src.dataset.provider", "DatasetProvider"),
    "DatasetProviderConfig": ("src.dataset.provider", "DatasetProviderConfig"),
    "DatasetRepository": ("src.dataset.repository", "DatasetRepository"),
    "DatasetSplits": ("src.dataset.provider", "DatasetSplits"),
    "ImageDecoder": ("src.dataset.images", "ImageDecoder"),
    "InspectDataset": ("src.dataset.provider", "InspectDataset"),
    "PositiveMixup": ("src.dataset.mixup", "PositiveMixup"),
    "SplitManager": ("src.dataset.splits", "SplitManager"),
    "TargetMode": ("src.dataset.target", "TargetMode"),
    "TfDatasetConfig": ("src.dataset.provider", "TfDatasetConfig"),
    "as_tf_dataset": ("src.dataset.provider", "as_tf_dataset"),
    "build_dataset_provider": ("src.dataset.provider", "build_dataset_provider"),
    "hard_negatives_from_positives": ("src.dataset.provider", "hard_negatives_from_positives"),
}


def __getattr__(name: str):
    spec = _LAZY.get(name)
    if spec is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module_name, attr = spec
    return getattr(importlib.import_module(module_name), attr)
