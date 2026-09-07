from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_FILTER_COLUMNS = (
    "Mass",
    "Suspicious_Lymph_Node",
    "Nipple_Retraction",
    "Skin_Retraction",
    "Skin_Thickening",
    "Suspicious_Calcification",
    "No_Finding",
)

DEFAULT_CLS_POSITIVE_COLUMNS = (
    "Mass",
    "Suspicious_Lymph_Node",
    "Nipple_Retraction",
    "Skin_Retraction",
    "Skin_Thickening",
    "Suspicious_Calcification",
)

ALL_FINDING_COLUMNS = (
    "Architectural_Distortion",
    "Asymmetry",
    "Focal_Asymmetry",
    "Global_Asymmetry",
    *DEFAULT_FILTER_COLUMNS,
)

DEFAULT_ROI_NORM_COLUMNS: tuple[str, str, str, str] = (
    "pad_resized_xmin_norm",
    "pad_resized_ymin_norm",
    "pad_resized_xmax_norm",
    "pad_resized_ymax_norm",
)


@dataclass
class DatasetConfig:
    """Rutas locales y URIs GCS del dataset de mamografias."""

    root_dir: Path = field(default_factory=lambda: Path.cwd().resolve())
    data_dirname: str = "mammo"
    gcs_images_tar: str = "gs://helen-data/square_images.tar.gz"
    gcs_data_csv: str = "gs://helen-data/square_data.csv"
    gcs_splits_json: str = "gs://helen-data/dataset_splits.json"
    tar_filename: str = "square_images.tar.gz"
    csv_filename: str = "square_data.csv"
    splits_filename: str = "dataset_splits.json"
    download_from_gcs: bool = True
    extract_images: bool = True
    filter_columns: tuple[str, ...] = DEFAULT_FILTER_COLUMNS
    cls_column: str = "cls"
    cls_positive_columns: tuple[str, ...] = DEFAULT_CLS_POSITIVE_COLUMNS
    cls_positive_value: int = 1
    birads_column: str = "breast_birads"

    @property
    def data_dir(self) -> Path:
        return self.root_dir / self.data_dirname

    @property
    def raw_img_dir(self) -> Path:
        return self.data_dir / "raw" / "images"

    @property
    def raw_csv_dir(self) -> Path:
        return self.data_dir / "raw" / "csv"

    @property
    def splits_dir(self) -> Path:
        return self.root_dir / "splits"

    @property
    def csv_main(self) -> Path:
        return self.raw_csv_dir / self.csv_filename

    @property
    def tar_local(self) -> Path:
        return self.raw_img_dir / self.tar_filename

    @property
    def splits_local(self) -> Path:
        return self.splits_dir / self.splits_filename
