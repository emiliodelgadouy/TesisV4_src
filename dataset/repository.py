from __future__ import annotations

import os
import tarfile
from pathlib import Path

import pandas as pd

from src.dataset.config import DatasetConfig

_EXTRACT_MARKER_NAME = ".extract_complete"


class DatasetRepository:
    """Descarga GCS, extrae imagenes y arma el DataFrame crudo con ``cls``."""

    def __init__(self, config: DatasetConfig | None = None) -> None:
        self.config = config or DatasetConfig()

    def ensure_dirs(self) -> None:
        for directory in (self.config.raw_img_dir, self.config.raw_csv_dir, self.config.splits_dir):
            directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _gcs_uri_to_https(uri: str) -> str:
        assert uri.startswith("gs://"), f"URI GCS inválida: {uri}"
        return "https://storage.googleapis.com/" + uri[len("gs://") :]

    def _download_from_gcs(self, source: str, destination_file: Path) -> None:
        import requests  # type: ignore

        destination_file.parent.mkdir(parents=True, exist_ok=True)
        url = self._gcs_uri_to_https(source)
        print(f"GET {url}")
        with requests.get(url, stream=True, timeout=300) as response:
            response.raise_for_status()
            with destination_file.open("wb") as f:
                for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                    f.write(chunk)

    def _extract_marker_path(self) -> Path:
        return self.config.raw_img_dir / _EXTRACT_MARKER_NAME

    def _has_extracted_images(self) -> bool:
        # Marker escrito al terminar extractall; evita saltar re-extract por residuos parciales.
        return self._extract_marker_path().is_file()

    def ensure_downloaded(self) -> None:
        self.ensure_dirs()
        config = self.config
        if not config.download_from_gcs:
            return

        if not config.tar_local.is_file() or config.tar_local.stat().st_size == 0:
            print("Downloading images...")
            self._download_from_gcs(config.gcs_images_tar, config.tar_local)
            if not config.tar_local.is_file() or config.tar_local.stat().st_size == 0:
                raise FileNotFoundError(f"No se pudo descargar {config.tar_local}.")
            print("Images downloaded")

        if config.extract_images and not self._has_extracted_images():
            print("Extracting images...")
            with tarfile.open(config.tar_local, "r:gz") as tar:
                tar.extractall(config.raw_img_dir)
            self._extract_marker_path().write_text("ok\n", encoding="utf-8")
            print(f"Images extracted to {config.raw_img_dir}")

        if not config.csv_main.is_file() or config.csv_main.stat().st_size == 0:
            print("Downloading CSV...")
            self._download_from_gcs(config.gcs_data_csv, config.csv_main)
            print("CSV downloaded")

        if not config.splits_local.is_file() or config.splits_local.stat().st_size == 0:
            print("Downloading dataset splits...")
            self._download_from_gcs(config.gcs_splits_json, config.splits_local)
            print(f"Splits downloaded to {config.splits_local}")

    def load_raw_dataframe(self) -> pd.DataFrame:
        self.ensure_downloaded()
        if not self.config.csv_main.is_file():
            raise FileNotFoundError(f"No existe el CSV principal: {self.config.csv_main}")
        ds_raw = pd.read_csv(self.config.csv_main, low_memory=False)
        ds_raw["path"] = ds_raw.apply(
            lambda row: str(self.config.raw_img_dir / str(row["patient_id"]) / str(row["image_id"])),
            axis=1,
        )
        return ds_raw

    def add_cls_column(self, df: pd.DataFrame) -> pd.DataFrame:
        missing_columns = [
            column for column in self.config.cls_positive_columns if column not in df.columns
        ]
        if missing_columns:
            raise KeyError(f"Faltan columnas para generar {self.config.cls_column}: {missing_columns}")

        df = df.copy()
        df[self.config.cls_column] = (
            df[list(self.config.cls_positive_columns)].eq(self.config.cls_positive_value).any(axis=1)
        ).astype("float32")
        return df

    @staticmethod
    def filter_existing_files(df: pd.DataFrame, path_column: str = "path") -> pd.DataFrame:
        """Descarta filas cuya imagen no exista en disco (evita NotFoundError en tf.data)."""
        if df.empty or path_column not in df.columns:
            return df
        exists = df[path_column].map(os.path.exists)
        missing = int((~exists).sum())
        if missing:
            print(
                f"[dataset] {missing}/{len(df)} filas descartadas: imagen no encontrada en disco "
                "(CSV con mas entradas que el tar o extraccion incompleta)."
            )
        return df[exists].copy()

    @staticmethod
    def _sample_dataframe(df: pd.DataFrame, fraction: float, seed: int | None) -> pd.DataFrame:
        if not 0 < fraction <= 1:
            raise ValueError("sample_fraction debe estar entre 0 y 1.")
        if df.empty or fraction == 1:
            return df.copy()
        return df.sample(frac=fraction, random_state=seed).copy()

    def download_and_build(
        self,
        config: dict | None = None,
        *,
        reduced: bool | None = None,
        sample_fraction: float = 0.10,
        sample_seed: int | None = 42,
    ) -> dict[str, object]:
        """Descarga/arma el dataset. ``reduced`` sale de ``config["GENERAL"]["REDUCED_DATASET"]``."""
        if reduced is None:
            reduced = bool(config["GENERAL"]["REDUCED_DATASET"]) if config is not None else False
        ds_raw = self.load_raw_dataframe()
        missing_columns = [column for column in self.config.filter_columns if column not in ds_raw.columns]
        if missing_columns:
            raise KeyError(f"Faltan columnas para el filtrado: {missing_columns}")

        ds_raw = self.add_cls_column(ds_raw)
        ds = ds_raw[
            ds_raw[list(self.config.filter_columns)]
            .eq(self.config.cls_positive_value)
            .any(axis=1)
        ].copy()
        if reduced:
            ds = self._sample_dataframe(ds, sample_fraction, sample_seed)

        return {
            "root": self.config.root_dir,
            "data_dir": self.config.data_dir,
            "raw_img_dir": self.config.raw_img_dir,
            "raw_csv_dir": self.config.raw_csv_dir,
            "csv_main": self.config.csv_main,
            "ds_raw": ds_raw,
            "ds": ds,
            "reduced": reduced,
            "sample_fraction": sample_fraction if reduced else 1.0,
        }

    @classmethod
    def download_and_build_dataset(
        cls,
        config: dict | None = None,
        *,
        dataset_config: DatasetConfig | None = None,
        reduced: bool | None = None,
        sample_fraction: float = 0.10,
        sample_seed: int | None = 42,
    ) -> dict[str, object]:
        return cls(dataset_config).download_and_build(
            config,
            reduced=reduced,
            sample_fraction=sample_fraction,
            sample_seed=sample_seed,
        )
