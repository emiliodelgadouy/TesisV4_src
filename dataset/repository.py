from __future__ import annotations

import os
import shutil
import subprocess
import tarfile
import time
from pathlib import Path

import pandas as pd

from src.dataset.config import DatasetConfig
from src.dataset.target import BIRADS_NEGATIVE, BIRADS_POSITIVE, cls_from_birads, parse_breast_birads

_EXTRACT_MARKER_NAME = ".extract_complete"
_PROGRESS_EVERY_S = 5.0


class DatasetRepository:
    """Descarga GCS, extrae imagenes y arma el DataFrame crudo con ``cls`` (BI-RADS)."""

    def __init__(self, config: DatasetConfig | None = None) -> None:
        self.config = config or DatasetConfig()

    def ensure_dirs(self) -> None:
        for directory in (self.config.raw_img_dir, self.config.raw_csv_dir, self.config.splits_dir):
            directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _gcs_uri_to_https(uri: str) -> str:
        assert uri.startswith("gs://"), f"URI GCS inválida: {uri}"
        return "https://storage.googleapis.com/" + uri[len("gs://") :]

    @staticmethod
    def _format_bytes(n: int) -> str:
        if n >= 1_000_000_000:
            return f"{n / 1e9:.2f} GB"
        if n >= 1_000_000:
            return f"{n / 1e6:.1f} MB"
        return f"{n} B"

    def _download_from_gcs(self, source: str, destination_file: Path) -> None:
        """Baja un objeto GCS. En Colab prioriza gsutil (rebanadas paralelas); si no, curl; si no, requests."""
        destination_file.parent.mkdir(parents=True, exist_ok=True)
        url = self._gcs_uri_to_https(source)
        partial = destination_file.with_name(destination_file.name + ".partial")
        if partial.exists():
            partial.unlink()
        print(f"GET {url}", flush=True)

        ok = self._download_with_gsutil(source, partial) or self._download_with_curl(url, partial)
        if not ok:
            self._download_with_requests(url, partial)
        if not partial.is_file() or partial.stat().st_size == 0:
            raise FileNotFoundError(f"Descarga vacia: {destination_file}")
        partial.replace(destination_file)
        print(f"Guardado {destination_file} ({self._format_bytes(destination_file.stat().st_size)})", flush=True)

    @staticmethod
    def _download_with_gsutil(source: str, destination_file: Path) -> bool:
        gsutil = shutil.which("gsutil")
        if gsutil is None:
            return False
        # Rebanadas paralelas: el tar es un objeto compuesto (~8 GB, 17 componentes).
        cmd = [
            gsutil,
            "-o",
            "GSUtil:sliced_object_download_threshold=150M",
            "-o",
            "GSUtil:sliced_object_download_max_components=8",
            "cp",
            source,
            str(destination_file),
        ]
        print("[download] gsutil cp (rebanadas paralelas)", flush=True)
        try:
            subprocess.run(cmd, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError, OSError) as exc:
            print(f"[download] gsutil no disponible, pruebo curl: {exc}", flush=True)
            if destination_file.exists():
                destination_file.unlink()
            return False
        return destination_file.is_file() and destination_file.stat().st_size > 0

    @staticmethod
    def _download_with_curl(url: str, destination_file: Path) -> bool:
        curl = shutil.which("curl")
        if curl is None:
            return False
        cmd = [
            curl,
            "-L",
            "--fail",
            "--retry",
            "5",
            "--retry-delay",
            "2",
            "--progress-bar",
            "-o",
            str(destination_file),
            url,
        ]
        print("[download] curl", flush=True)
        try:
            subprocess.run(cmd, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError, OSError) as exc:
            print(f"[download] curl fallo, pruebo requests: {exc}", flush=True)
            if destination_file.exists():
                destination_file.unlink()
            return False
        return destination_file.is_file() and destination_file.stat().st_size > 0

    def _download_with_requests(self, url: str, destination_file: Path) -> None:
        import requests  # type: ignore

        print("[download] requests", flush=True)
        downloaded = 0
        last_print = time.monotonic()
        with requests.get(url, stream=True, timeout=60) as response:
            response.raise_for_status()
            total = int(response.headers.get("content-length") or 0)
            if total:
                print(f"[download] {self._format_bytes(total)}", flush=True)
            with destination_file.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                    if not chunk:
                        continue
                    handle.write(chunk)
                    downloaded += len(chunk)
                    now = time.monotonic()
                    if now - last_print >= _PROGRESS_EVERY_S:
                        if total:
                            print(
                                f"[download] {self._format_bytes(downloaded)} / {self._format_bytes(total)}",
                                flush=True,
                            )
                        else:
                            print(f"[download] {self._format_bytes(downloaded)}", flush=True)
                        last_print = now

    def _extract_tar_gz(self, archive: Path, destination: Path) -> None:
        """GNU tar es bastante mas rapido que tarfile de Python para este .tar.gz."""
        tar_bin = shutil.which("tar")
        if tar_bin is not None:
            cmd = [tar_bin, "-xzf", str(archive), "-C", str(destination)]
            version = subprocess.run([tar_bin, "--version"], capture_output=True, text=True, check=False)
            if "GNU tar" in (version.stdout or ""):
                cmd.extend(
                    [
                        "--checkpoint=2000",
                        "--checkpoint-action=echo=[extract] checkpoint %d",
                    ]
                )
            print(f"[extract] {tar_bin} -xzf {archive.name}", flush=True)
            subprocess.run(cmd, check=True)
            return
        print("[extract] Python tarfile (mas lento; no hay tar en PATH)", flush=True)
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(destination)

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
            print("Downloading images...", flush=True)
            self._download_from_gcs(config.gcs_images_tar, config.tar_local)
            if not config.tar_local.is_file() or config.tar_local.stat().st_size == 0:
                raise FileNotFoundError(f"No se pudo descargar {config.tar_local}.")
            print("Images downloaded", flush=True)

        if config.extract_images and not self._has_extracted_images():
            print("Extracting images...", flush=True)
            self._extract_tar_gz(config.tar_local, config.raw_img_dir)
            self._extract_marker_path().write_text("ok\n", encoding="utf-8")
            print(f"Images extracted to {config.raw_img_dir}", flush=True)

        if not config.csv_main.is_file() or config.csv_main.stat().st_size == 0:
            print("Downloading CSV...", flush=True)
            self._download_from_gcs(config.gcs_data_csv, config.csv_main)
            print("CSV downloaded", flush=True)

        if not config.splits_local.is_file() or config.splits_local.stat().st_size == 0:
            print("Downloading dataset splits...", flush=True)
            self._download_from_gcs(config.gcs_splits_json, config.splits_local)
            print(f"Splits downloaded to {config.splits_local}", flush=True)

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
        """``cls``: BI-RADS 1-2 -> 0, BI-RADS 3-5 -> 1. El resto queda en NA."""
        column = self.config.birads_column
        if column not in df.columns:
            raise KeyError(f"Falta la columna {column} para generar {self.config.cls_column}")

        df = df.copy()
        df["birads"] = parse_breast_birads(df[column])
        df[self.config.cls_column] = cls_from_birads(df[column])
        n_pos = int(df[self.config.cls_column].eq(1).sum())
        n_neg = int(df[self.config.cls_column].eq(0).sum())
        n_drop = int(df[self.config.cls_column].isna().sum())
        print(
            f"cls desde {column}: {n_neg} neg (BI-RADS {list(BIRADS_NEGATIVE)}) / "
            f"{n_pos} pos (BI-RADS {list(BIRADS_POSITIVE)}); "
            f"{n_drop} filas sin assessment 1-5"
        )
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
        ds_raw = self.add_cls_column(ds_raw)
        ds = ds_raw[ds_raw[self.config.cls_column].notna()].copy()
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
