from __future__ import annotations

import os
import shutil
import subprocess
import tarfile
import tempfile
import time
from pathlib import Path

import pandas as pd

from src.dataset.config import DatasetConfig
from src.dataset.target import BIRADS_NEGATIVE, BIRADS_POSITIVE, cls_from_birads, parse_breast_birads

_EXTRACT_MARKER_NAME = ".extract_complete"
_PROGRESS_EVERY_S = 5.0
_PARALLEL_MIN_BYTES = 64 * 1024 * 1024
_PARALLEL_PARTS = 8


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
        """Baja un objeto GCS público.

        En Colab ``gsutil`` falla si usa las credenciales de la VM: esa identidad
        no tiene IAM en el bucket, aunque el objeto sea público. Se fuerza un
        entorno anónimo (HOME / CLOUDSDK_CONFIG aislados). El tar es un objeto
        compuesto: sin crcmod compilado gsutil no puede rebanar y copia en un
        solo stream (el ``.partial`` queda en 0 B hasta el final). Solo se usa
        gsutil si ``gsutil version -l`` reporta crcmod compilado; si no, curl
        paralelo por HTTPS. ``gcloud storage`` queda como fallback.
        """
        destination_file.parent.mkdir(parents=True, exist_ok=True)
        url = self._gcs_uri_to_https(source)
        partial = destination_file.with_name(destination_file.name + ".partial")
        if partial.exists():
            partial.unlink()
        print(f"GET {url}", flush=True)

        ok = (
            self._download_with_gsutil(source, partial)
            or self._download_with_curl_parallel(url, partial)
            or self._download_with_gcloud_storage(source, partial)
            or self._download_with_curl(url, partial)
        )
        if not ok:
            self._download_with_requests(url, partial)
        if not partial.is_file() or partial.stat().st_size == 0:
            raise FileNotFoundError(f"Descarga vacia: {destination_file}")
        expected = self._https_content_length(url)
        actual = partial.stat().st_size
        if expected and actual != expected:
            partial.unlink(missing_ok=True)
            raise FileNotFoundError(f"Tamaño inesperado {actual} != {expected} para {destination_file}")
        partial.replace(destination_file)
        print(f"Guardado {destination_file} ({self._format_bytes(destination_file.stat().st_size)})", flush=True)

    @staticmethod
    def _anonymous_gcs_env(tmp: Path) -> dict[str, str]:
        """Entorno sin credenciales de Colab/gcloud, para objetos públicos."""
        home = tmp / "home"
        config = tmp / "gcloud"
        home.mkdir()
        config.mkdir()
        boto = tmp / "boto"
        boto.write_text(
            "[Boto]\n"
            "https_validate_certificates = True\n"
            "[GSUtil]\n"
            "sliced_object_download_threshold = 150M\n"
            "sliced_object_download_max_components = 8\n"
            "check_hashes = if_fast_else_skip\n",
            encoding="utf-8",
        )
        (config / "properties").write_text(
            "[core]\n"
            "disable_usage_reporting = True\n"
            "disable_prompts = True\n"
            "[storage]\n"
            "sliced_object_download_threshold = 150Mi\n"
            "sliced_object_download_max_components = 8\n",
            encoding="utf-8",
        )
        env = os.environ.copy()
        for key in (
            "GOOGLE_APPLICATION_CREDENTIALS",
            "CLOUDSDK_AUTH_ACCESS_TOKEN",
            "CLOUDSDK_AUTH_ACCESS_TOKEN_FILE",
            "CLOUDSDK_CORE_ACCOUNT",
            "CLOUDSDK_CONFIG",
            "BOTO_CONFIG",
            "BOTO_PATH",
        ):
            env.pop(key, None)
        env["HOME"] = str(home)
        env["CLOUDSDK_CONFIG"] = str(config)
        env["CLOUDSDK_AUTH_DISABLE_CREDENTIALS"] = "True"
        env["CLOUDSDK_CORE_DISABLE_PROMPTS"] = "1"
        env["BOTO_CONFIG"] = str(boto)
        env["BOTO_PATH"] = str(boto)
        # Saltea el metadata de GCE/Colab (NO_GCE_CHECK). No usar un host
        # inválido: cada lookup espera el timeout de DNS y parece una descarga
        # trabada en 0 B.
        env["NO_GCE_CHECK"] = "True"
        env["GCE_METADATA_TIMEOUT"] = "0"
        return env

    @staticmethod
    def _bytes_on_disk(destination_file: Path) -> int:
        """Incluye ``.gstmp`` / slices al lado del destino (gsutil no escribe el .partial hasta el final)."""
        parent = destination_file.parent
        if not parent.is_dir():
            return 0
        total = 0
        prefix = destination_file.name
        for path in parent.iterdir():
            if not path.is_file():
                continue
            if path.suffix == ".log" or not path.name.startswith(prefix):
                continue
            try:
                total += path.stat().st_size
            except OSError:
                continue
        return total

    def _run_copy_with_progress(self, cmd: list[str], destination_file: Path, env: dict[str, str]) -> bool:
        log_path = destination_file.with_name(destination_file.name + ".log")
        started = time.monotonic()
        with log_path.open("wb") as log_f:
            try:
                proc = subprocess.Popen(cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT)
            except FileNotFoundError as exc:
                print(f"[download] no se pudo lanzar {cmd[0]}: {exc}", flush=True)
                return False
            last_print = time.monotonic()
            while proc.poll() is None:
                time.sleep(1.0)
                now = time.monotonic()
                if now - last_print >= _PROGRESS_EVERY_S:
                    size = self._bytes_on_disk(destination_file)
                    elapsed = now - started
                    print(f"[download] {self._format_bytes(size)} ({elapsed:.0f}s)", flush=True)
                    last_print = now
            code = proc.wait()
        if code != 0:
            detail = log_path.read_text(encoding="utf-8", errors="replace")[-1500:]
            print(f"[download] {' '.join(cmd[:3])} exit {code}. {detail.strip()}", flush=True)
            if destination_file.exists():
                destination_file.unlink()
            log_path.unlink(missing_ok=True)
            return False
        log_path.unlink(missing_ok=True)
        return destination_file.is_file() and destination_file.stat().st_size > 0

    @staticmethod
    def _gsutil_has_compiled_crcmod(gsutil: str) -> bool:
        env = os.environ.copy()
        env["NO_GCE_CHECK"] = "True"
        env["GCE_METADATA_TIMEOUT"] = "0"
        try:
            proc = subprocess.run(
                [gsutil, "version", "-l"],
                capture_output=True,
                text=True,
                timeout=20,
                check=False,
                env=env,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            return False
        text = f"{proc.stdout}\n{proc.stderr}".lower()
        return "compiled crcmod: true" in text

    def _download_with_gsutil(self, source: str, destination_file: Path) -> bool:
        gsutil = shutil.which("gsutil")
        if gsutil is None:
            return False
        if not self._gsutil_has_compiled_crcmod(gsutil):
            print("[download] gsutil sin crcmod compilado; no parte en rebanadas, uso curl paralelo", flush=True)
            return False
        print("[download] gsutil anónimo (rebanadas, sin credenciales Colab)", flush=True)
        with tempfile.TemporaryDirectory(prefix="gcs-anon-") as tmp:
            env = self._anonymous_gcs_env(Path(tmp))
            cmd = [
                gsutil,
                "-o",
                "GSUtil:sliced_object_download_threshold=150M",
                "-o",
                "GSUtil:sliced_object_download_max_components=8",
                "-o",
                "GSUtil:check_hashes=if_fast_else_skip",
                "cp",
                source,
                str(destination_file),
            ]
            return self._run_copy_with_progress(cmd, destination_file, env)

    def _download_with_gcloud_storage(self, source: str, destination_file: Path) -> bool:
        gcloud = shutil.which("gcloud")
        if gcloud is None:
            return False
        print("[download] gcloud storage cp anónimo", flush=True)
        with tempfile.TemporaryDirectory(prefix="gcs-anon-") as tmp:
            env = self._anonymous_gcs_env(Path(tmp))
            cmd = [
                gcloud,
                "--quiet",
                "storage",
                "cp",
                source,
                str(destination_file),
            ]
            return self._run_copy_with_progress(cmd, destination_file, env)

    def _https_content_length(self, url: str) -> int:
        try:
            size, _ = self._https_size_and_ranges(url)
            return size
        except Exception:
            return 0

    def _https_size_and_ranges(self, url: str) -> tuple[int, bool]:
        import requests  # type: ignore

        response = requests.head(url, timeout=30, allow_redirects=True)
        response.raise_for_status()
        size = int(response.headers.get("content-length") or 0)
        accept = response.headers.get("accept-ranges", "").lower()
        return size, accept == "bytes"

    @staticmethod
    def _byte_ranges(size: int, parts: int) -> list[tuple[int, int]]:
        parts = max(1, min(parts, size))
        chunk = size // parts
        ranges: list[tuple[int, int]] = []
        start = 0
        for index in range(parts):
            end = size - 1 if index == parts - 1 else start + chunk - 1
            ranges.append((start, end))
            start = end + 1
        return ranges

    def _download_with_curl_parallel(self, url: str, destination_file: Path, *, parts: int = _PARALLEL_PARTS) -> bool:
        curl = shutil.which("curl")
        if curl is None:
            return False
        try:
            size, supports_ranges = self._https_size_and_ranges(url)
        except Exception as exc:
            print(f"[download] HEAD fallo, curl simple: {exc}", flush=True)
            return False
        if size < _PARALLEL_MIN_BYTES or not supports_ranges:
            return False

        ranges = self._byte_ranges(size, parts)
        parts_dir = destination_file.parent / (destination_file.name + ".parts")
        if parts_dir.exists():
            shutil.rmtree(parts_dir)
        parts_dir.mkdir()
        print(f"[download] curl paralelo {len(ranges)}x ({self._format_bytes(size)})", flush=True)

        jobs: list[tuple[subprocess.Popen, Path, int, object, Path]] = []
        try:
            for index, (start, end) in enumerate(ranges):
                part = parts_dir / f"{index:02d}"
                err_path = parts_dir / f"{index:02d}.err"
                err_file = err_path.open("wb")
                proc = subprocess.Popen(
                    [
                        curl,
                        "-L",
                        "--fail",
                        "--retry",
                        "5",
                        "--retry-delay",
                        "2",
                        "-s",
                        "-S",
                        "-r",
                        f"{start}-{end}",
                        "-o",
                        str(part),
                        url,
                    ],
                    stderr=err_file,
                )
                jobs.append((proc, part, end - start + 1, err_file, err_path))

            last_print = time.monotonic()
            while True:
                failed = [job for job in jobs if job[0].poll() not in (None, 0)]
                running = [job for job in jobs if job[0].poll() is None]
                now = time.monotonic()
                if now - last_print >= _PROGRESS_EVERY_S or not running or failed:
                    done = sum(job[1].stat().st_size if job[1].exists() else 0 for job in jobs)
                    print(
                        f"[download] {self._format_bytes(done)} / {self._format_bytes(size)}",
                        flush=True,
                    )
                    last_print = now
                if failed:
                    detail = failed[0][4].read_text(encoding="utf-8", errors="replace") if failed[0][4].exists() else ""
                    print(f"[download] curl paralelo fallo, pruebo curl simple. {detail.strip()}", flush=True)
                    return False
                if not running:
                    break
                time.sleep(1.0)

            for proc, part, expected, _, err_path in jobs:
                if proc.wait() != 0 or not part.is_file() or part.stat().st_size != expected:
                    detail = err_path.read_text(encoding="utf-8", errors="replace") if err_path.exists() else ""
                    print(
                        f"[download] rango incompleto {part.name}: "
                        f"{part.stat().st_size if part.exists() else 0} != {expected}. {detail.strip()}",
                        flush=True,
                    )
                    return False

            with destination_file.open("wb") as out:
                for _, part, _, _, _ in jobs:
                    with part.open("rb") as handle:
                        shutil.copyfileobj(handle, out, 8 * 1024 * 1024)
            if destination_file.stat().st_size != size:
                print(
                    f"[download] tamaño final {destination_file.stat().st_size} != {size}",
                    flush=True,
                )
                destination_file.unlink(missing_ok=True)
                return False
            return True
        finally:
            for proc, _, _, err_file, _ in jobs:
                try:
                    err_file.close()
                except Exception:
                    pass
                if proc.poll() is None:
                    proc.kill()
            shutil.rmtree(parts_dir, ignore_errors=True)

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
