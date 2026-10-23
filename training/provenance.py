"""Procedencia de una corrida y chequeo de leakage entre splits.

Lo usan ``TrainingExperiment`` (cada corrida lo loguea sola) y ``ExperimentPlan``.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from pathlib import Path

LEAKAGE_COLUMNS = ("patient_id", "study_id")


def sha1(payload) -> str:
    return hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:12]


def src_commit() -> str:
    """Commit de ``src`` (con ``-dirty`` si hay cambios sin commitear)."""
    src_dir = Path(__file__).resolve().parents[1]
    try:
        head = subprocess.run(["git", "-C", str(src_dir), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(src_dir), "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{head}-dirty" if dirty else head


def environment() -> dict[str, str]:
    """Versiones y GPU del entorno. En Colab no se instalan los requirements: la
    version de TensorFlow la decide Colab, asi que se registra por corrida."""
    import comet_ml
    import keras
    import tensorflow as tf

    gpu_names = []
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            gpu_names.append(str(tf.config.experimental.get_device_details(gpu).get("device_name") or gpu.name))
        except Exception:
            gpu_names.append(gpu.name)
    return {
        "PROVENANCE_PYTHON": platform.python_version(),
        "PROVENANCE_TENSORFLOW": tf.__version__,
        "PROVENANCE_KERAS": keras.__version__,
        "PROVENANCE_COMET_ML": comet_ml.__version__,
        "PROVENANCE_GPU": ", ".join(gpu_names) or "cpu",
    }


def split_sha1(frames: dict) -> str | None:
    if any(frame is None for frame in frames.values()):
        return None
    return sha1({name: sorted(frame["image_id"].astype(str)) for name, frame in frames.items() if "image_id" in frame.columns})


def check_split_leakage(frames: dict) -> None:
    """Falla si un paciente o estudio cae en dos splits (``frames``: ``{split: DataFrame}``)."""
    for column in LEAKAGE_COLUMNS:
        if not all(frame is not None and column in frame.columns for frame in frames.values()):
            continue
        groups = {name: set(frame[column].astype(str)) for name, frame in frames.items()}
        names = list(groups)
        overlaps = {f"{a}∩{b}": len(groups[a] & groups[b]) for i, a in enumerate(names) for b in names[i + 1 :] if groups[a] & groups[b]}
        if overlaps:
            raise ValueError(f"Leakage de {column} entre splits: {overlaps}")


def provenance(config, frames: dict) -> dict[str, str | None]:
    """``PROVENANCE_*``: commit de ``src``, hash de splits y de config (sin ``COMET``), versiones y GPU."""
    return {
        "PROVENANCE_SRC_COMMIT": src_commit(),
        "PROVENANCE_SPLIT_SHA1": split_sha1(frames),
        "PROVENANCE_CONFIG_SHA1": sha1({key: value for key, value in config.items() if key != "COMET"}),
        **environment(),
    }
