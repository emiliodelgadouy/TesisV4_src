from __future__ import annotations

import os
from pathlib import Path


def medical_weights_dir() -> Path:
    base = Path(os.environ["MEDICAL_WEIGHTS_DIR"]) if os.environ.get("MEDICAL_WEIGHTS_DIR") else Path.home() / ".keras" / "medical_weights"
    base.mkdir(parents=True, exist_ok=True)
    return base
