"""Lectura de corridas terminadas del proyecto Comet (resumen y pesos finales)."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any

import comet_ml

from src.tracking.comet import COMET_API_KEY

# Una corrida cuenta como terminada si llego a loguear el reporte final.
_FINISHED_METRICS = ("test_roc_auc", "val_best_metric")
_SUMMARY_METRICS = (
    "val_best_metric",
    "val_roc_auc",
    "val_pr_auc",
    "test_roc_auc",
    "test_pr_auc",
    "restore_epoch",
    "restore_stage",
    "thr_youden",
    "training_wall_seconds",
)


def _slug(value: str) -> str:
    return re.sub(r"[-_]", "", str(value)).lower()


def _as_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class CometRunIndex:
    """Corridas no archivadas del proyecto, indexadas por nombre de experimento."""

    def __init__(self, config: dict[str, Any]) -> None:
        comet_cfg = config["COMET"]
        self.api_key = comet_cfg.get("API_KEY") or COMET_API_KEY or None
        self.workspace = comet_cfg.get("WORKSPACE")
        self.project_name = comet_cfg["PROJECT_NAME"]
        self._api = None
        self._by_name: dict[str, list] | None = None
        self._finished: dict[str, dict | None] = {}

    @property
    def api(self):
        if self._api is None:
            self._api = comet_ml.API(api_key=self.api_key) if self.api_key else comet_ml.API()
        return self._api

    def _resolve_project(self) -> tuple[str, str]:
        workspace = self.workspace or self.api.get_default_workspace()
        for item in self.api.get(workspace) or []:
            name = item if isinstance(item, str) else getattr(item, "name", str(item))
            if name == self.project_name or _slug(name) == _slug(self.project_name):
                return workspace, name
        return workspace, self.project_name

    def refresh(self) -> None:
        workspace, project = self._resolve_project()
        by_name: dict[str, list] = {}
        for exp in self.api.get_experiments(workspace, project) or []:
            by_name.setdefault(str(exp.name), []).append(exp)
        self._by_name = by_name
        self._finished = {}

    def forget(self, name: str) -> None:
        """Invalida lo cacheado para ``name`` (p. ej. despues de correrla en esta sesion)."""
        self._finished.pop(name, None)
        self._by_name = None

    def _candidates(self, name: str) -> list:
        if self._by_name is None:
            self.refresh()
        rows = []
        for exp in self._by_name.get(name, []):
            meta = exp.get_metadata() or {}
            start_ms = int(meta.get("startTimeMillis") or 0)
            started = datetime.fromtimestamp(start_ms / 1000).astimezone() if start_ms else None
            rows.append((start_ms, started, exp))
        rows.sort(key=lambda row: row[0], reverse=True)
        return rows

    def experiments_named(self, name: str, *, fresh: bool = False) -> list:
        """Experimentos no archivados con ese nombre, el mas nuevo primero."""
        if fresh:
            self._by_name = None
        return [exp for _, _, exp in self._candidates(name)]

    def finished(self, name: str) -> dict | None:
        """Resumen de la corrida terminada mas reciente con ese nombre, o None."""
        if name in self._finished:
            return self._finished[name]
        found = None
        for _, started, exp in self._candidates(name):
            metrics = {item.get("name"): item.get("valueCurrent") for item in exp.get_metrics_summary() or []}
            if not all(key in metrics for key in _FINISHED_METRICS):
                continue
            found = {
                "experiment": name,
                "comet_key": exp.id,
                "comet_url": exp.url,
                "started_at": started.isoformat(timespec="seconds") if started else None,
                "source": "comet",
            }
            for key in _SUMMARY_METRICS:
                value = _as_float(metrics.get(key))
                if value is not None:
                    found[key] = value
            break
        self._finished[name] = found
        return found

    def download_final_weights(self, name: str, dest_dir: str | Path) -> Path:
        """Baja ``{name}_final_weights`` (``log_model``) de la corrida terminada mas reciente."""
        summary = self.finished(name)
        if summary is None:
            raise FileNotFoundError(f"No hay una corrida terminada '{name}' en Comet")
        exp = self.api.get_experiment_by_key(summary["comet_key"])
        model_name = f"{name}_final_weights"
        if model_name not in (exp.get_model_names() or []):
            raise FileNotFoundError(f"{name} ({summary['comet_key']}) no tiene el modelo {model_name}")
        # Una corrida retomada puede subir el modelo dos veces: vale el asset mas reciente.
        assets = [item for item in exp.get_model_asset_list(model_name) or [] if str(item.get("fileName", "")).endswith(".h5")]
        if not assets:
            raise FileNotFoundError(f"El modelo {model_name} no tiene un .h5")
        asset = max(assets, key=lambda item: int(item.get("createdAt") or 0))
        dest_dir = Path(dest_dir) / name
        target = dest_dir / f"{asset['assetId']}_{asset['fileName']}"
        if target.is_file() and target.stat().st_size == int(asset.get("fileSize") or -1):
            return target
        dest_dir.mkdir(parents=True, exist_ok=True)
        payload = exp.get_asset(asset["assetId"], return_type="binary")
        tmp = target.with_name(target.name + ".tmp")
        tmp.write_bytes(payload)
        tmp.replace(target)
        return target
