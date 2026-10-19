"""Plan de experimentos por fases: correr, saltear lo hecho, retomar y comparar por validacion.

El estado de cada corrida vive en Comet (``RESUME.BACKEND="comet"``, assets
``resume_*`` del experimento) o en una carpeta (``"dir"``, ``RESUME.DIR``). Una
corrida se considera hecha si su estado dice ``done`` o, con
``RESUME.SKIP_IF_IN_COMET``, si Comet ya tiene una corrida terminada con el
mismo nombre (desde ``RESUME.COMET_DONE_SINCE``). Una corrida a medias se
retoma desde la ultima epoca guardada.

Las decisiones entre fases miran solo validacion (``val_best_metric``, la
metrica del checkpoint). Test se muestra al final, con ``show_test=True``.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from src.tracking.comet_results import CometRunIndex
from src.tracking.comet_store import CometRunStore
from src.training.experiment import TrainingExperiment, resolve_experiment_name
from src.training.inference import build_trained_model
from src.training.mode import TrainingMode
from src.training.resources import RandomSeeds
from src.training.resume import RunStore

DRIVE_ROOT = Path("/content/drive")
DEFAULT_MARGIN = 0.02


@dataclass(frozen=True)
class RunSpec:
    """Una corrida del plan. ``seed=None`` usa ``GENERAL.RANDOM_SEED`` y no agrega sufijo."""

    mode: str
    backbone: str
    input_size: int | tuple[int, int] | None = None
    seed: int | None = None
    pretrained: RunSpec | None = None

    @property
    def suffix(self) -> str | None:
        return f"seed{self.seed}" if self.seed is not None else None

    def with_seed(self, seed: int | None) -> RunSpec:
        return replace(self, seed=seed)

    def name(self, config) -> str:
        name, _ = resolve_experiment_name(
            config,
            self.mode,
            self.backbone,
            input_size=self.input_size,
            suffix=self.suffix,
        )
        return name


def prepare_resume(config) -> str | None:
    """Valida ``RESUME``. Con ``BACKEND="dir"`` resuelve ``RESUME.DIR`` y, si apunta
    a Drive sin montar, intenta montarlo; si no puede, cae a ``RESUME.FALLBACK_DIR``.
    """
    from src.notebook import is_running_in_colab

    resume_cfg = config.setdefault("RESUME", {})
    if not resume_cfg.get("ENABLED", True):
        print("[resume] desactivado (RESUME.ENABLED=False)")
        return None
    backend = resume_cfg.setdefault("BACKEND", "comet")
    if backend == "comet":
        print(f"[resume] estado y pesos como assets resume_* del experimento en Comet ({config['COMET']['PROJECT_NAME']})")
        return "comet"
    if backend != "dir":
        raise ValueError(f"RESUME.BACKEND debe ser 'comet' o 'dir', no {backend!r}")
    target = Path(resume_cfg.get("DIR") or "runs")
    under_drive = target.is_absolute() and DRIVE_ROOT in target.parents
    if under_drive and not (DRIVE_ROOT / "MyDrive").is_dir():
        if is_running_in_colab() and resume_cfg.get("MOUNT_DRIVE", True):
            try:
                from google.colab import drive  # type: ignore

                drive.mount(str(DRIVE_ROOT), timeout_ms=int(resume_cfg.get("MOUNT_TIMEOUT_MS", 120_000)))
            except Exception as exc:
                print(f"[resume] no pude montar Drive ({exc})")
        if not (DRIVE_ROOT / "MyDrive").is_dir():
            fallback = Path(resume_cfg.get("FALLBACK_DIR") or "runs")
            print(
                f"[resume] Drive no esta montado; uso {fallback.resolve()} "
                "(sobrevive a reiniciar el kernel, no a perder la VM)"
            )
            target = fallback
    target.mkdir(parents=True, exist_ok=True)
    resume_cfg["DIR"] = str(target)
    print(f"[resume] estado de corridas en {target.resolve()}")
    return target


class ExperimentPlan:
    """Corre ``RunSpec`` salteando lo terminado y retomando lo cortado."""

    def __init__(self, config, train_df, val_df, test_df) -> None:
        self.config = config
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        resume_cfg = config.get("RESUME") or {}
        self.backend = resume_cfg.get("BACKEND", "comet") if resume_cfg.get("ENABLED", True) else None
        self.resume_dir = resume_cfg.get("DIR") or "runs"
        self.skip_if_in_comet = bool(resume_cfg.get("SKIP_IF_IN_COMET", True))
        self.comet = CometRunIndex(config, done_since=resume_cfg.get("COMET_DONE_SINCE"))
        plan_cfg = config.get("PLAN") or {}
        self.margin = float(plan_cfg.get("MARGIN", DEFAULT_MARGIN))
        self.weights_cache = Path(plan_cfg.get("WEIGHTS_CACHE", "checkpoints/comet"))
        self._stores: dict = {}
        self._summaries: dict[str, dict] = {}

    def name(self, spec: RunSpec) -> str:
        return spec.name(self.config)

    def _store(self, spec: RunSpec):
        if self.backend is None:
            return None
        name = self.name(spec)
        if name not in self._stores:
            if self.backend == "comet":
                self._stores[name] = CometRunStore(self.comet, name)
            else:
                self._stores[name] = RunStore(self.resume_dir, name)
        return self._stores[name]

    def _remember(self, name: str, summary: dict | None) -> None:
        if summary:
            self._summaries[name] = {**summary, "source": "run"}
        self.comet.forget(name)

    def status(self, spec: RunSpec) -> tuple[str, dict | None]:
        """``done`` | ``partial`` | ``comet`` | ``pending`` y el resumen si existe."""
        name = self.name(spec)
        if name in self._summaries:
            return "done", self._summaries[name]
        store = self._store(spec)
        state = store.load() if store is not None else None
        if state and state.get("status") == "done":
            return "done", {**(state.get("summary") or {}), "source": "resume"}
        if state and state.get("status") in ("training", "trained"):
            return "partial", None
        if self.skip_if_in_comet:
            summary = self.comet.finished(self.name(spec))
            if summary is not None:
                return "comet", summary
        return "pending", None

    def lookup(self, spec: RunSpec) -> dict | None:
        status, summary = self.status(spec)
        if summary is not None:
            return summary
        if status == "partial":
            return None
        return self.comet.finished(self.name(spec))

    def show(self, specs) -> pd.DataFrame:
        rows = []
        for spec in specs:
            status, summary = self.status(spec)
            rows.append(
                {
                    "experiment": self.name(spec),
                    "status": status,
                    "val_best_metric": (summary or {}).get("val_best_metric"),
                }
            )
        return pd.DataFrame(rows)

    def _config_for(self, spec: RunSpec):
        config = copy.deepcopy(self.config)
        if spec.seed is not None:
            config["GENERAL"]["RANDOM_SEED"] = int(spec.seed)
        return config

    def _experiment(self, spec: RunSpec, config, **kwargs) -> TrainingExperiment:
        return TrainingExperiment(
            config,
            spec.mode,
            spec.backbone,
            self.train_df,
            self.val_df,
            self.test_df,
            experiment_suffix=spec.suffix,
            input_size=spec.input_size,
            resume_store=self._store(spec),
            **kwargs,
        )

    def _fresh_status(self, spec: RunSpec) -> tuple[str, dict | None]:
        """Relee el estado: otra sesion (o un intento fallido en esta) pudo haberlo cambiado."""
        store = self._store(spec)
        if store is not None:
            store.reload()
        return self.status(spec)

    def run(self, spec: RunSpec) -> dict:
        name = self.name(spec)
        status, summary = self._fresh_status(spec)
        if status in ("done", "comet"):
            print(f"[plan] {name}: ya terminada ({status}); se saltea")
            return summary
        print(f"\n[plan] {name}: {'retomo' if status == 'partial' else 'arranco'}")
        config = self._config_for(spec)
        RandomSeeds.set(config)
        pretrained = self.trained_builder(spec.pretrained) if spec.pretrained is not None else None
        experiment = self._experiment(spec, config, pretrained_builder=pretrained, return_summary=True)
        try:
            experiment.run()
        finally:
            self.comet.forget(name)
        self._remember(name, experiment.summary)
        return self._summaries.get(name, {})

    def run_all(self, specs) -> pd.DataFrame:
        for spec in specs:
            self.run(spec)
        return self.table(specs)

    def trained_builder(self, spec: RunSpec):
        """Builder entrenado de ``spec`` (para transferir): entrena si falta, si no carga pesos."""
        name = self.name(spec)
        status, summary = self._fresh_status(spec)
        if status in ("pending", "partial"):
            print(f"\n[plan] {name}: entreno el encoder a transferir")
            config = self._config_for(spec)
            RandomSeeds.set(config)
            experiment = self._experiment(spec, config, return_builder=True)
            try:
                builder = experiment.run()
            finally:
                self.comet.forget(name)
            self._remember(name, experiment.summary)
            return builder
        weights = self._final_weights(spec, status)
        print(f"[plan] {name}: cargo pesos ya entrenados ({weights})")
        return build_trained_model(
            self._config_for(spec),
            spec.mode,
            spec.backbone,
            weights,
            input_size=spec.input_size,
            best_value=(summary or {}).get("val_best_metric"),
        )

    def _final_weights(self, spec: RunSpec, status: str) -> Path:
        name = self.name(spec)
        store = self._store(spec)
        if status == "done" and isinstance(store, RunStore):
            state = store.load() or {}
            for entry in state.get("best_checkpoints", []):
                if store.has_file(entry.get("file")):
                    return store.path(entry["file"])
        return self.comet.download_final_weights(name, self.weights_cache)

    def final_weights(self, spec: RunSpec) -> Path:
        status, _ = self.status(spec)
        if status in ("pending", "partial"):
            raise FileNotFoundError(f"{self.name(spec)} todavia no termino")
        return self._final_weights(spec, status)

    def table(self, specs, *, show_test: bool = False) -> pd.DataFrame:
        rows = []
        for spec in specs:
            summary = self.lookup(spec) or {}
            row = {
                "experiment": self.name(spec),
                "mode": TrainingMode.parse(spec.mode),
                "backbone": spec.backbone,
                "seed": spec.seed if spec.seed is not None else self.config["GENERAL"]["RANDOM_SEED"],
                "val_best_metric": summary.get("val_best_metric"),
                "val_roc_auc": summary.get("val_roc_auc"),
                "source": summary.get("source"),
                "comet_url": summary.get("comet_url"),
            }
            if show_test:
                row["test_roc_auc"] = summary.get("test_roc_auc")
                row["test_pr_auc"] = summary.get("test_pr_auc")
            rows.append(row)
        return pd.DataFrame(rows)

    def compare(self, reference: RunSpec, candidates, *, margin: float | None = None):
        """Candidatos vs referencia por ``val_best_metric``.

        Pasa un candidato si supera a la referencia por mas de ``margin`` (ruido
        entre repeticiones). Gana el mejor que pasa; si ninguno, la referencia.
        Devuelve ``(tabla, ganador)``.
        """
        margin = self.margin if margin is None else float(margin)
        ref_summary = self.lookup(reference)
        if ref_summary is None or ref_summary.get("val_best_metric") is None:
            raise RuntimeError(f"La referencia {self.name(reference)} no tiene val_best_metric")
        ref_value = float(ref_summary["val_best_metric"])
        rows = [{"experiment": self.name(reference), "role": "referencia", "val_best_metric": ref_value, "delta": 0.0, "pasa": None}]
        winner, winner_value = reference, ref_value
        for spec in candidates:
            summary = self.lookup(spec) or {}
            value = summary.get("val_best_metric")
            if value is None:
                rows.append({"experiment": self.name(spec), "role": "candidato", "val_best_metric": None, "delta": None, "pasa": None})
                continue
            delta = float(value) - ref_value
            passes = delta > margin
            rows.append({"experiment": self.name(spec), "role": "candidato", "val_best_metric": float(value), "delta": delta, "pasa": passes})
            if passes and float(value) > winner_value:
                winner, winner_value = spec, float(value)
        table = pd.DataFrame(rows)
        missing = table.loc[table["val_best_metric"].isna(), "experiment"].tolist()
        if missing:
            print(f"[plan] sin resultado todavia: {missing}")
        print(f"[plan] margen {margin:.3f} → gana {self.name(winner)} (val {winner_value:.4f})")
        return table, winner

    def top(self, specs, k: int = 2) -> list[RunSpec]:
        scored = []
        for spec in specs:
            value = (self.lookup(spec) or {}).get("val_best_metric")
            if value is not None:
                scored.append((float(value), spec))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [spec for _, spec in scored[:k]]

    def seed_summary(self, groups: dict) -> pd.DataFrame:
        """Media ± desvio por configuracion: ``{"etiqueta": [RunSpec por semilla, ...]}``."""
        rows = []
        for label, specs in groups.items():
            table = self.table(specs, show_test=True)
            row = {"config": label, "n": int(table["val_best_metric"].notna().sum())}
            for column in ("val_best_metric", "test_roc_auc", "test_pr_auc"):
                values = pd.to_numeric(table[column], errors="coerce").dropna().to_numpy()
                row[f"{column}_mean"] = float(np.mean(values)) if len(values) else None
                row[f"{column}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
            rows.append(row)
        return pd.DataFrame(rows)
