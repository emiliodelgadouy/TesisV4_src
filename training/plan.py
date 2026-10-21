"""Plan de experimentos por fases: repetibles (resume, semillas, procedencia) y contrastables.

Repetible:

- El estado de cada corrida vive en Comet (``RESUME.BACKEND="comet"``, assets
  ``resume_*`` del experimento) o en una carpeta (``"dir"``). Lo terminado se
  saltea y lo cortado se retoma desde la ultima epoca guardada. Con
  ``RESUME.SKIP_IF_IN_COMET`` cuenta como hecha una corrida que Comet ya tiene
  terminada (desde ``RESUME.COMET_DONE_SINCE``).
- Cada corrida fija sus semillas y loguea commit de ``src``, hash del config,
  hash de los splits y esquema de etiqueta (``GENERAL.PROVENANCE_*`` / ``LABEL*``).

Contrastable:

- La etiqueta de entrenamiento (``RunSpec.label``, esquemas en
  ``src.dataset.target.LABEL_SCHEMES``) es un factor; la tarea de evaluacion es
  fija (``PLAN.TASK``) y siempre sobre las mismas imagenes.
- ``predictions`` predice el val/test canonico completo con los pesos finales y
  lo guarda como asset ``predictions_{split}.csv`` del experimento: cualquier
  par de corridas se compara despues sin reentrenar.
- ``evaluate`` y ``pairwise`` reportan metricas clinicas y diferencias de todos
  los pares con IC 95 % (bootstrap pareado agrupado por paciente), sin elegir.
- ``compare`` / ``top`` si eligen: un candidato pasa si mejora la PR-AUC por
  mas de ``PLAN.MARGIN`` y el IC 95 % de la diferencia queda por encima de 0.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from src.dataset.target import DEFAULT_LABEL, LABEL_SCHEMES, apply_label_scheme, birads_of
from src.tracking.comet_results import CometRunIndex
from src.tracking.comet_store import CometRunStore
from src.training.evaluator import ClinicalMetrics, PairedBootstrap
from src.training.experiment import TrainingExperiment, resolve_experiment_name
from src.training.inference import build_trained_model, predict_dataframe
from src.training.mode import TrainingMode, normalize_input_size, resized_size_label
from src.training.resources import GpuResources, RandomSeeds
from src.training.resume import RunStore

DRIVE_ROOT = Path("/content/drive")
DEFAULT_MARGIN = 0.02
GROUP_COLUMN = "patient_id"
PREDICTION_COLUMNS = ("image_id", "patient_id", "study_id", "laterality", "view", "birads")


@dataclass(frozen=True)
class RunSpec:
    """Una corrida del plan.

    ``seed=None`` usa ``GENERAL.RANDOM_SEED``; ``label=None`` usa el esquema
    ``A``. Ninguno de los dos agrega sufijo al nombre en ese caso.
    """

    mode: str
    backbone: str
    input_size: int | tuple[int, int] | None = None
    seed: int | None = None
    pretrained: RunSpec | None = None
    label: str | None = None

    @property
    def label_key(self) -> str:
        return (self.label or DEFAULT_LABEL).upper()

    @property
    def suffix(self) -> str | None:
        parts = []
        if self.label_key != DEFAULT_LABEL:
            parts.append(f"lbl{self.label_key.lower()}")
        if self.seed is not None:
            parts.append(f"seed{self.seed}")
        return "_".join(parts) or None

    def with_seed(self, seed: int | None) -> RunSpec:
        return replace(self, seed=seed)

    def with_label(self, label: str | None) -> RunSpec:
        return replace(self, label=label)

    @property
    def arch(self) -> str:
        """Arquitectura para tablas: ``standard``, ``resized_672``, ``abmil``, ..."""
        mode = TrainingMode.parse(self.mode)
        if mode == TrainingMode.RESIZED and self.input_size is not None:
            return f"resized_{resized_size_label(self.input_size)}"
        return str(mode)

    @property
    def estimated_cost(self) -> float:
        """Costo relativo a ``standard`` a 224 (pixeles procesados por imagen)."""
        mode = TrainingMode.parse(self.mode)
        if mode == TrainingMode.RESIZED:
            height, width = normalize_input_size(self.input_size or 672)
            return height * width / (224 * 224)
        if mode == TrainingMode.ABMIL:
            return 9.0
        if mode == TrainingMode.ABMIL_PATCH_HARDNEG:
            return 10.0
        return 1.0

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


def src_commit() -> str:
    """Commit de ``src`` (con ``-dirty`` si hay cambios sin commitear)."""
    src_dir = Path(__file__).resolve().parents[1]
    try:
        head = subprocess.run(["git", "-C", str(src_dir), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(src_dir), "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{head}-dirty" if dirty else head


def _sha1(payload) -> str:
    return hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:12]


def _task_scheme(task) -> dict:
    if isinstance(task, str):
        return LABEL_SCHEMES[task.upper()]
    negative, positive = task
    return {"negative": tuple(negative), "positive": tuple(positive), "drop": (), "title": f"{negative} vs {positive}"}


class ExperimentPlan:
    """Corre ``RunSpec`` salteando lo terminado, retomando lo cortado y comparando por pares."""

    def __init__(self, config, train_df, val_df, test_df) -> None:
        self.config = config
        self.frames = {"train": train_df, "val": val_df, "test": test_df}
        resume_cfg = config.get("RESUME") or {}
        self.backend = resume_cfg.get("BACKEND", "comet") if resume_cfg.get("ENABLED", True) else None
        self.resume_dir = resume_cfg.get("DIR") or "runs"
        self.skip_if_in_comet = bool(resume_cfg.get("SKIP_IF_IN_COMET", True))
        self.comet = CometRunIndex(config, done_since=resume_cfg.get("COMET_DONE_SINCE"))
        plan_cfg = config.get("PLAN") or {}
        self.margin = float(plan_cfg.get("MARGIN", DEFAULT_MARGIN))
        self.task = plan_cfg.get("TASK", DEFAULT_LABEL)
        self.bootstrap_samples = int(plan_cfg.get("BOOTSTRAP_SAMPLES", 1000))
        self.aliases = dict(plan_cfg.get("ALIASES") or {})
        self.weights_cache = Path(plan_cfg.get("WEIGHTS_CACHE", "checkpoints/comet"))
        self.predictions_dir = Path(plan_cfg.get("PREDICTIONS_DIR", "predictions"))
        self._stores: dict = {}
        self._summaries: dict[str, dict] = {}
        self._predictions: dict[tuple[str, str], pd.DataFrame] = {}
        self.failures: dict[str, str] = {}
        self.provenance = {
            "PROVENANCE_SRC_COMMIT": src_commit(),
            "PROVENANCE_SPLIT_SHA1": self._split_sha1(),
            "PROVENANCE_CONFIG_SHA1": _sha1({key: value for key, value in config.items() if key != "COMET"}),
        }
        if train_df is not None:
            missing = [name for name, frame in self.frames.items() if "image_id" not in frame.columns]
            if missing:
                raise ValueError(f"Los splits {missing} no tienen image_id (hace falta para alinear predicciones)")
            self.check_splits()

    # ------------------------------------------------------------------ splits

    def _split_sha1(self) -> str | None:
        if any(frame is None for frame in self.frames.values()):
            return None
        return _sha1({name: sorted(frame["image_id"].astype(str)) for name, frame in self.frames.items() if "image_id" in frame.columns})

    def check_splits(self) -> pd.DataFrame:
        """Falla si un paciente/estudio cae en dos splits; devuelve conteos por esquema."""
        for column in ("patient_id", "study_id"):
            if not all(column in frame.columns for frame in self.frames.values()):
                continue
            groups = {name: set(frame[column].astype(str)) for name, frame in self.frames.items()}
            overlaps = {
                f"{a}∩{b}": len(groups[a] & groups[b])
                for a, b in (("train", "val"), ("train", "test"), ("val", "test"))
                if groups[a] & groups[b]
            }
            if overlaps:
                raise ValueError(f"Leakage de {column} entre splits: {overlaps}")
        rows = []
        for key, scheme in LABEL_SCHEMES.items():
            for split, frame in self.frames.items():
                labeled = apply_label_scheme(frame, scheme)
                n_pos = int((labeled["cls"] == 1).sum())
                rows.append({"label": key, "split": split, "n": len(labeled), "n_pos": n_pos, "prevalence": n_pos / max(len(labeled), 1)})
        counts = pd.DataFrame(rows)
        print(f"[plan] splits sin leakage; split={self.provenance['PROVENANCE_SPLIT_SHA1']} src={self.provenance['PROVENANCE_SRC_COMMIT']}")
        return counts

    # ------------------------------------------------------------------ nombres y estado

    def name(self, spec: RunSpec) -> str:
        return spec.name(self.config)

    def comet_name(self, spec: RunSpec) -> str:
        """Nombre en Comet: el canonico o su alias (``PLAN.ALIASES``) para corridas viejas."""
        name = self.name(spec)
        return self.aliases.get(name, name)

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
        if self.skip_if_in_comet or name != self.comet_name(spec):
            summary = self.comet.finished(self.comet_name(spec))
            if summary is not None:
                return "comet", summary
        return "pending", None

    def lookup(self, spec: RunSpec) -> dict | None:
        status, summary = self.status(spec)
        if summary is not None:
            return summary
        if status == "partial":
            return None
        return self.comet.finished(self.comet_name(spec))

    def show(self, specs) -> pd.DataFrame:
        rows = []
        for spec in specs:
            status, summary = self.status(spec)
            name = self.name(spec)
            if status == "pending" and name in self.failures:
                status = "failed"
            rows.append(
                {
                    "experiment": name,
                    "backbone": spec.backbone,
                    "arch": spec.arch,
                    "comet_name": self.comet_name(spec),
                    "status": status,
                    "cost": round(spec.estimated_cost, 1),
                    "val_best_metric": (summary or {}).get("val_best_metric"),
                }
            )
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ correr

    def _config_for(self, spec: RunSpec):
        config = copy.deepcopy(self.config)
        general = config["GENERAL"]
        if spec.seed is not None:
            general["RANDOM_SEED"] = int(spec.seed)
        scheme = LABEL_SCHEMES[spec.label_key]
        general["LABEL"] = spec.label_key
        general["LABEL_NEGATIVE"] = list(scheme["negative"])
        general["LABEL_POSITIVE"] = list(scheme["positive"])
        general["LABEL_DROP"] = list(scheme["drop"])
        general["EVAL_TASK"] = str(self.task)
        general.update(self.provenance)
        return config

    def _frames_for(self, spec: RunSpec):
        scheme = LABEL_SCHEMES[spec.label_key]
        return tuple(apply_label_scheme(self.frames[split], scheme) for split in ("train", "val", "test"))

    def _experiment(self, spec: RunSpec, config, **kwargs) -> TrainingExperiment:
        train_df, val_df, test_df = self._frames_for(spec)
        return TrainingExperiment(
            config,
            spec.mode,
            spec.backbone,
            train_df,
            val_df,
            test_df,
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

    def run_all(self, specs, *, continue_on_error: bool = False) -> pd.DataFrame:
        """Corre en orden. Con ``continue_on_error`` un fallo (p. ej. OOM) se anota en
        ``failures`` y se sigue con la siguiente; al volver a correr se reintenta."""
        for spec in specs:
            name = self.name(spec)
            try:
                self.run(spec)
                self.failures.pop(name, None)
            except Exception as exc:
                if not continue_on_error:
                    raise
                self.failures[name] = f"{type(exc).__name__}: {exc}"
                print(f"[plan] {name}: FALLO, sigo con la siguiente ({self.failures[name][:300]})")
        failed = [self.name(spec) for spec in specs if self.name(spec) in self.failures]
        if failed:
            print(f"[plan] {len(failed)} corridas fallaron en esta sesion: {failed}")
        return self.table(specs)

    @staticmethod
    def by_cost(specs) -> list[RunSpec]:
        """Lo mas barato primero (la matriz se llena rapido; lo caro queda al final)."""
        return sorted(specs, key=lambda spec: spec.estimated_cost)

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
        store = self._store(spec)
        if status == "done" and isinstance(store, RunStore):
            state = store.load() or {}
            for entry in state.get("best_checkpoints", []):
                if store.has_file(entry.get("file")):
                    return store.path(entry["file"])
        return self.comet.download_final_weights(self.comet_name(spec), self.weights_cache)

    def final_weights(self, spec: RunSpec) -> Path:
        status, _ = self.status(spec)
        if status in ("pending", "partial"):
            raise FileNotFoundError(f"{self.name(spec)} todavia no termino")
        return self._final_weights(spec, status)

    # ------------------------------------------------------------------ predicciones

    def _inference_model(self, spec: RunSpec):
        config = self._config_for(spec)
        weights = self.final_weights(spec)
        pretrained = self.trained_builder(spec.pretrained) if spec.pretrained is not None else None
        return build_trained_model(config, spec.mode, spec.backbone, weights, input_size=spec.input_size, pretrained_builder=pretrained)

    def predictions(self, spec: RunSpec, split: str = "val") -> pd.DataFrame:
        """Probabilidades del split canonico completo (todos los BI-RADS), alineadas por ``image_id``.

        Se cachean como asset ``predictions_{split}.csv`` del experimento en Comet.
        """
        key = (self.name(spec), split)
        if key in self._predictions:
            return self._predictions[key]
        summary = self.lookup(spec)
        if summary is None:
            raise FileNotFoundError(f"{self.name(spec)} todavia no termino")
        frame = self.frames[split]
        asset_name = f"predictions_{split}.csv"
        api_exp = self.comet.api.get_experiment_by_key(summary["comet_key"]) if summary.get("comet_key") else None
        cached = None
        if api_exp is not None:
            assets = [asset for asset in api_exp.get_asset_list() or [] if asset.get("fileName") == asset_name]
            if assets:
                newest = max(assets, key=lambda asset: int(asset.get("createdAt") or 0))
                cached = pd.read_csv(io.BytesIO(api_exp.get_asset(newest["assetId"], return_type="binary")), dtype={"image_id": str})
                if "birads" not in cached.columns:
                    birads_by_image = pd.Series(birads_of(frame).to_numpy(), index=frame["image_id"].astype(str))
                    cached["birads"] = cached["image_id"].map(birads_by_image)
                if set(cached["image_id"]) != set(frame["image_id"].astype(str)):
                    print(f"[plan] {asset_name} de {self.comet_name(spec)} no coincide con el split actual; recalculo")
                    cached = None
        if cached is None:
            print(f"[plan] {self.name(spec)}: predigo {split} completo ({len(frame)} imagenes)")
            model = self._inference_model(spec)
            try:
                y_prob = predict_dataframe(self._config_for(spec), model, frame, mode=spec.mode, backbone_name=spec.backbone)
            finally:
                GpuResources.dispose_model_builder(model)
                GpuResources.release(clear_keras_session=True)
            cached = frame[[column for column in PREDICTION_COLUMNS if column in frame.columns and column != "birads"]].copy()
            cached["image_id"] = cached["image_id"].astype(str)
            cached["birads"] = birads_of(frame).to_numpy()
            cached["y_prob"] = y_prob
            self.predictions_dir.mkdir(parents=True, exist_ok=True)
            path = self.predictions_dir / f"{self.name(spec)}_{asset_name}"
            cached.to_csv(path, index=False)
            if api_exp is not None:
                api_exp.log_asset(str(path), name=asset_name)
        order = frame["image_id"].astype(str).to_numpy()
        cached = cached.drop_duplicates("image_id").set_index("image_id").loc[order].reset_index()
        self._predictions[key] = cached
        return cached

    def _task_arrays(self, predictions: pd.DataFrame, task):
        scheme = _task_scheme(task)
        birads = predictions["birads"].to_numpy()
        mask = np.isin(birads, list(scheme["negative"]) + list(scheme["positive"]))
        y = np.isin(birads[mask], list(scheme["positive"])).astype(int)
        groups = predictions[GROUP_COLUMN].astype(str).to_numpy()[mask] if GROUP_COLUMN in predictions.columns else None
        return mask, y, groups

    def evaluate(self, specs, *, split: str = "val", tasks=None) -> pd.DataFrame:
        """Metricas clinicas de cada corrida sobre las mismas imagenes, por tarea."""
        tasks = list(tasks) if tasks is not None else [self.task]
        pending = [self.name(spec) for spec in specs if self.lookup(spec) is None]
        if pending:
            print(f"[plan] sin resultado todavia: {pending}")
        rows = []
        for spec in specs:
            if self.name(spec) in pending:
                continue
            predictions = self.predictions(spec, split)
            for task in tasks:
                mask, y, _ = self._task_arrays(predictions, task)
                metrics = ClinicalMetrics.compute(y, predictions["y_prob"].to_numpy()[mask])
                rows.append(
                    {
                        "experiment": self.name(spec),
                        "backbone": spec.backbone,
                        "arch": spec.arch,
                        "label": spec.label_key,
                        "seed": spec.seed if spec.seed is not None else self.config["GENERAL"]["RANDOM_SEED"],
                        "split": split,
                        "task": _task_scheme(task)["title"],
                        **metrics,
                    }
                )
        return pd.DataFrame(rows)

    def matrix(self, specs, *, split: str = "val", metric: str = "pr_auc", task=None) -> pd.DataFrame:
        """Vista backbone × arquitectura de ``metric`` (media entre semillas si hay varias)."""
        table = self.evaluate(specs, split=split, tasks=[self.task if task is None else task])
        if table.empty or metric not in table.columns:
            return pd.DataFrame()
        columns = list(dict.fromkeys(spec.arch for spec in sorted(specs, key=lambda spec: spec.estimated_cost)))
        rows = list(dict.fromkeys(spec.backbone for spec in specs))
        pivot = table.pivot_table(index="backbone", columns="arch", values=metric, aggfunc="mean")
        return pivot.reindex(index=rows, columns=columns).round(3)

    # ------------------------------------------------------------------ decidir

    def compare(self, reference: RunSpec, candidates, *, split: str = "val", task=None, margin: float | None = None):
        """Candidatos vs referencia sobre las mismas imagenes de ``split`` y la tarea ``task``.

        Pasa un candidato si ``Δ PR-AUC > margin`` y el IC 95 % pareado (bootstrap
        agrupado por paciente) queda por encima de 0. Gana el mejor que pasa; si
        ninguno, la referencia. Devuelve ``(tabla, ganador)``.
        """
        task = self.task if task is None else task
        margin = self.margin if margin is None else float(margin)
        ref_predictions = self.predictions(reference, split)
        mask, y, groups = self._task_arrays(ref_predictions, task)
        ref_prob = ref_predictions["y_prob"].to_numpy()[mask]
        ref_metrics = ClinicalMetrics.compute(y, ref_prob)
        rows = [{"experiment": self.name(reference), "role": "referencia", **ref_metrics, "pasa": None}]
        winner, winner_value = reference, ref_metrics.get("pr_auc", float("-inf"))
        for spec in candidates:
            if self.lookup(spec) is None:
                rows.append({"experiment": self.name(spec), "role": "candidato", "pasa": None})
                continue
            prob = self.predictions(spec, split)["y_prob"].to_numpy()[mask]
            delta = PairedBootstrap.auc_delta(
                y, ref_prob, prob, groups=groups, samples=self.bootstrap_samples, seed=int(self.config["GENERAL"]["RANDOM_SEED"])
            )
            metrics = ClinicalMetrics.compute(y, prob)
            passes = bool(delta["pr_auc_delta"] > margin and delta.get("pr_auc_delta_ci_low", float("-inf")) > 0)
            rows.append(
                {
                    "experiment": self.name(spec),
                    "role": "candidato",
                    **metrics,
                    "pr_auc_delta": delta["pr_auc_delta"],
                    "pr_auc_delta_ci_low": delta.get("pr_auc_delta_ci_low"),
                    "pr_auc_delta_ci_high": delta.get("pr_auc_delta_ci_high"),
                    "roc_auc_delta": delta["roc_auc_delta"],
                    "roc_auc_delta_ci_low": delta.get("roc_auc_delta_ci_low"),
                    "roc_auc_delta_ci_high": delta.get("roc_auc_delta_ci_high"),
                    "pasa": passes,
                }
            )
            if passes and metrics["pr_auc"] > winner_value:
                winner, winner_value = spec, metrics["pr_auc"]
        table = pd.DataFrame(rows)
        missing = table.loc[table["pasa"].isna() & (table["role"] == "candidato"), "experiment"].tolist()
        if missing:
            print(f"[plan] sin resultado todavia: {missing}")
        title = _task_scheme(task)["title"]
        print(f"[plan] {split}, tarea {title}, margen {margin:.3f} + IC pareado > 0 → gana {self.name(winner)} (PR-AUC {winner_value:.4f})")
        return table, winner

    def pairwise(self, specs, *, split: str = "val", task=None) -> pd.DataFrame:
        """Todas contra todas sobre las mismas imagenes, sin elegir ganador.

        Una fila por par ``(a, b)``: metricas de cada uno y ``b - a`` con IC 95 %
        por bootstrap pareado agrupado por paciente.
        """
        task = self.task if task is None else task
        done = [spec for spec in specs if self.lookup(spec) is not None]
        skipped = [self.name(spec) for spec in specs if spec not in done]
        if skipped:
            print(f"[plan] sin resultado todavia: {skipped}")
        if not done:
            return pd.DataFrame()
        mask, y, groups = self._task_arrays(self.predictions(done[0], split), task)
        probs = {self.name(spec): self.predictions(spec, split)["y_prob"].to_numpy()[mask] for spec in done}
        names = list(probs)
        rows = []
        for i, name_a in enumerate(names):
            for name_b in names[i + 1 :]:
                delta = PairedBootstrap.auc_delta(
                    y, probs[name_a], probs[name_b], groups=groups, samples=self.bootstrap_samples, seed=int(self.config["GENERAL"]["RANDOM_SEED"])
                )
                rows.append(
                    {
                        "a": name_a,
                        "b": name_b,
                        "pr_auc_a": delta["pr_auc_ref"],
                        "pr_auc_b": delta["pr_auc_alt"],
                        "pr_auc_b_minus_a": delta["pr_auc_delta"],
                        "pr_auc_ci_low": delta.get("pr_auc_delta_ci_low"),
                        "pr_auc_ci_high": delta.get("pr_auc_delta_ci_high"),
                        "roc_auc_a": delta["roc_auc_ref"],
                        "roc_auc_b": delta["roc_auc_alt"],
                        "roc_auc_b_minus_a": delta["roc_auc_delta"],
                        "roc_auc_ci_low": delta.get("roc_auc_delta_ci_low"),
                        "roc_auc_ci_high": delta.get("roc_auc_delta_ci_high"),
                    }
                )
        return pd.DataFrame(rows)

    def top(self, specs, k: int = 2, *, split: str = "val", task=None) -> list[RunSpec]:
        """Los ``k`` mejores por PR-AUC sobre las mismas imagenes."""
        task = self.task if task is None else task
        scored = []
        for spec in specs:
            if self.lookup(spec) is None:
                continue
            predictions = self.predictions(spec, split)
            mask, y, _ = self._task_arrays(predictions, task)
            scored.append((ClinicalMetrics.compute(y, predictions["y_prob"].to_numpy()[mask])["pr_auc"], spec))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [spec for _, spec in scored[:k]]

    # ------------------------------------------------------------------ reportes

    def table(self, specs, *, show_test: bool = False) -> pd.DataFrame:
        rows = []
        for spec in specs:
            summary = self.lookup(spec) or {}
            row = {
                "experiment": self.name(spec),
                "mode": TrainingMode.parse(spec.mode),
                "backbone": spec.backbone,
                "label": spec.label_key,
                "seed": spec.seed if spec.seed is not None else self.config["GENERAL"]["RANDOM_SEED"],
                "val_best_metric": summary.get("val_best_metric"),
                "source": summary.get("source"),
                "comet_url": summary.get("comet_url"),
            }
            if show_test:
                row["test_roc_auc"] = summary.get("test_roc_auc")
                row["test_pr_auc"] = summary.get("test_pr_auc")
            rows.append(row)
        return pd.DataFrame(rows)

    def seed_summary(self, groups, *, split: str = "test", task=None) -> pd.DataFrame:
        """Media ± desvio entre semillas de las metricas clinicas.

        ``groups``: ``{"config": [RunSpec, ...]}`` o una lista de ``RunSpec`` (se
        agrupan por configuracion, ignorando la semilla).
        """
        if not isinstance(groups, dict):
            grouped: dict[str, list] = {}
            for spec in groups:
                grouped.setdefault(self.name(spec.with_seed(None)), []).append(spec)
            groups = grouped
        rows = []
        for label, specs in groups.items():
            done = [spec for spec in specs if self.lookup(spec) is not None]
            metrics = self.evaluate(done, split=split, tasks=[self.task if task is None else task]) if done else pd.DataFrame()
            row = {"config": label, "n_seeds": len(done), "split": split}
            for column in ("roc_auc", "pr_auc", "pr_auc_over_baseline", "sens_at_spec_90", "sens_at_spec_95", "prec_at_recall_80"):
                values = pd.to_numeric(metrics.get(column, pd.Series(dtype=float)), errors="coerce").dropna().to_numpy()
                row[f"{column}_mean"] = float(np.mean(values)) if len(values) else None
                row[f"{column}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
            rows.append(row)
        return pd.DataFrame(rows)
