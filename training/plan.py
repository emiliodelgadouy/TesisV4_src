"""Plan de experimentos: repetibles (resume, semillas, procedencia) y contrastables.

Repetible:

- El estado de cada corrida vive en Comet (``RESUME.BACKEND="comet"``, assets
  ``resume_*`` del experimento) o en una carpeta (``"dir"``). Lo terminado se
  saltea y lo cortado se retoma desde la ultima epoca guardada. Con
  ``RESUME.SKIP_IF_IN_COMET`` cuenta como hecha una corrida que Comet ya tiene
  terminada con el mismo nombre.
- Cada corrida fija sus semillas y loguea commit de ``src``, hash del config,
  hash de los splits, versiones de Python/TF/Keras/comet_ml, GPU y esquema de
  etiqueta (``GENERAL.PROVENANCE_*`` / ``LABEL*``).
- ``unique`` deduplica celdas compartidas entre cortes; ``timing`` compara horas
  reales con el costo estimado.

Contrastable:

- La etiqueta de entrenamiento (``RunSpec.label``, esquemas en
  ``src.dataset.target.LABEL_SCHEMES``) es un factor; la tarea de evaluacion es
  fija (``PLAN.TASK``) y siempre sobre las mismas imagenes.
- ``predictions`` predice el val/test canonico completo con los pesos finales y
  lo guarda como asset ``predictions_{split}.csv`` del experimento: cualquier
  par de corridas se compara despues sin reentrenar.
- ``evaluate`` y ``pairwise`` reportan metricas clinicas y diferencias de todos
  los pares con IC 95 % (bootstrap pareado agrupado por paciente), sin elegir.
"""

from __future__ import annotations

import copy
import io
import re
import tempfile
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

import comet_ml
import numpy as np
import pandas as pd

from src.backbones import get_backbone
from src.dataset.target import DEFAULT_LABEL, LABEL_SCHEMES, apply_label_scheme, birads_of
from src.tracking.comet import RUN_STATUS_CODES
from src.tracking.comet_results import CometRunIndex
from src.tracking.comet_store import CometRunStore
from src.training.evaluator import BreastAggregation, ClinicalMetrics, PairedBootstrap, ThresholdSelector
from src.training.experiment import TrainingExperiment, resolve_experiment_name
from src.training.inference import build_trained_model, predict_dataframe
from src.training.mode import TrainingMode, normalize_input_size, resized_size_label
from src.training.provenance import check_split_leakage, provenance
from src.training.resources import GpuResources, RandomSeeds
from src.training.resume import RunStore
from src.training.stage_runner import stages_from_state, training_progress

DRIVE_ROOT = Path("/content/drive")
GROUP_COLUMN = "patient_id"
SEED_SUMMARY_METRICS = (
    "roc_auc",
    "pr_auc",
    "pr_auc_over_baseline",
    "sens_at_spec_90",
    "sens_at_spec_95",
    "prec_at_recall_80",
    "sensitivity",
    "specificity",
    "ppv",
    "npv",
    "brier",
    "ece",
)
# Fraccion de FLOPs en atencion de ViT-S/16 a 224: N / (6 D) con N=196 tokens, D=384.
ATTENTION_FLOPS_SHARE_NATIVE = 196 / (6 * 384)
PREDICTION_COLUMNS = ("image_id", "patient_id", "study_id", "laterality", "view", "birads")
_RUN_NAME = re.compile(r"^(?P<prefix>.+?)(?:_lbl(?P<label>[a-z]+))?_seed(?P<seed>\d+)$")


@dataclass(frozen=True)
class RunSpec:
    """Una corrida del plan.

    ``seed=None`` usa ``GENERAL.RANDOM_SEED``; ``label=None`` usa el esquema
    ``A``. El nombre siempre lleva ``_seed{N}``; la etiqueta agrega ``_lbl{x}``
    solo si no es A.
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
        """Sufijo de etiqueta; la semilla la agrega ``resolve_experiment_name``."""
        if self.label_key != DEFAULT_LABEL:
            return f"lbl{self.label_key.lower()}"
        return None

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
        """Costo relativo a ``standard`` a 224 (pixeles procesados por imagen).

        En backbones con atencion global (ViT) la atencion crece con el cuadrado
        de los tokens, asi que RESIZED cuesta mas que el area.
        """
        mode = TrainingMode.parse(self.mode)
        if mode == TrainingMode.RESIZED:
            height, width = normalize_input_size(self.input_size or 672)
            area = height * width / (224 * 224)
            if get_backbone(self.backbone).batch_memory_scale == "attention":
                share = ATTENTION_FLOPS_SHARE_NATIVE
                return area * (1 + share * area) / (1 + share)
            return area
        if mode == TrainingMode.ABMIL:
            return 9.0
        if mode == TrainingMode.ABMIL_PATCH_HARDNEG:
            return 10.0
        return 1.0

    def name(self, config, *, include_seed: bool = True) -> str:
        name, _ = resolve_experiment_name(
            config,
            self.mode,
            self.backbone,
            input_size=self.input_size,
            suffix=self.suffix,
            seed=self.seed,
            include_seed=include_seed,
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


def log_table_asset(api_exp, table: pd.DataFrame, name: str) -> None:
    """Sube ``table`` como CSV ``name`` (se pisa) sin dejar nada en disco:
    ``APIExperiment.log_asset`` solo acepta una ruta."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / name
        table.to_csv(path, index=False)
        api_exp.log_asset(str(path), name=name, overwrite=True)


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
        self.comet = CometRunIndex(config)
        plan_cfg = config.get("PLAN") or {}
        self.task = plan_cfg.get("TASK", DEFAULT_LABEL)
        self.bootstrap_samples = int(plan_cfg.get("BOOTSTRAP_SAMPLES", 1000))
        self.weights_cache = Path(plan_cfg.get("WEIGHTS_CACHE", "checkpoints/comet"))
        # Experimento de Comet con la tabla de estado de la matriz; None/"" lo apaga.
        self.status_experiment_name = plan_cfg.get("STATUS_EXPERIMENT", "estado_matriz")
        self._status_exp = None
        self._stores: dict = {}
        self._summaries: dict[str, dict] = {}
        self._predictions: dict[tuple[str, str], pd.DataFrame] = {}
        self.failures: dict[str, str] = {}
        self.provenance = provenance(config, self.frames)
        if train_df is not None:
            missing = [name for name, frame in self.frames.items() if "image_id" not in frame.columns]
            if missing:
                raise ValueError(f"Los splits {missing} no tienen image_id (hace falta para alinear predicciones)")
            self.check_splits()

    # ------------------------------------------------------------------ splits

    def check_splits(self) -> pd.DataFrame:
        """Falla si un paciente/estudio cae en dos splits; devuelve conteos por esquema."""
        check_split_leakage(self.frames)
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
            summary = self.comet.finished(name)
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

    # ------------------------------------------------------------------ monitoreo

    def progress(self, spec: RunSpec, status: str | None = None) -> float:
        """Avance de entrenamiento en [0, 1]: 1 si termino, epocas hechas si quedo a medias."""
        status = status or self.status(spec)[0]
        if status in ("done", "comet"):
            return 1.0
        store = self._store(spec)
        if status != "partial" or store is None:
            return 0.0
        return training_progress(self.config, stages_from_state(store.load()))

    def status_table(self, specs, *, running: RunSpec | None = None) -> pd.DataFrame:
        """Una fila por corrida con estado, avance y resultados, para la vista de monitoreo.

        ``status``: ``pending`` | ``training`` (la que corre ahora) | ``partial``
        (cortada, se retoma) | ``failed`` | ``done``.
        """
        running_name = self.name(running) if running is not None else None
        rows = []
        for spec in specs:
            name = self.name(spec)
            status, summary = self.status(spec)
            progress = self.progress(spec, status)
            if status == "comet":
                status = "done"
            if name == running_name:
                status = "training"
            elif status == "pending" and name in self.failures:
                status = "failed"
            summary = summary or {}
            seconds = summary.get("training_wall_seconds")
            rows.append(
                {
                    "experiment": name,
                    "backbone": spec.backbone,
                    "arch": spec.arch,
                    "seed": spec.seed if spec.seed is not None else self.config["GENERAL"]["RANDOM_SEED"],
                    "label": spec.label_key,
                    "status": status,
                    "status_code": RUN_STATUS_CODES[status],
                    "progress": round(progress, 4),
                    "cost": round(spec.estimated_cost, 1),
                    "hours": round(float(seconds) / 3600, 3) if seconds is not None else None,
                    "val_best_metric": summary.get("val_best_metric"),
                    "test_roc_auc": summary.get("test_roc_auc"),
                    "test_pr_auc": summary.get("test_pr_auc"),
                    "error": self.failures.get(name, "")[:300],
                    "comet_url": summary.get("comet_url"),
                }
            )
        return pd.DataFrame(rows)

    def _status_experiment(self):
        """Experimento de Comet que guarda la tabla de estado de la matriz (se reusa entre sesiones)."""
        if self._status_exp is None:
            name = self.status_experiment_name
            found = self.comet.experiments_named(name, fresh=True)
            if found:
                self._status_exp = found[0]
            else:
                workspace, project = self.comet._resolve_project()
                self._status_exp = comet_ml.APIExperiment(api_key=self.comet.api_key, workspace=workspace, project_name=project, experiment_name=name)
                self._status_exp.add_tag("estado")
                self.comet.forget(name)
        return self._status_exp

    def publish_status(self, specs, *, running: RunSpec | None = None) -> pd.DataFrame:
        """Sube la tabla de estado como asset ``estado_matriz.csv`` (se pisa) y metricas
        agregadas (``runs_*``, ``hours_done``, ``matrix_progress``) al experimento
        ``PLAN.STATUS_EXPERIMENT``. Un error de red no corta la matriz."""
        table = self.status_table(specs, running=running)
        if not self.status_experiment_name or table.empty:
            return table
        try:
            exp = self._status_experiment()
            log_table_asset(exp, table, "estado_matriz.csv")
            counts = table["status"].value_counts()
            cost = table["cost"].astype(float)
            exp.log_metrics(
                {
                    "runs_total": len(table),
                    "runs_done": int(counts.get("done", 0)),
                    "runs_training": int(counts.get("training", 0)),
                    "runs_partial": int(counts.get("partial", 0)),
                    "runs_failed": int(counts.get("failed", 0)),
                    "runs_pending": int(counts.get("pending", 0)),
                    "hours_done": float(pd.to_numeric(table["hours"], errors="coerce").sum()),
                    "matrix_progress": float((cost * table["progress"]).sum() / cost.sum()) if cost.sum() else 0.0,
                }
            )
            exp.log_other("updated_at", datetime.now().astimezone().isoformat(timespec="seconds"))
            exp.log_other("running", self.name(running) if running is not None else "")
        except Exception as exc:
            print(f"[plan] no pude publicar el estado en Comet ({type(exc).__name__}: {exc})")
        return table

    def publish_report(self, table: pd.DataFrame, name: str) -> bool:
        """Sube una tabla del reporte como asset ``name`` (se pisa) del experimento
        ``PLAN.STATUS_EXPERIMENT``. En Colab no se guarda nada en disco: si falla,
        avisa y la tabla sigue en memoria para reintentar."""
        if not self.status_experiment_name or table is None or table.empty:
            return False
        try:
            exp = self._status_experiment()
            log_table_asset(exp, table, name)
            exp.log_other(f"{Path(name).stem}_updated_at", datetime.now().astimezone().isoformat(timespec="seconds"))
        except Exception as exc:
            print(f"[plan] no pude subir {name} a Comet ({type(exc).__name__}: {exc}); reintentar con plan.publish_report")
            return False
        print(f"[plan] {name} en Comet ({self.status_experiment_name})")
        return True

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
        """Relee el estado de lo no terminado: otra sesion (o un intento fallido en
        esta) pudo haberlo cambiado. Lo terminado no vuelve atras y no se relee."""
        status, summary = self.status(spec)
        if status in ("done", "comet"):
            return status, summary
        store = self._store(spec)
        if store is not None:
            store.reload()
        return self.status(spec)

    def run(self, spec: RunSpec) -> dict:
        return self._run(spec, *self._fresh_status(spec))

    def _run(self, spec: RunSpec, status: str, summary: dict | None) -> dict:
        name = self.name(spec)
        if status in ("done", "comet"):
            print(f"[plan] {name}: ya terminada ({status}); se saltea")
            return summary
        print(f"\n[plan] {name}: {'retomo' if status == 'partial' else 'arranco'}")
        config = self._config_for(spec)
        pretrained = self.trained_builder(spec.pretrained) if spec.pretrained is not None else None
        # Despues del encoder: armarlo o entrenarlo consume estado aleatorio.
        RandomSeeds.set(config)
        experiment = self._experiment(spec, config, pretrained_builder=pretrained, return_summary=True)
        try:
            experiment.run()
        finally:
            self.comet.forget(name)
        self._remember(name, experiment.summary)
        return self._summaries.get(name, {})

    def run_all(self, specs, *, continue_on_error: bool = False) -> pd.DataFrame:
        """Corre en orden. Con ``continue_on_error`` un fallo (p. ej. OOM) se anota en
        ``failures`` y se sigue con la siguiente; al volver a correr se reintenta.

        Antes y despues de cada corrida publica la tabla de estado en Comet
        (``publish_status``).
        """
        specs = list(specs)
        self.publish_status(specs)
        for spec in specs:
            name = self.name(spec)
            status, summary = self._fresh_status(spec)
            pending = status not in ("done", "comet")
            if pending:
                self.publish_status(specs, running=spec)
            try:
                self._run(spec, status, summary)
                self.failures.pop(name, None)
            except Exception as exc:
                if not continue_on_error:
                    raise
                self.failures[name] = f"{type(exc).__name__}: {exc}"
                print(f"[plan] {name}: FALLO, sigo con la siguiente ({self.failures[name][:300]})")
            finally:
                if pending:
                    self.publish_status(specs)
        failed = [self.name(spec) for spec in specs if self.name(spec) in self.failures]
        if failed:
            print(f"[plan] {len(failed)} corridas fallaron en esta sesion: {failed}")
        return self.status_table(specs)

    @staticmethod
    def by_cost(specs) -> list[RunSpec]:
        """Lo mas barato primero (la matriz se llena rapido; lo caro queda al final)."""
        return sorted(specs, key=lambda spec: spec.estimated_cost)

    def unique(self, specs) -> list[RunSpec]:
        """Sin duplicados por nombre de experimento, conservando el orden: una celda
        compartida por dos cortes se entrena una sola vez."""
        seen: dict[str, RunSpec] = {}
        for spec in specs:
            seen.setdefault(self.name(spec), spec)
        return list(seen.values())

    def spec_from_name(self, name: str) -> RunSpec | None:
        """``RunSpec`` de un nombre de experimento (``resized_672_swint_seed42``), o None.

        Solo vale si el plan vuelve a generar exactamente ese nombre: un canvas que
        ya no esta en ``RESIZED.INPUT_SIZES`` o un sufijo ajeno (p. ej. ``hpo*``) no cuenta.
        """
        match = _RUN_NAME.match(str(name))
        if match is None:
            return None
        prefix, label, seed = match.group("prefix"), match.group("label"), int(match.group("seed"))
        if label is not None and label.upper() not in LABEL_SCHEMES:
            return None
        input_size = None
        if prefix.startswith("resized_"):
            size, _, backbone = prefix[len("resized_") :].partition("_")
            mode = TrainingMode.RESIZED
            height, _, width = size.partition("x")
            if not height.isdigit() or (width and not width.isdigit()):
                return None
            input_size = int(height) if not width else (int(height), int(width))
        else:
            mode = next((mode for mode in sorted(TrainingMode.ALL, key=len, reverse=True) if prefix.startswith(f"{mode}_")), None)
            if mode is None:
                return None
            backbone = prefix[len(mode) + 1 :]
        pretrained = RunSpec(TrainingMode.PATCH_HARDNEG, backbone) if mode == TrainingMode.ABMIL_PATCH_HARDNEG else None
        spec = RunSpec(mode, backbone, input_size=input_size, seed=seed, pretrained=pretrained, label=label.upper() if label else None)
        try:
            return spec if self.name(spec) == name else None
        except Exception:
            return None

    def finished_specs(self, modes=None) -> list[RunSpec]:
        """Corridas terminadas (no archivadas) del proyecto de Comet, como ``RunSpec``.

        Sale del listado de Comet, no de una lista armada en el notebook: sirve para
        resumir todo lo que ya existe. ``modes`` filtra por modo (``["standard", "resized"]``).
        """
        wanted = {TrainingMode.parse(mode) for mode in modes} if modes is not None else None
        specs = []
        for name in self.comet.names(fresh=True):
            spec = self.spec_from_name(name)
            if spec is None or (wanted is not None and TrainingMode.parse(spec.mode) not in wanted):
                continue
            if self.lookup(spec) is not None:
                specs.append(spec)
        return sorted(specs, key=lambda spec: (TrainingMode.ALL.index(TrainingMode.parse(spec.mode)), spec.backbone, spec.estimated_cost, spec.seed, spec.label_key))

    def timing(self, specs, *, hours_per_cost: float | None = None) -> pd.DataFrame:
        """Horas reales de entrenamiento (``training_wall_seconds``) vs el costo estimado.

        ``h_per_cost`` sirve para recalibrar la estimacion de lo que falta; las
        pendientes quedan con ``hours`` vacio y ``estimated_hours`` segun la mediana
        de ``h_per_cost`` de las terminadas (o ``hours_per_cost`` si se pasa).
        """
        rows = []
        for spec in specs:
            summary = self.lookup(spec) or {}
            seconds = summary.get("training_wall_seconds")
            hours = float(seconds) / 3600 if seconds is not None else None
            cost = spec.estimated_cost
            rows.append(
                {
                    "experiment": self.name(spec),
                    "backbone": spec.backbone,
                    "arch": spec.arch,
                    "cost": round(cost, 1),
                    "hours": hours,
                    "h_per_cost": hours / cost if hours is not None else None,
                }
            )
        table = pd.DataFrame(rows)
        if table.empty:
            return table
        table[["hours", "h_per_cost"]] = table[["hours", "h_per_cost"]].apply(pd.to_numeric, errors="coerce")
        measured = table["h_per_cost"].dropna()
        rate = hours_per_cost if hours_per_cost is not None else (float(measured.median()) if len(measured) else None)
        table["estimated_hours"] = table["hours"].fillna(table["cost"] * rate) if rate is not None else table["hours"]
        done_hours = table["hours"].sum()
        remaining = table.loc[table["hours"].isna(), "estimated_hours"].sum()
        print(f"[plan] {int(table['hours'].notna().sum())}/{len(table)} con tiempo: {done_hours:.1f} h hechas, ~{remaining:.1f} h estimadas para el resto" + (f" ({rate:.3f} h por unidad de costo)" if rate is not None else ""))
        return table.round(3)

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
        return self.comet.download_final_weights(self.name(spec), self.weights_cache)

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
                    print(f"[plan] {asset_name} de {self.name(spec)} no coincide con el split actual; recalculo")
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
            if api_exp is not None:
                log_table_asset(api_exp, cached, asset_name)
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

    def _level_arrays(self, predictions: pd.DataFrame, task, level: str):
        """``(y_true, y_prob, groups)`` de la tarea ``task`` por imagen o por mama (``level="breast"``)."""
        mask, y, groups = self._task_arrays(predictions, task)
        prob = predictions["y_prob"].to_numpy()[mask]
        if level == "image":
            return y, prob, groups
        if level != "breast":
            raise ValueError(f"level debe ser 'image' o 'breast', no {level!r}")
        breast = BreastAggregation.aggregate(predictions.loc[mask].reset_index(drop=True), y, prob)
        if breast is None:
            raise ValueError("Las predicciones no tienen study_id/patient_id + laterality para agregar por mama")
        return breast

    def evaluate(self, specs, *, split: str = "val", tasks=None, level: str = "image", ci: bool = False) -> pd.DataFrame:
        """Metricas clinicas de cada corrida sobre las mismas imagenes, por tarea.

        ``level="breast"`` agrega por mama (maximo entre vistas). Las metricas de
        umbral usan el Youden de la misma corrida en val, al mismo nivel. ``ci``
        agrega el IC 95 % de ROC-AUC y PR-AUC (bootstrap agrupado por paciente).
        """
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
                y, prob, groups = self._level_arrays(predictions, task, level)
                y_val, prob_val, _ = self._level_arrays(self.predictions(spec, "val"), task, level)
                metrics = ClinicalMetrics.compute(y, prob, threshold=ThresholdSelector.youden_j(y_val, prob_val))
                if ci:
                    metrics.update(ClinicalMetrics.bootstrap_ci(y, prob, groups=groups, samples=self.bootstrap_samples, seed=int(self.config["GENERAL"]["RANDOM_SEED"])))
                rows.append(
                    {
                        "experiment": self.name(spec),
                        "backbone": spec.backbone,
                        "arch": spec.arch,
                        "label": spec.label_key,
                        "seed": spec.seed if spec.seed is not None else self.config["GENERAL"]["RANDOM_SEED"],
                        "split": split,
                        "level": level,
                        "task": _task_scheme(task)["title"],
                        **metrics,
                    }
                )
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ comparar

    def pairwise(self, specs, *, split: str = "val", task=None, level: str = "image") -> pd.DataFrame:
        """Todas contra todas sobre las mismas imagenes, sin elegir ganador.

        Una fila por par ``(a, b)``: metricas de cada uno y ``b - a`` con IC 95 %
        por bootstrap pareado agrupado por paciente. ``level="breast"`` compara
        por mama (maximo entre vistas).
        """
        task = self.task if task is None else task
        done = [spec for spec in specs if self.lookup(spec) is not None]
        skipped = [self.name(spec) for spec in specs if spec not in done]
        if skipped:
            print(f"[plan] sin resultado todavia: {skipped}")
        if not done:
            return pd.DataFrame()
        y, _, groups = self._level_arrays(self.predictions(done[0], split), task, level)
        probs = {self.name(spec): self._level_arrays(self.predictions(spec, split), task, level)[1] for spec in done}
        names = list(probs)
        rows = []
        for i, name_a in enumerate(names):
            for name_b in names[i + 1 :]:
                delta = PairedBootstrap.auc_delta(
                    y, probs[name_a], probs[name_b], groups=groups, samples=self.bootstrap_samples, seed=int(self.config["GENERAL"]["RANDOM_SEED"])
                )
                rows.append(
                    {
                        "level": level,
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

    # ------------------------------------------------------------------ reportes

    def seed_summary(self, groups, *, split: str = "test", task=None, level: str = "image") -> pd.DataFrame:
        """Media ± desvio entre semillas de las metricas clinicas.

        ``groups``: ``{"config": [RunSpec, ...]}`` o una lista de ``RunSpec`` (se
        agrupan por configuracion, ignorando la semilla).
        """
        if not isinstance(groups, dict):
            grouped: dict[str, list] = {}
            for spec in groups:
                grouped.setdefault(spec.name(self.config, include_seed=False), []).append(spec)
            groups = grouped
        rows = []
        for label, specs in groups.items():
            done = [spec for spec in specs if self.lookup(spec) is not None]
            metrics = self.evaluate(done, split=split, tasks=[self.task if task is None else task], level=level) if done else pd.DataFrame()
            row = {"config": label, "n_seeds": len(done), "split": split, "level": level}
            for column in SEED_SUMMARY_METRICS:
                values = pd.to_numeric(metrics.get(column, pd.Series(dtype=float)), errors="coerce").dropna().to_numpy()
                row[f"{column}_mean"] = float(np.mean(values)) if len(values) else None
                row[f"{column}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
            rows.append(row)
        return pd.DataFrame(rows)
