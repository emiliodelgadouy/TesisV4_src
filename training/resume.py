"""Estado reanudable de una corrida: pesos por epoca, mejores checkpoints y callbacks.

El estado vive en un *store*:

- ``CometRunStore`` (``src/tracking/comet_store.py``, default): assets del mismo
  experimento de Comet. Sobrevive a perder la VM sin montar nada.
- ``RunStore``: una carpeta ``{RESUME.DIR}/{experiment_name}/`` (disco o Drive).

Los dos aplican el mismo contrato en ``commit``: primero suben los pesos que el
estado referencia, despues el estado, y recien entonces borran los pesos que
ya no se referencian. Un corte a mitad de camino deja el estado anterior, que
apunta a archivos completos.

Al reanudar no se recuperan los momentos de Adam (el optimizador se recompila);
si se recuperan los pesos, el learning rate y los contadores de EarlyStopping y
ReduceLROnPlateau.
"""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
from tensorflow import keras

STATE_FILE = "state.json"
LAST_WEIGHTS_PREFIX = "last_"
SNAPSHOT_DIR = Path("checkpoints") / "resume_snapshots"


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def to_jsonable(value):
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _optional_float(value) -> float | None:
    if value is None:
        return None
    try:
        out = float(np.asarray(value))
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def get_learning_rate(optimizer) -> float:
    value = optimizer.learning_rate
    try:
        return float(value.numpy())
    except AttributeError:
        return float(np.asarray(value))


def set_learning_rate(optimizer, value: float) -> None:
    try:
        optimizer.learning_rate.assign(float(value))
    except AttributeError:
        optimizer.learning_rate = float(value)


def referenced_files(state: dict) -> set[str]:
    names = {entry["file"] for entry in state.get("best_checkpoints", []) if entry.get("file")}
    if state.get("last_weights"):
        names.add(state["last_weights"])
    return names


class RunStore:
    """Store en carpeta: ``state.json`` + archivos de pesos."""

    keeps_final_weights = True

    def __init__(self, root: str | Path, experiment_name: str) -> None:
        self.root = Path(root)
        self.experiment_name = experiment_name
        self.dir = self.root / experiment_name
        self.uploaded: set[str] = set()
        self._state: dict | None = None
        self._loaded = False

    @property
    def location(self) -> str:
        return str(self.dir)

    def path(self, name: str) -> Path:
        return self.dir / name

    def has_file(self, name: str | None) -> bool:
        return bool(name) and self.path(name).is_file()

    def load(self) -> dict | None:
        return self._state if self._loaded else self.reload()

    def reload(self) -> dict | None:
        self._loaded = True
        self._state = None
        path = self.path(STATE_FILE)
        if path.is_file():
            try:
                self._state = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                print(f"[resume] {path} ilegible ({exc}); la corrida arranca de cero")
        # Solo cuenta como subido lo que el estado valido referencia: un archivo
        # viejo con el mismo nombre (epoca repetida tras un corte) se pisa.
        self.uploaded = {name for name in referenced_files(self._state or {}) if self.has_file(name)}
        return self._state

    def bind(self, comet_key: str | None) -> None:
        del comet_key

    def _put(self, local_path: Path, name: str) -> None:
        target = self.path(name)
        if target.resolve() == Path(local_path).resolve():
            return
        tmp = target.with_name(target.name + ".tmp")
        shutil.copyfile(local_path, tmp)
        os.replace(tmp, target)

    def commit(self, files: dict, state: dict, *, temp=(), wait: bool = False) -> None:
        """``files``: nombre → path local (``None`` si ya esta en el store). Sincronico."""
        del wait
        self.dir.mkdir(parents=True, exist_ok=True)
        try:
            for name, local in files.items():
                if local is None or name in self.uploaded:
                    continue
                self._put(Path(local), name)
                self.uploaded.add(name)
            payload = {**to_jsonable(state), "updated_at": _now()}
            target = self.path(STATE_FILE)
            tmp = target.with_name(target.name + ".tmp")
            tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, target)
            self._state, self._loaded = payload, True
            keep = set(files)
            for path in self.dir.glob("*.weights.h5"):
                if path.name not in keep:
                    path.unlink(missing_ok=True)
            self.uploaded &= keep
        finally:
            for name in temp:
                local = files.get(name)
                if local is not None:
                    Path(local).unlink(missing_ok=True)

    def wait(self) -> None:
        return None

    def fetch_file(self, name: str, local_dir: str | Path) -> Path:
        source = self.path(name)
        if not source.is_file():
            raise FileNotFoundError(f"Falta {source} en el store de {self.experiment_name}")
        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)
        target = local_dir / name
        if target.resolve() != source.resolve():
            shutil.copyfile(source, target)
        return target


class TrainingResume:
    """Conecta un store con el ``ModelTrainer`` y las etapas de entrenamiento."""

    def __init__(self, store) -> None:
        self.store = store
        self.state = store.load() or {
            "experiment": store.experiment_name,
            "status": "new",
            "resume_count": 0,
            "stages": {},
            "best_checkpoints": [],
            "stage_summaries": {},
            "elapsed_seconds": 0.0,
        }
        self.state = json.loads(json.dumps(self.state))
        self.model = None
        self.resumed = self.state.get("status") in ("training", "trained")
        self._weights_restored = False
        self._last_local: Path | None = None

    @property
    def status(self) -> str:
        return str(self.state.get("status", "new"))

    @property
    def is_done(self) -> bool:
        return self.status == "done"

    @property
    def comet_key(self) -> str | None:
        return self.state.get("comet_key")

    @property
    def summary(self) -> dict | None:
        return self.state.get("summary")

    @property
    def elapsed_seconds(self) -> float:
        return float(self.state.get("elapsed_seconds") or 0.0)

    @property
    def resume_count(self) -> int:
        return int(self.state.get("resume_count") or 0)

    def _stage(self, stage: int) -> dict:
        return self.state.setdefault("stages", {}).setdefault(str(stage), {"epochs_done": 0, "finished": False})

    def stage_finished(self, stage: int) -> bool:
        return bool(self.state.get("stages", {}).get(str(stage), {}).get("finished"))

    def stage_epochs_done(self, stage: int) -> int:
        return int(self.state.get("stages", {}).get(str(stage), {}).get("epochs_done", 0))

    def attach(self, model) -> None:
        """Baja los mejores checkpoints guardados a ``checkpoints/`` y los registra en el trainer."""
        self.model = model
        trainer = model.trainer
        restored = []
        for info in self.state.get("best_checkpoints", []):
            local = self.store.fetch_file(info["file"], trainer.checkpoint_dir)
            entry = {key: value for key, value in info.items() if key != "file"}
            restored.append({**entry, "path": local})
        trainer.best_checkpoints = restored
        if self.resumed:
            stages = {key: {"epochs_done": value.get("epochs_done"), "finished": value.get("finished")} for key, value in self.state.get("stages", {}).items()}
            print(f"[resume] {self.store.experiment_name}: retomo de {self.store.location} (estado={self.status}, etapas={stages})")

    def _files(self) -> dict:
        """Todo lo que el estado referencia, con su path local si hay que subirlo."""
        files: dict = {}
        for info in self.model.trainer.best_checkpoints if self.model is not None else []:
            files[Path(info["path"]).name] = Path(info["path"])
        last = self.state.get("last_weights")
        if last:
            files[last] = self._last_local if self._last_local is not None and self._last_local.name == last else None
        for entry in self.state.get("best_checkpoints", []):
            files.setdefault(entry["file"], None)
        return files

    def _commit(self, *, temp=(), wait: bool = False) -> None:
        self.store.commit(self._files(), self.state, temp=temp, wait=wait)

    def begin(self, comet_key: str | None) -> None:
        if self.resumed:
            self.state["resume_count"] = self.resume_count + 1
            self.state["resumed_at"] = _now()
        else:
            self.state["started_at"] = _now()
        if self.status == "new":
            self.state["status"] = "training"
        if comet_key:
            self.state["comet_key"] = comet_key
        self.store.bind(comet_key)
        self._commit()

    def restore_timer(self, training_timer) -> None:
        for stage, summary in (self.state.get("stage_summaries") or {}).items():
            training_timer.stage_summaries[int(stage)] = {key: float(value) for key, value in summary.items()}

    def _load_weights(self, name: str) -> None:
        local = self.store.fetch_file(name, SNAPSHOT_DIR)
        try:
            self.model.model.load_weights(str(local))
        finally:
            if local.parent.resolve() == SNAPSHOT_DIR.resolve():
                local.unlink(missing_ok=True)

    def _best_file(self, stage: int) -> str | None:
        info = next((item for item in self.state.get("best_checkpoints", []) if int(item["stage"]) == stage), None)
        return info["file"] if info else None

    def _load_first(self, *names) -> str | None:
        for name in names:
            if not name:
                continue
            try:
                self._load_weights(name)
                return name
            except FileNotFoundError as exc:
                print(f"[resume] {exc}")
        return None

    def prepare_stage(self, stage: int) -> int:
        """Carga los pesos del punto de reanudacion y devuelve el ``initial_epoch`` de la etapa.

        Solo actua en la primera etapa que se corre en esta sesion de una corrida retomada.
        """
        if not self.resumed or self._weights_restored:
            return 0
        self._weights_restored = True
        point = self.state.get("weights_point") or {}
        last = self.state.get("last_weights")
        if not point:
            return 0
        point_stage = int(point.get("stage", 0))
        if point_stage == stage and not point.get("stage_end"):
            loaded = self._load_first(last, self._best_file(stage))
            if loaded is None:
                print(f"[resume] etapa {stage}: no hay pesos guardados; la etapa arranca de cero")
                return 0
            initial_epoch = self.stage_epochs_done(stage)
            print(f"[resume] etapa {stage}: retomo despues de la epoca {initial_epoch} ({loaded})")
            return initial_epoch
        if point_stage < stage:
            names = (last, self._best_file(point_stage)) if point.get("stage_end") else (self._best_file(point_stage), last)
            if self._load_first(*names):
                print(f"[resume] etapa {stage}: arranco con los pesos de la etapa {point_stage}")
        return 0

    def restore_callbacks(self, stage: int, trainer) -> None:
        saved = self.state.get("stages", {}).get(str(stage), {})
        if not saved.get("epochs_done"):
            return
        early_stopping = getattr(trainer, "early_stopping", None)
        reduce_lr = getattr(trainer, "reduce_lr", None)
        es_state = saved.get("early_stopping") or {}
        if early_stopping is not None and es_state:
            early_stopping.wait = int(es_state.get("wait", 0))
            early_stopping.best_epoch = int(es_state.get("best_epoch", 0))
            best = _optional_float(es_state.get("best"))
            if best is not None:
                early_stopping.best = best
        rlr_state = saved.get("reduce_lr") or {}
        if reduce_lr is not None and rlr_state:
            reduce_lr.wait = int(rlr_state.get("wait", 0))
            reduce_lr.cooldown_counter = int(rlr_state.get("cooldown_counter", 0))
            best = _optional_float(rlr_state.get("best"))
            if best is not None:
                reduce_lr.best = best
        learning_rate = saved.get("learning_rate")
        if learning_rate is not None:
            set_learning_rate(self.model.model.optimizer, learning_rate)

    def _record_best_checkpoints(self) -> None:
        entries = []
        for info in self.model.trainer.best_checkpoints:
            entry = {key: value for key, value in info.items() if key != "path"}
            entry["file"] = Path(info["path"]).name
            entries.append(entry)
        self.state["best_checkpoints"] = entries

    def _snapshot_last_weights(self, stage: int, epoch: int, *, stage_end: bool) -> str:
        tag = "end" if stage_end else f"e{epoch:03d}"
        # Prefijo por corrida: el upload asincronico de una corrida puede seguir
        # cuando ya arranco la siguiente.
        name = f"{self.store.experiment_name}_{LAST_WEIGHTS_PREFIX}s{stage}_{tag}.weights.h5"
        SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        local = SNAPSHOT_DIR / name
        self.model.model.save_weights(str(local))
        self._last_local = local
        self.state["last_weights"] = name
        self.state["weights_point"] = {"stage": stage, "epoch": epoch, "stage_end": stage_end}
        return local.name

    def record_epoch(self, stage: int, epoch: int, *, stop_training: bool, elapsed_seconds: float) -> None:
        trainer = self.model.trainer
        self._record_best_checkpoints()
        last = self._snapshot_last_weights(stage, epoch, stage_end=False)
        stage_state = self._stage(stage)
        stage_state["epochs_done"] = int(epoch)
        stage_state["finished"] = bool(stop_training)
        early_stopping = getattr(trainer, "early_stopping", None)
        reduce_lr = getattr(trainer, "reduce_lr", None)
        if early_stopping is not None:
            stage_state["early_stopping"] = {
                "wait": int(early_stopping.wait),
                "best": _optional_float(early_stopping.best),
                "best_epoch": int(getattr(early_stopping, "best_epoch", 0) or 0),
            }
        if reduce_lr is not None:
            stage_state["reduce_lr"] = {
                "wait": int(reduce_lr.wait),
                "best": _optional_float(reduce_lr.best),
                "cooldown_counter": int(reduce_lr.cooldown_counter),
            }
        stage_state["learning_rate"] = get_learning_rate(self.model.model.optimizer)
        self.state["elapsed_seconds"] = float(elapsed_seconds)
        self._commit(temp={last})

    def finish_stage(self, stage: int, epochs_done: int, *, summary: dict | None, elapsed_seconds: float) -> None:
        self._record_best_checkpoints()
        last = self._snapshot_last_weights(stage, epochs_done, stage_end=True)
        stage_state = self._stage(stage)
        stage_state["epochs_done"] = int(epochs_done)
        stage_state["finished"] = True
        if summary:
            self.state.setdefault("stage_summaries", {})[str(stage)] = summary
        self.state["elapsed_seconds"] = float(elapsed_seconds)
        self._commit(temp={last})

    def mark_trained(self) -> None:
        self.state["status"] = "trained"
        self._commit()

    def mark_done(self, summary: dict, *, best_file: str | None) -> None:
        """Cierra la corrida. El store en carpeta conserva el mejor checkpoint global
        (para transferir o reevaluar); Comet ya lo tiene como ``log_model``."""
        self.state["status"] = "done"
        self.state["finished_at"] = _now()
        self.state["summary"] = summary
        keep = best_file if self.store.keeps_final_weights else None
        self.state["best_checkpoints"] = [entry for entry in self.state.get("best_checkpoints", []) if entry["file"] == keep]
        self.state["last_weights"] = None
        self.state["weights_point"] = None
        files = {}
        if keep is not None and self.model is not None:
            info = next((info for info in self.model.trainer.best_checkpoints if Path(info["path"]).name == keep), None)
            files[keep] = Path(info["path"]) if info else None
        self.store.commit(files, self.state, wait=True)

    def callback(self, stage: int, training_timer=None) -> keras.callbacks.Callback:
        return ResumeCallback(self, stage, training_timer=training_timer)


class ResumeCallback(keras.callbacks.Callback):
    """Va al final de la lista: corre despues del checkpoint, EarlyStopping y ReduceLROnPlateau."""

    def __init__(self, resume: TrainingResume, stage: int, *, training_timer=None) -> None:
        super().__init__()
        self.resume = resume
        self.stage = int(stage)
        self.training_timer = training_timer

    def on_train_begin(self, logs=None):
        self.resume.restore_callbacks(self.stage, self.resume.model.trainer)

    def on_epoch_end(self, epoch, logs=None):
        elapsed = self.training_timer.elapsed_since_training_start() if self.training_timer is not None else 0.0
        self.resume.record_epoch(
            self.stage,
            int(epoch) + 1,
            stop_training=bool(getattr(self.model, "stop_training", False)),
            elapsed_seconds=elapsed,
        )


def history_with_epochs(epochs_done: int) -> keras.callbacks.History:
    """History vacio que reporta ``epochs_done`` epocas (para etapas ya completadas)."""
    history = keras.callbacks.History()
    history.history = {}
    history.epoch = list(range(int(epochs_done)))
    return history


def epochs_completed(history) -> int:
    epochs = list(getattr(history, "epoch", None) or [])
    return int(epochs[-1]) + 1 if epochs else 0


__all__ = [
    "ResumeCallback",
    "RunStore",
    "TrainingResume",
    "epochs_completed",
    "history_with_epochs",
    "referenced_files",
]
