"""Estado reanudable de una corrida: pesos por epoca, mejores checkpoints y callbacks.

Cada corrida vive en ``{RESUME.DIR}/{experiment_name}/``. Con ``RESUME.DIR`` en
Google Drive el avance sobrevive a que se caiga la VM de Colab; en disco local
sobrevive a reiniciar el kernel.

``state.json`` se escribe despues de copiar los pesos que referencia, siempre
con archivo temporal + ``os.replace``: si la VM muere a mitad de una copia, el
estado sigue apuntando a archivos completos.

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


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _to_jsonable(value):
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
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


class RunStore:
    """Directorio de una corrida: ``state.json`` + archivos de pesos."""

    def __init__(self, root: str | Path, experiment_name: str) -> None:
        self.root = Path(root)
        self.experiment_name = experiment_name
        self.dir = self.root / experiment_name

    def path(self, name: str) -> Path:
        return self.dir / name

    def load(self) -> dict | None:
        path = self.path(STATE_FILE)
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[resume] {path} ilegible ({exc}); la corrida arranca de cero")
            return None

    def save(self, state: dict) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        state = {**state, "updated_at": _now()}
        target = self.path(STATE_FILE)
        tmp = target.with_name(target.name + ".tmp")
        tmp.write_text(json.dumps(_to_jsonable(state), indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, target)

    def put_file(self, local_path: str | Path, name: str | None = None) -> str:
        local_path = Path(local_path)
        name = name or local_path.name
        target = self.path(name)
        if target.resolve() == local_path.resolve():
            return name
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + ".tmp")
        shutil.copyfile(local_path, tmp)
        os.replace(tmp, target)
        return name

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

    def has_file(self, name: str | None) -> bool:
        return bool(name) and self.path(name).is_file()

    def remove_files(self, pattern: str, keep: set[str]) -> None:
        if not self.dir.is_dir():
            return
        for path in self.dir.glob(pattern):
            if path.name not in keep and path.name != STATE_FILE:
                path.unlink(missing_ok=True)


class TrainingResume:
    """Conecta un ``RunStore`` con el ``ModelTrainer`` y las etapas de entrenamiento."""

    def __init__(self, store: RunStore) -> None:
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
        self.model = None
        self.resumed = self.state.get("status") in ("training", "trained")
        # Nombres ya subidos en esta sesion: un archivo viejo con el mismo nombre
        # (epoca repetida tras un corte) se pisa en vez de reutilizarse.
        self._synced: set[str] = set()
        self._weights_restored = False

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
        """Copia los mejores checkpoints guardados a ``checkpoints/`` y los registra en el trainer."""
        self.model = model
        trainer = model.trainer
        restored = []
        for info in self.state.get("best_checkpoints", []):
            local = self.store.fetch_file(info["file"], trainer.checkpoint_dir)
            restored.append({**info, "path": local})
            self._synced.add(info["file"])
        trainer.best_checkpoints = [
            {key: value for key, value in info.items() if key != "file"} for info in restored
        ]
        if self.resumed:
            stages = {key: value for key, value in self.state.get("stages", {}).items()}
            print(f"[resume] {self.store.experiment_name}: retomo (estado={self.status}, etapas={stages})")

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
        self.store.save(self.state)

    def restore_timer(self, training_timer) -> None:
        for stage, summary in (self.state.get("stage_summaries") or {}).items():
            training_timer.stage_summaries[int(stage)] = {key: float(value) for key, value in summary.items()}

    def _load_weights(self, name: str) -> None:
        self.model.model.load_weights(str(self.store.path(name)))

    def _best_file(self, stage: int) -> str | None:
        info = next((item for item in self.state.get("best_checkpoints", []) if int(item["stage"]) == stage), None)
        return info["file"] if info else None

    def prepare_stage(self, stage: int) -> int:
        """Carga los pesos del punto de reanudacion y devuelve el ``initial_epoch`` de la etapa.

        Solo actua en la primera etapa que se corre en esta sesion de una corrida retomada.
        """
        if not self.resumed or self._weights_restored:
            return 0
        self._weights_restored = True
        point = self.state.get("weights_point") or {}
        last = self.state.get("last_weights")
        if not point or not self.store.has_file(last):
            return 0
        point_stage = int(point.get("stage", 0))
        if point_stage == stage and not point.get("stage_end"):
            self._load_weights(last)
            initial_epoch = self.stage_epochs_done(stage)
            print(f"[resume] etapa {stage}: retomo despues de la epoca {initial_epoch} ({last})")
            return initial_epoch
        if point_stage < stage:
            if point.get("stage_end"):
                self._load_weights(last)
            else:
                best = self._best_file(point_stage)
                self._load_weights(best if self.store.has_file(best) else last)
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

    def _sync_best_checkpoints(self) -> None:
        entries = []
        for info in self.model.trainer.best_checkpoints:
            path = Path(info["path"])
            name = path.name
            if name not in self._synced:
                self.store.put_file(path, name)
                self._synced.add(name)
            entry = {key: value for key, value in info.items() if key != "path"}
            entry["file"] = name
            entries.append(entry)
        self.state["best_checkpoints"] = entries

    def _cleanup_best_files(self) -> None:
        keep = {entry["file"] for entry in self.state.get("best_checkpoints", [])}
        self.store.remove_files("*_stage_*_epoch*.weights.h5", keep)

    def _save_last_weights(self, stage: int, epoch: int, *, stage_end: bool) -> None:
        trainer = self.model.trainer
        tag = "end" if stage_end else f"e{epoch:03d}"
        name = f"{LAST_WEIGHTS_PREFIX}s{stage}_{tag}.weights.h5"
        local = Path(trainer.checkpoint_dir) / name
        local.parent.mkdir(parents=True, exist_ok=True)
        self.model.model.save_weights(str(local))
        self.store.put_file(local, name)
        local.unlink(missing_ok=True)
        self.state["last_weights"] = name
        self.state["weights_point"] = {"stage": stage, "epoch": epoch, "stage_end": stage_end}

    def record_epoch(self, stage: int, epoch: int, *, stop_training: bool, elapsed_seconds: float) -> None:
        trainer = self.model.trainer
        self._sync_best_checkpoints()
        self._save_last_weights(stage, epoch, stage_end=False)
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
        self.store.save(self.state)
        self._cleanup_last_files()
        self._cleanup_best_files()

    def finish_stage(self, stage: int, epochs_done: int, *, summary: dict | None, elapsed_seconds: float) -> None:
        self._sync_best_checkpoints()
        self._save_last_weights(stage, epochs_done, stage_end=True)
        stage_state = self._stage(stage)
        stage_state["epochs_done"] = int(epochs_done)
        stage_state["finished"] = True
        if summary:
            self.state.setdefault("stage_summaries", {})[str(stage)] = summary
        self.state["elapsed_seconds"] = float(elapsed_seconds)
        self.store.save(self.state)
        self._cleanup_last_files()
        self._cleanup_best_files()

    def mark_trained(self) -> None:
        self.state["status"] = "trained"
        self.store.save(self.state)

    def mark_done(self, summary: dict, *, best_file: str | None) -> None:
        """Cierra la corrida: deja solo el mejor checkpoint global (para transferir o reevaluar)."""
        self.state["status"] = "done"
        self.state["finished_at"] = _now()
        self.state["summary"] = summary
        if best_file:
            self.state["best_checkpoints"] = [
                entry for entry in self.state.get("best_checkpoints", []) if entry["file"] == best_file
            ]
        self.state["last_weights"] = None
        self.state["weights_point"] = None
        self.store.save(self.state)
        self._cleanup_last_files()
        self._cleanup_best_files()

    def _cleanup_last_files(self) -> None:
        keep = {self.state["last_weights"]} if self.state.get("last_weights") else set()
        self.store.remove_files(f"{LAST_WEIGHTS_PREFIX}*.weights.h5", keep)

    def callback(self, stage: int, training_timer=None) -> keras.callbacks.Callback:
        return ResumeCallback(self, stage, training_timer=training_timer)

    def best_global_file(self, monitor_mode: str) -> str | None:
        entries = [entry for entry in self.state.get("best_checkpoints", []) if _optional_float(entry.get("value")) is not None]
        if not entries:
            return None
        pick = min if monitor_mode == "min" else max
        return pick(entries, key=lambda entry: float(entry["value"]))["file"]


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
]
