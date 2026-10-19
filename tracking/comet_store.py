"""Store de resume sobre assets de Comet (``resume_*``) del experimento de la corrida.

Los uploads van por la REST API (``APIExperiment``) en un thread: el
entrenamiento no espera a la red. Si llega un commit nuevo mientras se sube
el anterior, el pendiente se reemplaza por el mas nuevo (solo importa el
ultimo estado). Cada commit sube los pesos que faltan, despues
``resume_state_{N}.json`` con N creciente, y recien entonces borra estados
viejos y pesos no referenciados.
"""

from __future__ import annotations

import json
import re
import tempfile
import threading
import traceback
from pathlib import Path

from src.training.resume import referenced_files, to_jsonable

ASSET_PREFIX = "resume_"
STATE_RE = re.compile(r"resume_state_(\d+)\.json")


class CometRunStore:
    keeps_final_weights = False

    def __init__(self, index, experiment_name: str) -> None:
        self.index = index
        self.experiment_name = experiment_name
        self.uploaded: set[str] = set()
        self._api_exp = None
        self._version = 0
        self._state: dict | None = None
        self._loaded = False
        self._lock = threading.Lock()
        self._idle = threading.Event()
        self._idle.set()
        self._pending = None
        self._worker: threading.Thread | None = None

    @property
    def location(self) -> str:
        key = self._api_exp.id if self._api_exp is not None else "?"
        return f"comet:{self.experiment_name}/{key}"

    def _assets(self) -> list[dict]:
        return list(self._api_exp.get_asset_list() or []) if self._api_exp is not None else []

    def load(self) -> dict | None:
        return self._state if self._loaded else self.reload(fresh=False)

    def reload(self, *, fresh: bool = True) -> dict | None:
        """Estado mas reciente entre los experimentos con ese nombre (el mas nuevo primero).

        ``fresh`` vuelve a listar el proyecto; ``load`` usa la lista cacheada del indice.
        """
        self.wait()
        self._loaded = True
        self._state = None
        self._api_exp = None
        self._version = 0
        self.uploaded = set()
        for api_exp in self.index.experiments_named(self.experiment_name, fresh=fresh):
            assets = list(api_exp.get_asset_list() or [])
            versions = [(int(match.group(1)), asset) for asset in assets if (match := STATE_RE.fullmatch(str(asset.get("fileName", ""))))]
            if not versions:
                continue
            version, asset = max(versions, key=lambda item: item[0])
            try:
                state = json.loads(api_exp.get_asset(asset["assetId"], return_type="binary"))
            except Exception as exc:
                print(f"[resume] estado ilegible en {api_exp.id} ({exc}); pruebo el anterior")
                continue
            names = {str(item.get("fileName", "")) for item in assets}
            self._api_exp = api_exp
            self._version = version
            self._state = state
            self.uploaded = {name for name in referenced_files(state) if ASSET_PREFIX + name in names}
            break
        return self._state

    def bind(self, comet_key: str | None) -> None:
        """Las escrituras van al experimento ``comet_key`` (el que esta logueando la corrida)."""
        if not comet_key:
            return
        self.wait()
        if self._api_exp is not None and self._api_exp.id == comet_key:
            return
        self._api_exp = self.index.api.get_experiment_by_key(comet_key)
        self._version = 0
        self.uploaded = set()

    def fetch_file(self, name: str, local_dir: str | Path) -> Path:
        candidates = [asset for asset in self._assets() if asset.get("fileName") == ASSET_PREFIX + name]
        if not candidates:
            raise FileNotFoundError(f"Falta {name} en {self.location}")
        asset = max(candidates, key=lambda item: int(item.get("createdAt") or 0))
        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)
        target = local_dir / name
        tmp = target.with_name(target.name + ".tmp")
        tmp.write_bytes(self._api_exp.get_asset(asset["assetId"], return_type="binary"))
        tmp.replace(target)
        return target

    def commit(self, files: dict, state: dict, *, temp=(), wait: bool = False) -> None:
        if self._api_exp is None:
            raise RuntimeError(f"{self.experiment_name}: bind() antes de guardar estado en Comet")
        job = {"files": dict(files), "state": json.dumps(to_jsonable(state), ensure_ascii=False), "temp": set(temp)}
        with self._lock:
            previous = self._pending
            if previous is not None:
                for name in previous["temp"]:
                    if name in job["files"]:
                        job["temp"].add(name)
                    else:
                        _unlink(previous["files"].get(name))
            self._pending = job
            self._idle.clear()
            if self._worker is None or not self._worker.is_alive():
                self._worker = threading.Thread(target=self._run, name=f"resume-{self.experiment_name}", daemon=True)
                self._worker.start()
        self._state, self._loaded = json.loads(job["state"]), True
        if wait:
            self.wait()

    def wait(self) -> None:
        self._idle.wait()

    def _run(self) -> None:
        while True:
            with self._lock:
                job = self._pending
                self._pending = None
                if job is None:
                    self._worker = None
                    self._idle.set()
                    return
            try:
                self._apply(job)
            except Exception as exc:
                print(f"[resume] no pude guardar el estado de {self.experiment_name} en Comet: {exc}")
                traceback.print_exc()
            finally:
                for name in job["temp"]:
                    _unlink(job["files"].get(name))

    def _apply(self, job: dict) -> None:
        api_exp = self._api_exp
        files = job["files"]
        for name, local in files.items():
            if local is None or name in self.uploaded:
                continue
            api_exp.log_asset(str(local), name=ASSET_PREFIX + name)
            self.uploaded.add(name)
        version = self._version + 1
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as handle:
            handle.write(job["state"])
            state_path = Path(handle.name)
        try:
            api_exp.log_asset(str(state_path), name=f"resume_state_{version:06d}.json")
        finally:
            state_path.unlink(missing_ok=True)
        self._version = version
        keep = {ASSET_PREFIX + name for name in files}
        newest: dict[str, dict] = {}
        stale = []
        for asset in self._assets():
            file_name = str(asset.get("fileName", ""))
            match = STATE_RE.fullmatch(file_name)
            if match:
                if int(match.group(1)) < version:
                    stale.append(asset)
                continue
            if not file_name.startswith(ASSET_PREFIX) or not file_name.endswith(".weights.h5"):
                continue
            if file_name not in keep:
                stale.append(asset)
                continue
            current = newest.get(file_name)
            if current is None or int(asset.get("createdAt") or 0) > int(current.get("createdAt") or 0):
                if current is not None:
                    stale.append(current)
                newest[file_name] = asset
            else:
                stale.append(asset)
        for asset in stale:
            api_exp.delete_asset(asset["assetId"])
        self.uploaded &= set(files)


def _unlink(path) -> None:
    if path is not None:
        Path(path).unlink(missing_ok=True)
