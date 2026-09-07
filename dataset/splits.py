from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.dataset.config import DatasetConfig
from src.dataset.provider import hard_negatives_from_positives
from src.dataset.repository import DatasetRepository
from src.dataset.target import TargetMode


class SplitManager:
    """Carga o regenera train/val/test y remuestrea train para modos patch."""

    # Misma ruta local que descarga ``DatasetConfig.splits_local`` desde GCS.
    DEFAULT_SPLITS_PATH = DatasetConfig().splits_local

    def get(self, config, ds: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Devuelve train/val/test regenerando o cargando los ids fijos.

        Lo decide ``config["GENERAL"]["REGENERATE_SPLITS"]``:

        - ``True``: regenera el split estratificado, undersamplea los negativos del
          train y guarda los ids (train ya undersampleado) en ``DATASET_SPLITS_PATH``.
        - ``False``: carga los ids fijos desde ``DATASET_SPLITS_PATH`` (no regenera ni
          undersamplea).
        """
        if config["GENERAL"]["REGENERATE_SPLITS"]:
            train, val, test = self.stratified_train_val_test_split(config, ds)
            train = self.undersample_negatives(config, train)
            for name, frame in (("train", train), ("val", val), ("test", test)):
                self._assert_binary_split_ready(frame, split_name=name)
            self.save(config, train, val, test)
            return train, val, test
        train, val, test = self.load(config, ds)
        for name, frame in (("train", train), ("val", val), ("test", test)):
            self._assert_binary_split_ready(frame, split_name=name)
        return train, val, test

    @classmethod
    def resolve_path(cls, config) -> Path:
        """Ruta de splits: el dataset reducido usa un archivo aparte para no pisar el canonico."""
        path = cls.DEFAULT_SPLITS_PATH
        if config.get("GENERAL", {}).get("REDUCED_DATASET"):
            return path.with_name(f"{path.stem}_reduced{path.suffix}")
        return path

    @staticmethod
    def stratified_train_val_test_split(
        config,
        ds,
        *,
        split_column="split",
        train_split_value="training",
        test_split_value="test",
        label_column="cls",
        group_column="patient_id",
    ):
        val_split = float(config["GENERAL"]["VALIDATION_SPLIT_RATIO"])
        if not (0.0 < val_split < 1.0):
            raise ValueError(
                f"VALIDATION_SPLIT_RATIO debe estar en (0, 1); recibido {val_split!r}"
            )
        seed = config["GENERAL"]["RANDOM_SEED"]
        tbl_training_full = ds[ds[split_column] == train_split_value].reset_index(drop=True)
        tbl_test = ds[ds[split_column] == test_split_value].reset_index(drop=True)

        n_splits = max(2, round(1 / val_split))
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        train_idx, val_idx = next(
            sgkf.split(
                tbl_training_full,
                y=tbl_training_full[label_column],
                groups=tbl_training_full[group_column],
            )
        )

        tbl_train = tbl_training_full.iloc[train_idx].reset_index(drop=True)
        tbl_val = tbl_training_full.iloc[val_idx].reset_index(drop=True)
        return tbl_train, tbl_val, tbl_test

    @staticmethod
    def undersample_negatives(
        config,
        df: pd.DataFrame,
        *,
        label_column: str = "cls",
    ) -> pd.DataFrame:
        """Submuestrea negativos a `ratio`:1 respecto de los positivos. Solo para train."""
        ratio = float(config["GENERAL"]["NO_FINDING_TO_FINDING_RATIO"])
        seed = config["GENERAL"]["RANDOM_SEED"]
        pos = df[df[label_column] >= 0.5]
        neg = df[df[label_column] < 0.5]
        if len(pos) == 0:
            raise ValueError("undersample_negatives: no hay positivos en el DataFrame")
        n_neg = int(min(len(neg), int(round(len(pos) * ratio))))
        if n_neg <= 0:
            raise ValueError(
                f"undersample_negatives: n_neg={n_neg} invalido "
                f"(pos={len(pos)}, neg={len(neg)}, ratio={ratio})"
            )
        neg = neg.sample(n=n_neg, random_state=seed)
        out = pd.concat([pos, neg], ignore_index=True)
        return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    @staticmethod
    def logit_initial_bias(n_positive: int, n_negative: int, *, eps: float = 1e-6) -> float:
        """Log-odds inicial; evita division por cero / ±inf con conteos nulos."""
        n_pos = max(float(n_positive), eps)
        n_neg = max(float(n_negative), eps)
        return float(np.log(n_pos / n_neg))

    @classmethod
    def resample_train_for_patch(
        cls,
        train_df: pd.DataFrame,
        patch_ratio: dict[str, int | float],
        seed: int,
        *,
        label_column: str = "cls",
    ) -> tuple[pd.DataFrame, float, float]:
        """Remuestrea train en positivos + hard negatives + random negatives segun patch_ratio.

        ``patch_ratio`` usa claves POSITIVE, HARD_NEGATIVE y RANDOM_NEGATIVE (multiplicadores
        respecto del conteo de positivos). Devuelve (df, focal_alpha, initial_bias).
        """
        positive_train = train_df[train_df[label_column] == 1].copy()
        negative_pool = train_df[train_df[label_column] == 0].copy()
        n_positive = len(positive_train)

        def _resample(df: pd.DataFrame, n: int) -> pd.DataFrame:
            n = int(round(n))
            if n <= 0 or len(df) == 0:
                return df.iloc[0:0]
            if n <= len(df):
                return df.sample(n=n, random_state=seed)
            reps = pd.concat([df] * (n // len(df) + 1), ignore_index=True)
            return reps.sample(n=n, random_state=seed)

        positive_final = _resample(positive_train, n_positive * patch_ratio["POSITIVE"])
        hard_negative_final = _resample(
            hard_negatives_from_positives(positive_train),
            n_positive * patch_ratio["HARD_NEGATIVE"],
        )
        random_negative_final = _resample(negative_pool, n_positive * patch_ratio["RANDOM_NEGATIVE"])
        train_patch = (
            pd.concat([positive_final, hard_negative_final, random_negative_final], ignore_index=True)
            .sample(frac=1, random_state=seed)
            .reset_index(drop=True)
        )
        frac_neg = (train_patch[label_column] == 0).sum() / len(train_patch)
        bias = cls.logit_initial_bias(
            int((train_patch[label_column] == 1).sum()),
            int((train_patch[label_column] == 0).sum()),
        )
        print(
            "resample_train_for_patch:",
            {
                "POSITIVE": len(positive_final),
                "HARD_NEGATIVE": len(hard_negative_final),
                "RANDOM_NEGATIVE": len(random_negative_final),
                "TOTAL": len(train_patch),
                "FRAC_NEG": round(float(frac_neg), 3),
            },
        )
        return train_patch, float(frac_neg), bias

    def save(
        self,
        config,
        train: pd.DataFrame,
        val: pd.DataFrame,
        test: pd.DataFrame,
        *,
        patient_id_column: str = "patient_id",
        image_id_column: str = "image_id",
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        """Persiste los ids de train/val/test (orden incluido) en JSON o CSV."""
        general = config["GENERAL"]
        path = self.resolve_path(config)
        path.parent.mkdir(parents=True, exist_ok=True)
        suffix = path.suffix.lower()
        meta = {
            "n_train": int(len(train)),
            "n_val": int(len(val)),
            "n_test": int(len(test)),
            "random_seed": general["RANDOM_SEED"],
            "positive_mode": general.get("POSITIVE_MODE"),
            "target_mode": TargetMode.from_config(config).name,
            "validation_split_ratio": general["VALIDATION_SPLIT_RATIO"],
            "no_finding_to_finding_ratio": general["NO_FINDING_TO_FINDING_RATIO"],
            "reduced_dataset": general["REDUCED_DATASET"],
            "includes_undersampled_train": True,
            **(metadata or {}),
        }

        if suffix == ".json":
            payload = {
                "meta": meta,
                "train": self._ids_records(
                    train, patient_id_column=patient_id_column, image_id_column=image_id_column
                ),
                "val": self._ids_records(
                    val, patient_id_column=patient_id_column, image_id_column=image_id_column
                ),
                "test": self._ids_records(
                    test, patient_id_column=patient_id_column, image_id_column=image_id_column
                ),
            }
            path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        elif suffix == ".csv":
            frames = []
            for split_name, df in (("train", train), ("val", val), ("test", test)):
                ids = self._normalize_split_id_columns(
                    df, patient_id_column=patient_id_column, image_id_column=image_id_column
                )
                ids = ids.rename(
                    columns={patient_id_column: "patient_id", image_id_column: "image_id"}
                )
                ids.insert(0, "split", split_name)
                ids.insert(1, "order", np.arange(len(ids), dtype=np.int64))
                frames.append(ids)
            pd.concat(frames, ignore_index=True).to_csv(path, index=False)
            meta_path = path.with_suffix(path.suffix + ".meta.json")
            meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        else:
            raise ValueError(f"Formato de splits no soportado: {suffix!r} (usa .json o .csv)")

        print(
            f"Splits guardados en {path} "
            f"(train={meta['n_train']}, val={meta['n_val']}, test={meta['n_test']})"
        )
        return path

    def load(
        self,
        config,
        ds: pd.DataFrame,
        *,
        patient_id_column: str = "patient_id",
        image_id_column: str = "image_id",
        require_existing_files: bool = True,
        path_column: str = "path",
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Carga train/val/test desde un JSON/CSV de ids, en el mismo orden guardado."""
        path = self.resolve_path(config)
        if not path.is_file():
            raise FileNotFoundError(f"No existe el archivo de splits: {path}")

        suffix = path.suffix.lower()
        if suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            train = self._frame_from_split_ids(
                ds,
                payload["train"],
                split_name="train",
                patient_id_column=patient_id_column,
                image_id_column=image_id_column,
            )
            val = self._frame_from_split_ids(
                ds,
                payload["val"],
                split_name="val",
                patient_id_column=patient_id_column,
                image_id_column=image_id_column,
            )
            test = self._frame_from_split_ids(
                ds,
                payload["test"],
                split_name="test",
                patient_id_column=patient_id_column,
                image_id_column=image_id_column,
            )
            meta = payload.get("meta") or {}
        elif suffix == ".csv":
            raw = pd.read_csv(path, dtype={"patient_id": str, "image_id": str, "split": str})
            if "order" in raw.columns:
                raw = raw.sort_values(["split", "order"], kind="mergesort")
            train = self._frame_from_split_ids(
                ds,
                raw.loc[raw["split"] == "train", ["patient_id", "image_id"]],
                split_name="train",
                patient_id_column=patient_id_column,
                image_id_column=image_id_column,
            )
            val = self._frame_from_split_ids(
                ds,
                raw.loc[raw["split"] == "val", ["patient_id", "image_id"]],
                split_name="val",
                patient_id_column=patient_id_column,
                image_id_column=image_id_column,
            )
            test = self._frame_from_split_ids(
                ds,
                raw.loc[raw["split"] == "test", ["patient_id", "image_id"]],
                split_name="test",
                patient_id_column=patient_id_column,
                image_id_column=image_id_column,
            )
            meta = {"n_train": len(train), "n_val": len(val), "n_test": len(test)}
        else:
            raise ValueError(f"Formato de splits no soportado: {suffix!r} (usa .json o .csv)")

        if require_existing_files:
            train = DatasetRepository.filter_existing_files(train, path_column)
            val = DatasetRepository.filter_existing_files(val, path_column)
            test = DatasetRepository.filter_existing_files(test, path_column)

        print(
            f"Splits cargados desde {path} "
            f"(train={len(train)}, val={len(val)}, test={len(test)}"
            + (f", meta_keys={sorted(meta.keys())}" if meta else "")
            + ")"
        )
        return train, val, test

    @staticmethod
    def _assert_binary_split_ready(
        df: pd.DataFrame,
        *,
        split_name: str,
        label_column: str = "cls",
    ) -> None:
        if len(df) == 0:
            raise ValueError(f"Split {split_name!r} quedo vacio")
        n_pos = int((df[label_column] >= 0.5).sum())
        n_neg = int(len(df) - n_pos)
        if n_pos == 0 or n_neg == 0:
            raise ValueError(
                f"Split {split_name!r} necesita ambas clases; "
                f"recibido pos={n_pos}, neg={n_neg}, total={len(df)}"
            )

    @staticmethod
    def _normalize_split_id_columns(
        df: pd.DataFrame,
        *,
        patient_id_column: str,
        image_id_column: str,
    ) -> pd.DataFrame:
        missing = [c for c in (patient_id_column, image_id_column) if c not in df.columns]
        if missing:
            raise KeyError(f"Faltan columnas de id para splits: {missing}")
        out = df[[patient_id_column, image_id_column]].copy()
        out[patient_id_column] = out[patient_id_column].astype(str)
        out[image_id_column] = out[image_id_column].astype(str)
        return out

    @classmethod
    def _ids_records(
        cls,
        df: pd.DataFrame,
        *,
        patient_id_column: str,
        image_id_column: str,
    ) -> list[dict[str, str]]:
        ids = cls._normalize_split_id_columns(
            df, patient_id_column=patient_id_column, image_id_column=image_id_column
        )
        return ids.rename(
            columns={patient_id_column: "patient_id", image_id_column: "image_id"}
        ).to_dict(orient="records")

    @staticmethod
    def _frame_from_split_ids(
        ds: pd.DataFrame,
        records: list[dict[str, Any]] | pd.DataFrame,
        *,
        split_name: str,
        patient_id_column: str,
        image_id_column: str,
    ) -> pd.DataFrame:
        """Reconstruye un split preservando el orden exacto del archivo de IDs."""
        if isinstance(records, pd.DataFrame):
            id_df = records[[patient_id_column, image_id_column]].copy()
        else:
            id_df = pd.DataFrame(records)
            if id_df.empty:
                id_df = pd.DataFrame(columns=["patient_id", "image_id"])
            id_df = id_df.rename(
                columns={"patient_id": patient_id_column, "image_id": image_id_column}
            )
            id_df = id_df[[patient_id_column, image_id_column]]

        id_df[patient_id_column] = id_df[patient_id_column].astype(str)
        id_df[image_id_column] = id_df[image_id_column].astype(str)
        id_df = id_df.reset_index(drop=True)
        id_df["_split_order"] = np.arange(len(id_df), dtype=np.int64)

        ds_keys = ds.copy()
        ds_keys[patient_id_column] = ds_keys[patient_id_column].astype(str)
        ds_keys[image_id_column] = ds_keys[image_id_column].astype(str)

        merged = id_df.merge(
            ds_keys,
            on=[patient_id_column, image_id_column],
            how="left",
            indicator=True,
        )
        missing = int((merged["_merge"] != "both").sum())
        if missing:
            raise ValueError(
                f"Split {split_name!r}: {missing}/{len(id_df)} ids no estan en el dataset actual. "
                "Regenera el archivo de splits o revisa POSITIVE_MODE / reduce."
            )
        return (
            merged.sort_values("_split_order", kind="mergesort")
            .drop(columns=["_merge", "_split_order"])
            .reset_index(drop=True)
        )
