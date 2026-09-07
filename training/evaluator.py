from __future__ import annotations

import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve

from src.dataset.provider import as_tf_dataset


class Predictor:
    """Prediccion por batches y umbrales sobre logits/probabilidades."""

    @staticmethod
    def logit_initial_bias(n_positive: int, n_negative: int, *, eps: float = 1e-6) -> float:
        """Log-odds inicial; evita division por cero / ±inf con conteos nulos."""
        n_pos = max(float(n_positive), eps)
        n_neg = max(float(n_negative), eps)
        return float(np.log(n_pos / n_neg))

    @staticmethod
    def resolve_steps_per_execution(
        n_rows: int,
        batch_size: int,
        *,
        max_steps: int = 32,
    ) -> int:
        """Acota steps_per_execution al numero real de batches por epoca."""
        steps_per_epoch = max(1, (int(n_rows) + int(batch_size) - 1) // int(batch_size))
        return max(1, min(int(max_steps), steps_per_epoch))

    @staticmethod
    def _sigmoid(z) -> np.ndarray:
        z = np.asarray(z, dtype=np.float64)
        return np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))

    @classmethod
    def _eval_tf_dataset(cls, dataset):
        """Vista ordenada y estable para evaluacion (InspectDataset o tf.data)."""
        if hasattr(dataset, "ordered"):
            return as_tf_dataset(dataset.ordered())
        return as_tf_dataset(dataset)

    @classmethod
    def predict_probs_and_labels(cls, model, dataset) -> tuple[np.ndarray, np.ndarray]:
        """Una sola pasada sobre el dataset: devuelve (y_true, y_prob) alineados.

        Itera el pipeline ``tf.data`` una sola vez, tomando labels y logits del mismo
        batch (en vez de recorrerlo dos veces decodificando/croppeando las imagenes).
        """
        eval_ds = cls._eval_tf_dataset(dataset)
        keras_model = model.model if hasattr(model, "model") else model

        labels_batches: list[np.ndarray] = []
        logits_batches: list[np.ndarray] = []
        for xb, yb in eval_ds:
            labels_batches.append(np.asarray(yb, dtype=np.int64).reshape(-1))
            logits_batches.append(
                np.asarray(keras_model.predict_on_batch(xb), dtype=np.float64).reshape(-1)
            )

        if not labels_batches:
            return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.float64)

        y_true = np.concatenate(labels_batches)
        y_prob = cls._sigmoid(np.concatenate(logits_batches))
        return y_true, y_prob


class ThresholdSelector:
    """Seleccion de umbral de probabilidad sobre validacion."""

    @staticmethod
    def apply(y_prob, threshold: float) -> np.ndarray:
        return (y_prob >= threshold).astype(np.int64)

    @staticmethod
    def _inputs_ok(y_true, y_prob) -> bool:
        y_true = np.asarray(y_true).reshape(-1)
        y_prob = np.asarray(y_prob).reshape(-1)
        if y_true.size == 0 or y_prob.size == 0:
            return False
        classes = np.unique(y_true.astype(int))
        return classes.size >= 2

    @classmethod
    def youden_j(cls, y_true, y_prob, *, default: float = 0.5) -> float:
        """Umbral que maximiza el indice J de Youden (sensibilidad + especificidad - 1)."""
        if not cls._inputs_ok(y_true, y_prob):
            print("Advertencia: threshold_youden_j sin ambas clases; se usa default", default)
            return float(default)
        fpr, tpr, thr = roc_curve(y_true, y_prob)
        best = int(np.argmax(tpr - fpr))
        value = float(thr[best])
        if not np.isfinite(value):
            return float(default)
        return value

    @classmethod
    def best_f1(cls, y_true, y_prob, *, default: float = 0.5) -> float:
        """Umbral que maximiza F1 sobre la clase positiva."""
        if not cls._inputs_ok(y_true, y_prob):
            print("Advertencia: threshold_best_f1 sin ambas clases; se usa default", default)
            return float(default)
        precision, recall, thr = precision_recall_curve(y_true, y_prob)
        f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
        if f1.size == 0:
            return float(default)
        best = int(np.argmax(f1))
        value = float(thr[best])
        if not np.isfinite(value):
            return float(default)
        return value

    @classmethod
    def recall_target(
        cls,
        y_true,
        y_prob,
        target_recall: float = 0.90,
        *,
        default: float = 0.5,
    ) -> float:
        """Umbral mas alto (mayor precision) que aun garantiza recall >= target_recall."""
        if not cls._inputs_ok(y_true, y_prob):
            print(
                "Advertencia: threshold_recall_target sin ambas clases; se usa default",
                default,
            )
            return float(default)
        precision, recall, thr = precision_recall_curve(y_true, y_prob)
        recall_t, precision_t = recall[:-1], precision[:-1]
        ok = np.flatnonzero(recall_t >= target_recall)
        if ok.size == 0:
            return cls.best_f1(y_true, y_prob, default=default)
        best = int(ok[np.argmax(precision_t[ok])])
        value = float(thr[best])
        if not np.isfinite(value):
            return float(default)
        return value
