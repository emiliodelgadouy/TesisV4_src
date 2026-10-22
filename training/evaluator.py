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


def bootstrap_draws(y_true, *, groups=None, rng):
    """Generador de indices de remuestreo: por grupo (p. ej. paciente) si hay ``groups``;
    si no, imagenes con reposicion dentro de cada clase."""
    y_true = np.asarray(y_true).reshape(-1)
    if groups is not None:
        groups = np.asarray(groups).reshape(-1)
        uniq, inverse = np.unique(groups, return_inverse=True)
        index_by_group = [np.flatnonzero(inverse == group) for group in range(len(uniq))]
        while True:
            picked = rng.integers(0, len(uniq), size=len(uniq))
            yield np.concatenate([index_by_group[i] for i in picked])
    positive = np.flatnonzero(y_true == 1)
    negative = np.flatnonzero(y_true == 0)
    while True:
        yield np.concatenate([rng.choice(positive, size=len(positive), replace=True), rng.choice(negative, size=len(negative), replace=True)])


class ClinicalMetrics:
    """Metricas para comparar modelos sobre las mismas imagenes: discriminacion,
    puntos operativos, calibracion y, con ``threshold``, metricas de umbral."""

    CALIBRATION_BINS = 10

    @staticmethod
    def compute(y_true, y_prob, *, threshold: float | None = None) -> dict[str, float]:
        from sklearn.metrics import average_precision_score, roc_auc_score

        y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
        y_prob = np.asarray(y_prob, dtype=np.float64).reshape(-1)
        n_pos = int((y_true == 1).sum())
        n_neg = int((y_true == 0).sum())
        out = {"n_pos": n_pos, "n_neg": n_neg, "prevalence": n_pos / max(n_pos + n_neg, 1)}
        if n_pos == 0 or n_neg == 0:
            return out
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = float(average_precision_score(y_true, y_prob))

        def sens_at_spec(target: float) -> float:
            ok = (1.0 - fpr) >= target
            return float(tpr[ok].max()) if ok.any() else float("nan")

        ok_recall = recall >= 0.80
        out.update(
            {
                "roc_auc": float(roc_auc_score(y_true, y_prob)),
                "pr_auc": pr_auc,
                "pr_auc_over_baseline": pr_auc / out["prevalence"],
                "sens_at_spec_90": sens_at_spec(0.90),
                "sens_at_spec_95": sens_at_spec(0.95),
                "prec_at_recall_80": float(precision[ok_recall].max()) if ok_recall.any() else float("nan"),
                "brier": float(np.mean((y_prob - y_true) ** 2)),
                "ece": ClinicalMetrics.expected_calibration_error(y_true, y_prob),
            }
        )
        if threshold is not None:
            out.update(ClinicalMetrics.at_threshold(y_true, y_prob, threshold))
        return out

    @staticmethod
    def at_threshold(y_true, y_prob, threshold: float) -> dict[str, float]:
        """Sensibilidad, especificidad, PPV, NPV, F1 y exactitud balanceada con ``y_prob >= threshold``."""
        y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
        y_pred = (np.asarray(y_prob, dtype=np.float64).reshape(-1) >= threshold).astype(np.int64)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        tn = int(((y_pred == 0) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())

        def ratio(num, den):
            return num / den if den else float("nan")

        sens, spec = ratio(tp, tp + fn), ratio(tn, tn + fp)
        return {
            "threshold": float(threshold),
            "sensitivity": sens,
            "specificity": spec,
            "ppv": ratio(tp, tp + fp),
            "npv": ratio(tn, tn + fn),
            "f1": ratio(2 * tp, 2 * tp + fp + fn),
            "balanced_accuracy": (sens + spec) / 2,
        }

    @classmethod
    def calibration_curve(cls, y_true, y_prob, *, bins: int | None = None):
        """Bins de igual ancho en [0, 1]: probabilidad media, fraccion de positivos y n (solo bins no vacios)."""
        y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
        y_prob = np.asarray(y_prob, dtype=np.float64).reshape(-1)
        edges = np.linspace(0.0, 1.0, int(bins or cls.CALIBRATION_BINS) + 1)
        which = np.clip(np.digitize(y_prob, edges[1:-1]), 0, len(edges) - 2)
        mean_prob, frac_pos, counts = [], [], []
        for b in range(len(edges) - 1):
            in_bin = which == b
            if in_bin.any():
                mean_prob.append(float(y_prob[in_bin].mean()))
                frac_pos.append(float(y_true[in_bin].mean()))
                counts.append(int(in_bin.sum()))
        return np.asarray(mean_prob), np.asarray(frac_pos), np.asarray(counts)

    @classmethod
    def expected_calibration_error(cls, y_true, y_prob, *, bins: int | None = None) -> float:
        mean_prob, frac_pos, counts = cls.calibration_curve(y_true, y_prob, bins=bins)
        if counts.sum() == 0:
            return float("nan")
        return float(np.sum(counts * np.abs(mean_prob - frac_pos)) / counts.sum())

    @staticmethod
    def bootstrap_ci(y_true, y_prob, *, groups=None, samples: int = 1000, seed: int = 42) -> dict[str, float]:
        """IC 95 % de ROC-AUC y PR-AUC; remuestrea ``groups`` (paciente) si se pasan."""
        from sklearn.metrics import average_precision_score, roc_auc_score

        y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
        y_prob = np.asarray(y_prob, dtype=np.float64).reshape(-1)
        if samples <= 0 or np.unique(y_true).size < 2:
            return {}
        draws = bootstrap_draws(y_true, groups=groups, rng=np.random.default_rng(seed))
        roc, pr = [], []
        for _ in range(int(samples) * 5):
            if len(roc) >= samples:
                break
            idx = next(draws)
            if np.unique(y_true[idx]).size < 2:
                continue
            roc.append(roc_auc_score(y_true[idx], y_prob[idx]))
            pr.append(average_precision_score(y_true[idx], y_prob[idx]))
        if not roc:
            return {}
        return {
            "roc_auc_ci_low": float(np.quantile(roc, 0.025)),
            "roc_auc_ci_high": float(np.quantile(roc, 0.975)),
            "pr_auc_ci_low": float(np.quantile(pr, 0.025)),
            "pr_auc_ci_high": float(np.quantile(pr, 0.975)),
        }


class BreastAggregation:
    """Nivel mama: una fila por (estudio, lateralidad) con la probabilidad maxima entre
    sus vistas (CC/MLO) y la etiqueta maxima. Es la unidad de decision clinica."""

    KEYS = (("study_id", "laterality"), ("patient_id", "laterality"))

    @classmethod
    def keys_for(cls, frame) -> list[str] | None:
        for keys in cls.KEYS:
            if all(key in frame.columns for key in keys):
                return list(keys)
        return None

    @classmethod
    def aggregate(cls, frame, y_true, y_prob):
        """``(y_true, y_prob, groups)`` por mama (``groups`` = paciente), o ``None`` si faltan columnas.

        ``frame`` tiene que estar alineado fila a fila con ``y_true`` / ``y_prob``.
        """
        keys = cls.keys_for(frame)
        if keys is None or len(frame) != len(y_prob):
            return None
        extra = ["patient_id"] if "patient_id" in frame.columns and "patient_id" not in keys else []
        table = frame[keys + extra].reset_index(drop=True).copy()
        table[keys] = table[keys].astype(str)
        table["y_true"] = np.asarray(y_true, dtype=np.int64).reshape(-1)
        table["y_prob"] = np.asarray(y_prob, dtype=np.float64).reshape(-1)
        agg = {"y_true": "max", "y_prob": "max", **{column: "first" for column in extra}}
        grouped = table.groupby(keys, sort=True).agg(agg).reset_index()
        groups = grouped["patient_id"].astype(str).to_numpy() if "patient_id" in grouped.columns else None
        return grouped["y_true"].to_numpy(), grouped["y_prob"].to_numpy(), groups


class PairedBootstrap:
    """Comparacion de dos modelos sobre las mismas imagenes (bootstrap pareado)."""

    @staticmethod
    def auc_delta(y_true, prob_ref, prob_alt, *, groups=None, samples: int = 1000, seed: int = 42) -> dict[str, float]:
        """ROC-AUC y PR-AUC de ``alt - ref`` con IC 95 %.

        Remuestrea grupos (p. ej. paciente) si se pasan; si no, imagenes por clase.
        """
        from sklearn.metrics import average_precision_score, roc_auc_score

        y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
        prob_ref = np.asarray(prob_ref, dtype=np.float64).reshape(-1)
        prob_alt = np.asarray(prob_alt, dtype=np.float64).reshape(-1)
        draws = bootstrap_draws(y_true, groups=groups, rng=np.random.default_rng(seed))

        out = {
            "roc_auc_ref": float(roc_auc_score(y_true, prob_ref)),
            "roc_auc_alt": float(roc_auc_score(y_true, prob_alt)),
            "pr_auc_ref": float(average_precision_score(y_true, prob_ref)),
            "pr_auc_alt": float(average_precision_score(y_true, prob_alt)),
        }
        roc_delta, pr_delta = [], []
        for _ in range(int(samples) * 5):
            if len(roc_delta) >= samples:
                break
            idx = next(draws)
            yt = y_true[idx]
            if np.unique(yt).size < 2:
                continue
            roc_delta.append(roc_auc_score(yt, prob_alt[idx]) - roc_auc_score(yt, prob_ref[idx]))
            pr_delta.append(average_precision_score(yt, prob_alt[idx]) - average_precision_score(yt, prob_ref[idx]))
        out["roc_auc_delta"] = out["roc_auc_alt"] - out["roc_auc_ref"]
        out["pr_auc_delta"] = out["pr_auc_alt"] - out["pr_auc_ref"]
        if roc_delta:
            out["roc_auc_delta_ci_low"] = float(np.quantile(roc_delta, 0.025))
            out["roc_auc_delta_ci_high"] = float(np.quantile(roc_delta, 0.975))
            out["pr_auc_delta_ci_low"] = float(np.quantile(pr_delta, 0.025))
            out["pr_auc_delta_ci_high"] = float(np.quantile(pr_delta, 0.975))
        return out


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
