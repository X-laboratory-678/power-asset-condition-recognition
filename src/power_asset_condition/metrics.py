"""Eleven-class metrics and audited-row re-scoring."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

N_CLASSES = 11


def check_probabilities(probabilities: np.ndarray) -> np.ndarray:
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != N_CLASSES or not np.isfinite(p).all() or (p < 0).any():
        raise ValueError("expected finite nonnegative N x 11 probabilities")
    if not np.allclose(p.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("probabilities are not row normalized")
    return p


def score(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    p = check_probabilities(probabilities)
    y = np.asarray(labels, dtype=np.int64)
    if y.shape != (len(p),) or not np.isin(y, np.arange(N_CLASSES)).all():
        raise ValueError("labels must be one class ID per row")
    pred = p.argmax(axis=1)
    return {
        "n": len(y),
        "correct": int((pred == y).sum()),
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, labels=np.arange(N_CLASSES), average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "per_class_recall": [float((pred[y == c] == c).mean()) if np.any(y == c) else None for c in range(N_CLASSES)],
    }


def paired_score(label_by_id: dict[str, int], sample_ids: list[str], probabilities: np.ndarray, keep: set[str] | None = None) -> dict:
    p = check_probabilities(probabilities)
    if len(sample_ids) != len(p) or len(sample_ids) != len(set(sample_ids)):
        raise ValueError("prediction IDs are duplicated or misaligned")
    indices = [i for i, sid in enumerate(sample_ids) if keep is None or sid in keep]
    if keep is not None and {sample_ids[i] for i in indices} != keep:
        raise ValueError("retained audit IDs are absent from predictions")
    if any(sample_ids[i] not in label_by_id for i in indices):
        raise ValueError("missing label for selected prediction")
    return score(np.array([label_by_id[sample_ids[i]] for i in indices]), p[indices])
