"""Support-only linear readouts and the frozen visual partner."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.preprocessing import StandardScaler


def _normalized_blocks(*blocks: np.ndarray) -> np.ndarray:
    result = []
    for block in blocks:
        x = np.asarray(block, dtype=np.float32)
        norm = np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
        result.append(x / norm)
    return np.concatenate(result, axis=1)


def qwen2_dino_features(qwen2: np.ndarray, dino_cls: np.ndarray, dino_patch: np.ndarray) -> np.ndarray:
    if len(qwen2) != len(dino_cls) or len(qwen2) != len(dino_patch):
        raise ValueError("feature blocks have different row counts")
    if qwen2.shape[1] != 3584 or dino_cls.shape[1] != 768 or dino_patch.shape[1] != 768:
        raise ValueError("expected 3584 + 768 + 768 frozen partner")
    return _normalized_blocks(qwen2, dino_cls, dino_patch)


def fit_logistic(train_x: np.ndarray, train_y: np.ndarray, c: float):
    x = np.asarray(train_x, dtype=np.float32)
    y = np.asarray(train_y, dtype=np.int64)
    if x.ndim != 2 or y.shape != (len(x),) or not np.isfinite(x).all() or set(y.tolist()) != set(range(11)):
        raise ValueError("readout needs finite features and all 11 support classes")
    scaler = StandardScaler().fit(x)
    classifier = LogisticRegression(C=c, max_iter=1000, solver="lbfgs", tol=1e-6, random_state=0)
    classifier.fit(scaler.transform(x), y)
    return scaler, classifier


def predict_logistic(readout, query_x: np.ndarray) -> np.ndarray:
    scaler, classifier = readout
    if classifier.classes_.tolist() != list(range(11)):
        raise ValueError("classifier classes are not canonical")
    return classifier.predict_proba(scaler.transform(np.asarray(query_x, dtype=np.float32)))


def fit_frozen_partner(x: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int = 20261301):
    """Select C by support-only group CV, then refit all current supports."""
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    groups = np.asarray(groups).astype(str)
    if len(x) != len(y) or len(y) != len(groups):
        raise ValueError("support feature/label/group counts differ")
    group_class = {}
    for label, group in zip(y, groups):
        if group in group_class and group_class[group] != label:
            raise ValueError("support group has multiple classes")
        group_class[group] = label
    fold_of_group = {}
    rng = np.random.default_rng(seed)
    for label in range(11):
        class_groups = sorted(g for g, value in group_class.items() if value == label)
        if len(class_groups) < 2:
            raise ValueError("support class needs two distinct groups for CV")
        rng.shuffle(class_groups)
        fold_of_group.update({group: index % 2 for index, group in enumerate(class_groups)})
    folds = []
    for fold in range(2):
        va = np.array([i for i, group in enumerate(groups) if fold_of_group[group] == fold])
        tr = np.array([i for i, group in enumerate(groups) if fold_of_group[group] != fold])
        folds.append((tr, va))
    for tr, va in folds:
        if set(groups[tr]) & set(groups[va]) or set(y[tr]) != set(range(11)):
            raise ValueError("support CV is not class-complete or group-disjoint")
    scores = {}
    for c in (0.01, 0.1, 1.0):
        scores[c] = float(np.mean([
            np.mean(predict_logistic(fit_logistic(x[tr], y[tr], c), x[va]).argmax(axis=1) == y[va])
            for tr, va in folds
        ]))
    selected = min(scores, key=lambda c: (-scores[c], c))
    return fit_logistic(x, y, selected), {"selected_c": selected, "cv_accuracy": scores}


def frozen_qwen3_ridge(train_x: np.ndarray, train_y: np.ndarray, query_x: np.ndarray) -> np.ndarray:
    scaler = StandardScaler().fit(train_x)
    clf = RidgeClassifier(alpha=1.0).fit(scaler.transform(train_x), train_y)
    logits = clf.decision_function(scaler.transform(query_x))
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    return probabilities / probabilities.sum(axis=1, keepdims=True)
