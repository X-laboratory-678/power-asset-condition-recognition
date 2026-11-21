"""Paper probability fusion rules; fits only on support OOF predictions."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit, softmax

from .metrics import check_probabilities


def full_member_fusion(task: np.ndarray, reference: np.ndarray, alpha: float) -> np.ndarray:
    a, b = check_probabilities(task), check_probabilities(reference)
    if a.shape != b.shape or not 0 <= alpha <= 1:
        raise ValueError("full-data member shapes or weight invalid")
    return alpha * b + (1 - alpha) * a


def full_ensemble(task_members: np.ndarray, reference_members: np.ndarray, alphas=(0.9, 0.7, 0.7)) -> np.ndarray:
    if task_members.shape != reference_members.shape or task_members.shape[0] != 3:
        raise ValueError("full-data ensemble requires three paired members")
    return np.mean([full_member_fusion(task_members[i], reference_members[i], alphas[i]) for i in range(3)], axis=0)


def fixed_fewshot(task: np.ndarray, reference: np.ndarray) -> np.ndarray:
    a, b = check_probabilities(task), check_probabilities(reference)
    if a.shape != b.shape:
        raise ValueError("few-shot branch shapes differ")
    return 0.25 * a + 0.75 * b


def _nll(y: np.ndarray, p: np.ndarray) -> float:
    return float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-7, 1.0)).mean())


def _temperature(p: np.ndarray, y: np.ndarray) -> float:
    logits = np.log(np.clip(check_probabilities(p), 1e-7, 1.0))
    def objective(log_t):
        return _nll(y, softmax(logits / np.exp(log_t), axis=1)) + 0.5 * log_t * log_t
    bounds = (np.log(0.5), np.log(2.0))
    result = minimize_scalar(objective, method="bounded", bounds=bounds)
    candidates = [bounds[0], bounds[1], float(result.x)]
    return float(np.exp(min(candidates, key=lambda x: (objective(x), x))))


def _calibrate(p: np.ndarray, temperature: float) -> np.ndarray:
    return softmax(np.log(np.clip(check_probabilities(p), 1e-7, 1.0)) / temperature, axis=1)


def _entropy(p: np.ndarray) -> np.ndarray:
    return -(p * np.log(np.clip(p, 1e-7, 1))).sum(axis=1)


@dataclass(frozen=True)
class EntropyGate:
    task_temperature: float
    reference_temperature: float
    beta: float

    @classmethod
    def fit(cls, task_oof: np.ndarray, reference_oof: np.ndarray, y: np.ndarray) -> "EntropyGate":
        a, b = check_probabilities(task_oof), check_probabilities(reference_oof)
        y = np.asarray(y, dtype=np.int64)
        if a.shape != b.shape or y.shape != (len(a),):
            raise ValueError("support OOF inputs are misaligned")
        ta, tb = _temperature(a, y), _temperature(b, y)
        ac, bc = _calibrate(a, ta), _calibrate(b, tb)
        delta = (_entropy(bc) - _entropy(ac)) / np.log(11)
        def objective(beta):
            weight = np.clip(expit(np.log(0.25 / 0.75) + beta * delta), 0.1, 0.5)
            mixed = weight[:, None] * ac + (1 - weight[:, None]) * bc
            return _nll(y, mixed) + 0.5 * beta * beta
        result = minimize_scalar(objective, method="bounded", bounds=(0.0, 1.5))
        beta = min((0.0, 1.5, float(result.x)), key=lambda x: (objective(x), x))
        return cls(ta, tb, beta)

    def predict(self, task: np.ndarray, reference: np.ndarray) -> np.ndarray:
        ac = _calibrate(task, self.task_temperature)
        bc = _calibrate(reference, self.reference_temperature)
        if ac.shape != bc.shape:
            raise ValueError("gate branch shapes differ")
        delta = (_entropy(bc) - _entropy(ac)) / np.log(11)
        weight = np.clip(expit(np.log(0.25 / 0.75) + self.beta * delta), 0.1, 0.5)
        return weight[:, None] * ac + (1 - weight[:, None]) * bc
