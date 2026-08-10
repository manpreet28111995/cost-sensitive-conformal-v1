from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class CostMatrix:
    false_positive: float = 1.0
    false_negative: float = 10.0
    defer: float = 0.0

    def prediction_cost(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        y_true = np.asarray(y_true, dtype=int)
        y_pred = np.asarray(y_pred, dtype=int)
        costs = np.zeros(y_true.shape[0], dtype=float)
        costs[(y_true == 0) & (y_pred == 1)] = self.false_positive
        costs[(y_true == 1) & (y_pred == 0)] = self.false_negative
        return costs


def evaluate_prediction_sets(
    y_true: np.ndarray,
    prediction_sets: list[frozenset[int]],
    costs: CostMatrix,
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    if len(prediction_sets) != y_true.size:
        raise ValueError("prediction_sets and y_true must have the same length")

    prediction_mask = np.zeros((y_true.size, 2), dtype=bool)
    for idx, pred_set in enumerate(prediction_sets):
        if not pred_set <= {0, 1}:
            raise ValueError("prediction sets must contain only binary labels 0/1")
        prediction_mask[idx, list(pred_set)] = True
    return evaluate_prediction_mask(y_true, prediction_mask, costs)


def evaluate_prediction_mask(
    y_true: np.ndarray,
    prediction_mask: NDArray[np.bool_],
    costs: CostMatrix,
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    prediction_mask = np.asarray(prediction_mask, dtype=bool)
    if prediction_mask.shape != (y_true.size, 2):
        raise ValueError("prediction_mask must have shape (n_samples, 2)")

    contains_truth = prediction_mask[np.arange(y_true.size), y_true].astype(int)
    set_sizes = prediction_mask.sum(axis=1)
    automated = set_sizes == 1
    deferred = ~automated

    per_case_cost = prediction_mask_costs(y_true, prediction_mask, costs)

    metrics = {
        "marginal_coverage": float(np.mean(contains_truth)),
        "coverage_class_0": _safe_mean(contains_truth[y_true == 0]),
        "coverage_class_1": _safe_mean(contains_truth[y_true == 1]),
        "average_set_size": float(np.mean(set_sizes)),
        "abstention_rate": float(np.mean(deferred)),
        "mean_cost": float(np.mean(per_case_cost)),
        "mean_cost_automated_only": np.nan,
    }
    if automated.any():
        metrics["mean_cost_automated_only"] = float(np.mean(per_case_cost[automated]))
    return metrics


def evaluate_hard_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    costs: CostMatrix,
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    correct = (y_true == y_pred).astype(int)
    mean_cost = float(np.mean(hard_prediction_costs(y_true, y_pred, costs)))
    return {
        "marginal_coverage": float(np.mean(correct)),
        "coverage_class_0": _safe_mean(correct[y_true == 0]),
        "coverage_class_1": _safe_mean(correct[y_true == 1]),
        "average_set_size": 1.0,
        "abstention_rate": 0.0,
        "mean_cost": mean_cost,
        "mean_cost_automated_only": mean_cost,
    }


def prediction_mask_costs(
    y_true: np.ndarray,
    prediction_mask: NDArray[np.bool_],
    costs: CostMatrix,
) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=int)
    prediction_mask = np.asarray(prediction_mask, dtype=bool)
    if prediction_mask.shape != (y_true.size, 2):
        raise ValueError("prediction_mask must have shape (n_samples, 2)")

    set_sizes = prediction_mask.sum(axis=1)
    automated = set_sizes == 1
    y_pred = np.argmax(prediction_mask, axis=1)
    per_case_cost = np.full(y_true.shape[0], costs.defer, dtype=float)
    if automated.any():
        per_case_cost[automated] = costs.prediction_cost(
            y_true[automated], y_pred[automated]
        )
    return per_case_cost


def hard_prediction_costs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    costs: CostMatrix,
) -> np.ndarray:
    return costs.prediction_cost(y_true, y_pred)


def tune_cost_threshold(
    cal_y: np.ndarray,
    cal_positive_proba: np.ndarray,
    costs: CostMatrix,
    grid_size: int = 501,
) -> float:
    thresholds = np.linspace(0.0, 1.0, grid_size)
    pred = cal_positive_proba[:, None] >= thresholds[None, :]
    y = np.asarray(cal_y, dtype=int)[:, None]
    fp = (y == 0) & pred
    fn = (y == 1) & ~pred
    mean_costs = (
        costs.false_positive * fp.mean(axis=0)
        + costs.false_negative * fn.mean(axis=0)
    )
    return float(thresholds[int(np.argmin(mean_costs))])


def _safe_mean(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.mean(values))
