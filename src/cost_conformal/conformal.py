from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray


ConformalMode = Literal["marginal", "mondrian"]


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample split-conformal quantile for nonconformity scores."""
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or scores.size == 0:
        raise ValueError("scores must be a non-empty 1D array")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")

    sorted_scores = np.sort(scores)
    k = int(np.ceil((scores.size + 1) * (1.0 - alpha)))
    k = min(max(k, 1), scores.size)
    return float(sorted_scores[k - 1])


@dataclass(frozen=True)
class ConformalPredictor:
    """Split conformal predictor for binary probabilistic classifiers."""

    alpha: float
    mode: ConformalMode = "marginal"
    class_thresholds: tuple[float, float] | None = None
    marginal_threshold: float | None = None

    @classmethod
    def fit(
        cls,
        cal_proba: np.ndarray,
        cal_y: np.ndarray,
        alpha: float,
        mode: ConformalMode,
    ) -> "ConformalPredictor":
        cal_proba = _validate_proba(cal_proba)
        cal_y = np.asarray(cal_y, dtype=int)
        if set(np.unique(cal_y)) - {0, 1}:
            raise ValueError("cal_y must contain binary labels encoded as 0/1")

        true_class_scores = 1.0 - cal_proba[np.arange(cal_y.size), cal_y]

        if mode == "marginal":
            return cls(
                alpha=alpha,
                mode=mode,
                marginal_threshold=conformal_quantile(true_class_scores, alpha),
            )

        if mode == "mondrian":
            thresholds: list[float] = []
            for label in (0, 1):
                label_scores = true_class_scores[cal_y == label]
                if label_scores.size == 0:
                    raise ValueError(f"calibration split has no class {label} examples")
                thresholds.append(conformal_quantile(label_scores, alpha))
            return cls(alpha=alpha, mode=mode, class_thresholds=tuple(thresholds))

        raise ValueError(f"unsupported conformal mode: {mode}")

    def predict_mask(self, proba: np.ndarray) -> NDArray[np.bool_]:
        """Return an (n_samples, 2) mask indicating which labels are in each set."""
        proba = _validate_proba(proba)
        scores = 1.0 - proba
        if self.mode == "marginal":
            if self.marginal_threshold is None:
                raise ValueError("marginal predictor has no threshold")
            return scores <= self.marginal_threshold

        if self.class_thresholds is None:
            raise ValueError("mondrian predictor has no class thresholds")
        return scores <= np.asarray(self.class_thresholds, dtype=float)

    def predict_sets(self, proba: np.ndarray) -> list[frozenset[int]]:
        mask = self.predict_mask(proba)
        return [frozenset(np.flatnonzero(row).astype(int).tolist()) for row in mask]


@dataclass(frozen=True)
class AdaptiveConformalPredictor:
    """Binary APS/RAPS split conformal predictor."""

    alpha: float
    threshold: float
    penalty: float = 0.0
    seed: int = 0

    @classmethod
    def fit(
        cls,
        cal_proba: np.ndarray,
        cal_y: np.ndarray,
        alpha: float,
        penalty: float = 0.0,
        seed: int = 0,
    ) -> "AdaptiveConformalPredictor":
        cal_proba = _validate_proba(cal_proba)
        cal_y = np.asarray(cal_y, dtype=int)
        u = np.random.default_rng(seed).uniform(size=(cal_proba.shape[0], 1))
        scores = _adaptive_scores(cal_proba, penalty, u)
        true_scores = scores[np.arange(cal_y.size), cal_y]
        return cls(
            alpha=alpha,
            threshold=conformal_quantile(true_scores, alpha),
            penalty=penalty,
            seed=seed,
        )

    def predict_mask(self, proba: np.ndarray) -> NDArray[np.bool_]:
        proba = _validate_proba(proba)
        u = np.random.default_rng(self.seed + 1).uniform(size=(proba.shape[0], 1))
        return _adaptive_scores(proba, self.penalty, u) <= self.threshold


def _adaptive_scores(proba: np.ndarray, penalty: float, u: np.ndarray) -> np.ndarray:
    """Randomized APS/RAPS nonconformity scores (Romano et al., 2020).

    Score for a label = probability mass of all strictly-higher-ranked labels
    plus a random fraction ``u`` of the label's own mass. The randomization is
    what prevents the binary threshold from saturating at 1.0 (which previously
    forced every prediction set to contain both labels). RAPS adds ``penalty``
    to ranks beyond the first (k_reg = 1)."""
    proba = _validate_proba(proba)
    order = np.argsort(-proba, axis=1)
    sorted_proba = np.take_along_axis(proba, order, axis=1)
    cumulative = np.cumsum(sorted_proba, axis=1)
    exclusive = cumulative - sorted_proba          # mass strictly above each rank
    sorted_scores = exclusive + u * sorted_proba    # randomized inclusion of own mass
    if penalty:
        sorted_scores = sorted_scores + penalty * np.array([0.0, 1.0])
    scores = np.empty_like(sorted_scores)
    np.put_along_axis(scores, order, sorted_scores, axis=1)
    return scores


def _validate_proba(proba: np.ndarray) -> np.ndarray:
    proba = np.asarray(proba, dtype=float)
    if proba.ndim != 2 or proba.shape[1] != 2:
        raise ValueError("proba must have shape (n_samples, 2)")
    if not np.all(np.isfinite(proba)):
        raise ValueError("proba contains non-finite values")
    return proba
