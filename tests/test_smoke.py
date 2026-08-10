from __future__ import annotations

import numpy as np

from cost_conformal.conformal import AdaptiveConformalPredictor, ConformalPredictor
from cost_conformal.data import OPENML_DATASETS, _OPENML_SPECS
from cost_conformal.evaluation import (
    CostMatrix,
    evaluate_prediction_sets,
    prediction_mask_costs,
)
from scripts.run_benchmark import validate_deferral_monotonicity
import pandas as pd


def test_mondrian_prediction_sets_and_metrics() -> None:
    cal_proba = np.array(
        [
            [0.90, 0.10],
            [0.85, 0.15],
            [0.20, 0.80],
            [0.10, 0.90],
        ]
    )
    cal_y = np.array([0, 0, 1, 1])
    test_proba = np.array([[0.70, 0.30], [0.35, 0.65]])
    test_y = np.array([0, 1])

    predictor = ConformalPredictor.fit(cal_proba, cal_y, alpha=0.10, mode="mondrian")
    pred_sets = predictor.predict_sets(test_proba)
    metrics = evaluate_prediction_sets(test_y, pred_sets, CostMatrix())

    assert len(pred_sets) == 2
    assert 0.0 <= metrics["marginal_coverage"] <= 1.0
    assert 0.0 <= metrics["abstention_rate"] <= 1.0


def test_openml_benchmark_registry_has_15_datasets() -> None:
    assert len(OPENML_DATASETS) == 15
    assert "creditcard_2023" not in OPENML_DATASETS


def test_sick_numeric_uses_openml_id_with_target() -> None:
    assert _OPENML_SPECS["sick_numeric"]["kw"] == {"data_id": 41946}
    assert _OPENML_SPECS["seismic_bumps"]["kw"] == {"data_id": 45562}


def test_openml_specs_are_pinned_to_data_ids() -> None:
    assert all(set(spec["kw"]) == {"data_id"} for spec in _OPENML_SPECS.values())
    assert all(isinstance(spec["kw"]["data_id"], int) for spec in _OPENML_SPECS.values())


def test_prediction_mask_costs_are_per_case() -> None:
    y = np.array([0, 1, 1])
    mask = np.array([[True, False], [True, False], [True, True]])
    costs = prediction_mask_costs(y, mask, CostMatrix(false_negative=10.0, defer=2.0))

    assert costs.tolist() == [0.0, 10.0, 2.0]


def test_adaptive_conformal_predictor_returns_binary_mask() -> None:
    cal_proba = np.array([[0.9, 0.1], [0.3, 0.7], [0.8, 0.2], [0.2, 0.8]])
    cal_y = np.array([0, 1, 0, 1])
    predictor = AdaptiveConformalPredictor.fit(
        cal_proba, cal_y, alpha=0.1, penalty=0.01
    )
    mask = predictor.predict_mask(np.array([[0.6, 0.4]]))

    assert mask.shape == (1, 2)
    assert mask.dtype == bool


def test_deferral_monotonicity_guard_accepts_valid_curve() -> None:
    frame = pd.DataFrame(
        {
            "dataset": ["toy"] * 3,
            "defer_cost": [0.0, 0.5, 1.0],
            "mondrian_cost": [0.1, 0.2, 0.3],
            "threshold_cost": [0.4, 0.4, 0.4],
        }
    )
    validate_deferral_monotonicity(frame)
