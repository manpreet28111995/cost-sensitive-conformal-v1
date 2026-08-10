# Results v3

## Purpose

`results_v3/` reruns benchmark after adding `random_oversampling_threshold`. `results_v2/` is preserved as prior result set.

Same benchmark design in both runs. v3 is needed because v2 did not contain oversampling results, and existing checkpoints cannot produce a method that was not executed.

## Run configuration

| Setting | Value |
|---|---|
| Datasets | 15 OpenML imbalanced binary datasets |
| Models | HGB, LogReg, RF, ExtraTrees, GB, AdaBoost, GNB |
| Calibration | None, sigmoid, isotonic |
| Seeds | 7, 19, 31, 42, 101, 202, 303, 404, 505, 606 |
| Target alpha | 0.10 |
| Cost matrix | `C_FP=1`, `C_FN=10` |
| Review capacity | 10% |
| Deferral costs | 0, 0.5, 1, 2 |
| Parallel jobs | 4 |

Expected total: `15 × 7 × 3 × 10 = 3,150` checkpoint runs.

## Methods in each checkpoint

| Method | Role |
|---|---|
| `cost_tuned_threshold` | Cost-tuned point-prediction baseline |
| `class_weighted_threshold` | Retrained classifier with balanced sample weights, then cost-tuned threshold |
| `random_oversampling_threshold` | Retrained classifier after random minority oversampling, then cost-tuned threshold |
| `smote_threshold` | Retrained classifier after minority interpolation, then cost-tuned threshold |
| `marginal_cp` | Global split-conformal quantile |
| `mondrian_cp` | Class-conditional split-conformal quantiles |
| `aps_cp` | Adaptive Prediction Sets |
| `raps_cp` | Regularized Adaptive Prediction Sets |
| `confidence_rejector` | Probability-confidence rejection |
| `risk_controlled_rejector` | Risk-targeted rejection threshold |
| `class_conditional_rejector` | Separate rejection thresholds by class |
| `capacity_limited_confidence_rejector` | Confidence rejection capped at review budget |
| `capacity_limited_risk_controlled_rejector` | Risk-controlled rejection capped at review budget |
| `capacity_limited_class_conditional_rejector` | Class-conditional rejection capped at review budget |
| `capacity_limited_aps` | APS deferral capped at review budget |
| `capacity_limited_raps` | RAPS deferral capped at review budget |
| `capacity_limited_mondrian` | Mondrian deferral capped at review budget |
| `cost_controlled_mondrian` | Proposed cost-tuned Mondrian method |

## v2 versus v3

| Change | v2 | v3 |
|---|---|---|
| Oversampling baseline | Missing | Added |
| Other method definitions | Original set | Same |
| Benchmark dimensions | 15 × 7 × 3 × 10 | Same |
| Outputs | Full CSV summaries and figures | Generated after checkpoints complete |
| Status | Complete prior run | Still populating |

## Why oversampling matters

Class weighting changes training loss through sample weights. Random oversampling changes training data composition by duplicating minority examples. They test different imbalance remedies. Neither is SMOTE, focal loss, a cost-sensitive ensemble, or learning-to-defer.

## Not covered yet

v3 still does not provide:

- dedicated cost-sensitive ensemble;
- learning-to-defer baseline;
- focal-loss classifier (outside scope);

Do not claim v3 resolves all baseline concerns until these methods are added or the scope is explicitly narrowed.

## Completion check

Run:

```bash
find results_v3/checkpoints -name '*.pkl' | wc -l
```

Expected final count: `3150`. Summary CSVs and figures should be present only after benchmark aggregation completes.

Manifest: [`run_manifest.json`](run_manifest.json).
