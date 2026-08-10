# Cost-Sensitive Conformal Prediction for Imbalanced High-Stakes Decisions

A leakage-safe benchmark of conformal prediction methods for imbalanced,
cost-asymmetric binary decisions (credit, fraud, healthcare, industrial, etc.),
with a cost-controlled abstention rule.

**Core finding.** The benchmark confirms that marginal conformal prediction can
meet its overall coverage target while under-covering rare, costly classes
(averaging only 30.8% minority coverage across 3,150 experimental fits).
Class-conditional (Mondrian) conformal prediction restores class-wise coverage
(92.1% minority coverage) under its exchangeability assumptions. The paper's main contribution is the
operational analysis: asymmetric costs, review capacity, reviewer error, and
dataset-specific break-even thresholds.

## Overall Benchmark Summary (`results_v4`)

Aggregated across 15 imbalanced datasets, 7 base models, 3 probability calibrations, and 10 random seeds ($N=3,150$ total evaluations per method, target error $\alpha=0.10 \implies 90\%$ target coverage):

| Method | Marginal Coverage | Minority Coverage ($C_1$) | Average Set Size | Abstention Rate | Mean Cost |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `cost_controlled_mondrian` | **98.8%** | **96.6%** | 1.67 | 67.0% | **0.0312** |
| `mondrian_cp` | **91.8%** | **92.1%** | 1.37 | 39.2% | **0.1418** |
| `class_weighted_threshold` | 77.9% | 80.0% | 1.00 | 0.0% | 0.3435 |
| `cost_tuned_threshold` | 77.9% | 79.4% | 1.00 | 0.0% | 0.3442 |
| `random_oversampling_threshold` | 77.8% | 79.9% | 1.00 | 0.0% | 0.3442 |
| `smote_threshold` | 77.2% | 80.4% | 1.00 | 0.0% | 0.3478 |
| `aps_cp` | 90.2% | 55.5% | 1.09 | 19.7% | 0.3823 |
| `raps_cp` | 90.2% | 48.7% | 1.07 | 15.7% | 0.4445 |
| `marginal_cp` | 91.2% | **30.8%** | 1.05 | 12.8% | 0.4076 |
| `class_conditional_rejector` | 94.9% | 47.3% | 1.13 | 13.3% | 0.4199 |
| `risk_controlled_rejector` | 93.6% | 44.7% | 1.08 | 8.2% | 0.4910 |
| `capacity_limited_mondrian` | 88.2% | 73.3% | 1.06 | 8.5% | 0.4099 |
| `confidence_rejector` | 99.1% | 97.1% | 1.77 | 76.9% | 0.0261 |

## Repository structure

```
.
├── src/cost_conformal/        # the engineered package (importable, tested)
│   ├── conformal.py           # marginal + Mondrian conformal predictors
│   ├── data.py                # dataset loaders (OpenML + CSV)
│   ├── evaluation.py          # cost matrix, coverage/cost metrics
│   └── plots.py               # figures
├── scripts/
│   └── run_benchmark.py       # CLI benchmark runner + deferral-cost analysis
├── tests/test_smoke.py        # smoke test
├── requirements.txt
└── pyproject.toml
```

## Quick start

The benchmark CLI downloads 15 imbalanced datasets from OpenML
automatically and regenerates the tables and figures.

```bash
pip install -e .
python scripts/run_benchmark.py \
  --models hgb logreg rf et gb ada gnb \
  --probability-calibrations none sigmoid isotonic \
  --output-dir results_v4
```

Runs on CPU with 10 seeds by default across all available cores (`--jobs -1`). Results are saved to the output directory
as CSV tables and PNG figures, including 95% confidence intervals and paired
bootstrap deferral-cost sensitivity. Each dataset/seed writes a checkpoint under
`<output-dir>/checkpoints`, so rerunning the same command resumes completed
pairs. Use `--no-resume` to ignore checkpoints.

## Full Benchmark Grid (IEEE OJCS Submission)

The full evaluation grid runs:

```text
15 datasets * 7 models * 3 probability calibrations * 10 seeds = 3150 runs
```

Start or resume the full grid:

```bash
python scripts/run_benchmark.py \
  --models hgb logreg rf et gb ada gnb \
  --probability-calibrations none sigmoid isotonic \
  --output-dir results_v4 \
  --jobs -1
```

Resume works automatically from `results_v4/checkpoints`. Do not pass
`--no-resume` unless you want to recompute the grid.

## Key Implementation Features

- **Randomized APS/RAPS Nonconformity Scores**: Conformal scoring implements uniform randomization ($u \sim U(0,1)$) per Romano et al. (2020), preventing quantile threshold saturation ($\hat{q} \ge 1.0$) and eliminating set-size degeneration under severe class imbalance.
- **Parallel Runner**: Multi-core parallel execution via `joblib.Parallel` over dataset/model/calibration cells.

## Using the package (engineered version)

```bash
pip install -e .
python scripts/run_benchmark.py \
  --model hgb --output-dir results_v4
pytest -q
```

## Datasets

Fifteen imbalanced datasets, auto-loaded from OpenML:

- Finance/fraud/credit: `uci_default`, `bank_marketing`, `adult`, `fraud`,
  `credit_g`
- Healthcare/science/industrial: `aps_failure`, `diabetes130us`, `miniboone`,
  `mammography`, `sick_numeric`, `ozone_level_8hr`, `seismic_bumps`
- General imbalanced tabular benchmarks: `wilt`, `pc1`, `oil_spill`

The Kaggle 2023 balanced credit-card dataset is intentionally not included.

Main output tables:

- `run_manifest.json`: arguments, package versions, git commit, datasets, seeds
- `dataset_profile.csv`: sample sizes, feature counts, positive rates, split counts
- `method_table.csv`: per-dataset, per-seed metrics
- `method_summary.csv`: per-dataset method means with 95% CIs across seeds
- `method_stat_tests.csv`: paired seed-level comparisons against Mondrian CP
- `cross_dataset_stat_tests.csv`: dataset-level paired tests and ranks
- `model_summary.csv`: model/method/calibration aggregate ranks and metrics
- `frontier.csv`: cost-vs-abstention frontier points
- `deferral_table.csv`: per-seed paired-bootstrap cost comparisons
- `deferral_sensitivity.csv`: Mondrian-vs-threshold cost difference with 95% CI
  and paired-bootstrap probability that Mondrian is cheaper
- `credit_default_case.csv`: focused credit-default summary for `uci_default`
  and `credit_g`
- `credit_default_scenarios.csv`: credit-default cost scenarios over
  false-negative and review-cost settings
- `ablation_summary.csv`: alpha-frontier and deferral-cost ablations
- `failure_analysis.csv`: high-abstention, low-review-tolerance, and
  undercoverage flags

Compared methods:

- Conformal: `marginal_cp`, `mondrian_cp`, `aps_cp`, `raps_cp`,
  `cost_controlled_mondrian`
- Decision baselines: `cost_tuned_threshold`, `class_weighted_threshold`,
  `random_oversampling_threshold`, `smote_threshold`
- Reject/deferral baselines: `confidence_rejector`,
  `risk_controlled_rejector`, `class_conditional_rejector`,
  `capacity_limited_confidence_rejector`,
  `capacity_limited_risk_controlled_rejector`,
  `capacity_limited_class_conditional_rejector`, `capacity_limited_mondrian`
- Budget-matched conformal baselines: `capacity_limited_aps`,
  `capacity_limited_raps`

Focal-loss classifiers are outside scope. Benchmark focuses on thresholding,
resampling, conformal prediction, and review-budget policies.

Sensitivity grid:

```bash
python scripts/run_sensitivity.py --output-dir sensitivity
```

This runs `C_FP ∈ {0.5, 1, 2}`, `C_FN ∈ {5, 10, 20, 50}`, review capacity
`{0, 0.05, 0.10, 0.20, 0.30}`, and review cost `{0, 0.25, 0.5, 1, 2}`.

Base models:

- `hgb`, `logreg`, `rf`, `et`, `gb`, `ada`, `gnb`

Probability calibration:

- `none`, `sigmoid`, `isotonic`


## Notes for collaborators

- Keep large data files out of git (see `.gitignore`); download datasets via the
  loaders instead.
- Results folders and LaTeX build artifacts are git-ignored.
- Choose a license before making the repo public (an MIT template is included in
  `LICENSE` — edit or replace it).

## Citation

If you use this repository, benchmark, or code in your research, please cite our manuscript:

```bibtex
@article{singh2026cost,
  title={Cost-Sensitive Conformal Prediction and Human-in-the-Loop Abstention for Imbalanced High-Stakes Decision Support: A Multi-Domain Benchmark},
  author={Singh, Manpreet and Srikantha, Akshatha and Parashar, Deepak and Joshi, Rahul},
  journal={IEEE Open Journal of the Computer Society},
  note={Submitted},
  year={2026}
}
```

**APA Format:**
> Singh, M., Srikantha, A., Parashar, D., & Joshi, R. (2026). Cost-Sensitive Conformal Prediction and Human-in-the-Loop Abstention for Imbalanced High-Stakes Decision Support: A Multi-Domain Benchmark. *IEEE Open Journal of the Computer Society* (Submitted).

