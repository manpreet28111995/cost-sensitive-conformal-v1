#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import platform
import subprocess
import sys

import numpy as np
import pandas as pd
import sklearn
from joblib import Parallel, delayed
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))
os.environ.setdefault("SCIKIT_LEARN_DATA", str(ROOT / ".cache" / "scikit_learn_data"))
(ROOT / ".matplotlib-cache").mkdir(exist_ok=True)
(ROOT / ".cache").mkdir(exist_ok=True)

from cost_conformal.conformal import AdaptiveConformalPredictor, ConformalPredictor
from cost_conformal.data import (
    OPENML_DATASETS,
    Dataset,
    load_builtin_dataset,
    load_csv_dataset,
)
from cost_conformal.evaluation import (
    CostMatrix,
    evaluate_hard_predictions,
    evaluate_prediction_mask,
    hard_prediction_costs,
    prediction_mask_costs,
    tune_cost_threshold,
)
from cost_conformal.plots import (
    plot_cost_abstention_frontier,
    plot_coverage_bars,
    plot_deferral_sensitivity,
    plot_minority_coverage_by_dataset,
)

MODEL_CHOICES = ("hgb", "logreg", "rf", "et", "gb", "ada", "gnb")
CALIBRATION_CHOICES = ("none", "sigmoid", "isotonic")
CREDIT_DATASETS = {"uci_default", "credit_g"}
CREDIT_FN_COSTS = (5.0, 10.0, 20.0)
CREDIT_DEFER_COSTS = (0.0, 0.25, 0.5, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark cost-sensitive conformal prediction under imbalance."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(OPENML_DATASETS),
        help="Datasets to run. Default: all 15 OpenML benchmarks.",
    )
    parser.add_argument("--csv", type=Path, help="Optional CSV dataset path.")
    parser.add_argument("--target", help="Target column for --csv.")
    parser.add_argument("--positive-label", default="1", help="Positive label for --csv.")
    parser.add_argument("--dataset-name", help="Name to use for the CSV dataset.")
    parser.add_argument("--alpha", type=float, default=0.10, help="Target error level.")
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[7, 19, 31, 42, 101, 202, 303, 404, 505, 606],
    )
    parser.add_argument(
        "--model",
        choices=MODEL_CHOICES,
        default="hgb",
        help="Base classifier kept for backward compatibility.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_CHOICES,
        help="Run multiple base classifiers. Overrides --model.",
    )
    parser.add_argument(
        "--probability-calibration",
        choices=CALIBRATION_CHOICES,
        default="none",
        help="Probability calibration kept for backward compatibility.",
    )
    parser.add_argument(
        "--probability-calibrations",
        nargs="+",
        choices=CALIBRATION_CHOICES,
        help="Run multiple probability-calibration settings. Overrides --probability-calibration.",
    )
    parser.add_argument("--fp-cost", type=float, default=1.0)
    parser.add_argument("--fn-cost", type=float, default=10.0)
    parser.add_argument("--defer-cost", type=float, default=0.0)
    parser.add_argument("--defer-costs", nargs="+", type=float, default=[0.0, 0.5, 1.0, 2.0])
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--raps-penalty", type=float, default=0.1)
    parser.add_argument("--review-capacity", type=float, default=0.10)
    parser.add_argument("--frontier-alpha-min", type=float, default=0.01)
    parser.add_argument("--frontier-alpha-max", type=float, default=0.45)
    parser.add_argument("--frontier-steps", type=int, default=23)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--jobs", type=int, default=-1,
        help="Parallel worker processes over (dataset,seed,model,calibration) cells. "
             "-1 = all cores (default), 1 = serial.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_false",
        dest="resume",
        help="Ignore existing per-dataset/seed checkpoints.",
    )
    return parser.parse_args()


def _process_cell(dataset, seed, model_name, calibration_name, args, costs, checkpoint_dir):
    """Run (or load from checkpoint) a single (dataset, seed, model, calibration) cell.
    Safe to call from parallel workers: each cell is independent and self-checkpoints."""
    checkpoint = checkpoint_path(
        checkpoint_dir, dataset.name, seed, model_name, calibration_name, args
    )
    if args.resume and checkpoint.exists():
        print(
            f"Skipping {dataset.name} seed={seed} model={model_name} "
            f"calibration={calibration_name}; checkpoint exists.",
            flush=True,
        )
        return load_checkpoint(checkpoint)
    print(
        f"Running {dataset.name} seed={seed} model={model_name} "
        f"calibration={calibration_name}...",
        flush=True,
    )
    result = run_one_dataset(
        dataset, seed, args.alpha, costs,
        defer_costs=args.defer_costs,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        raps_penalty=args.raps_penalty,
        review_capacity=args.review_capacity,
        model_name=model_name,
        probability_calibration=calibration_name,
        frontier_alpha_min=args.frontier_alpha_min,
        frontier_alpha_max=args.frontier_alpha_max,
        frontier_steps=args.frontier_steps,
    )
    save_checkpoint(checkpoint, result)
    print(f"Saved checkpoint: {checkpoint.name}", flush=True)
    return result


def main() -> None:
    args = parse_args()
    if not 0.0 < args.alpha < 1.0:
        raise SystemExit("--alpha must be in (0, 1)")
    if not 0.0 < args.frontier_alpha_min < args.frontier_alpha_max < 1.0:
        raise SystemExit(
            "--frontier-alpha-min and --frontier-alpha-max must satisfy "
            "0 < min < max < 1"
        )
    if args.frontier_steps < 2:
        raise SystemExit("--frontier-steps must be at least 2")
    if not 0.0 <= args.review_capacity <= 1.0:
        raise SystemExit("--review-capacity must be in [0, 1]")

    print(
        f"Benchmark start: {len(args.datasets) + int(bool(args.csv))} datasets, "
        f"{len(args.seeds)} seeds, models={args.models or [args.model]}, "
        f"calibrations={args.probability_calibrations or [args.probability_calibration]}, "
        f"output={args.output_dir}",
        flush=True,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    costs = CostMatrix(
        false_positive=args.fp_cost,
        false_negative=args.fn_cost,
        defer=args.defer_cost,
    )

    datasets: list[Dataset] = [
        load_builtin_dataset(name, args.seeds[0]) for name in args.datasets
    ]
    if args.csv:
        if not args.target:
            raise SystemExit("--target is required when --csv is provided")
        datasets.append(
            load_csv_dataset(
                args.csv,
                target=args.target,
                positive_label=args.positive_label,
                dataset_name=args.dataset_name,
            )
        )
    write_manifest(args.output_dir / "run_manifest.json", args, datasets)
    dataset_profile = profile_datasets(
        datasets,
        args.seeds,
        args.probability_calibrations or [args.probability_calibration],
    )

    method_rows: list[dict[str, float | str | int]] = []
    frontier_rows: list[dict[str, float | str | int]] = []
    deferral_rows: list[dict[str, float | str | int | bool]] = []
    credit_scenario_rows: list[dict[str, float | str | int]] = []

    jobs = [
        (dataset, seed, model_name, calibration_name)
        for dataset in datasets
        for seed in args.seeds
        for model_name in (args.models or [args.model])
        for calibration_name in (args.probability_calibrations or [args.probability_calibration])
    ]
    print(
        f"Dispatching {len(jobs)} cells over jobs={args.jobs} "
        f"(-1 = all cores). Each cell self-checkpoints.",
        flush=True,
    )
    # Threads avoid macOS loky semaphore limits. Caller should cap BLAS/OpenMP
    # threads when using jobs > 1.
    results = Parallel(n_jobs=args.jobs, prefer="threads", verbose=10)(
        delayed(_process_cell)(
            dataset, seed, model_name, calibration_name, args, costs, checkpoint_dir
        )
        for (dataset, seed, model_name, calibration_name) in jobs
    )
    for result in results:
        method_rows.extend(result["method_rows"])
        frontier_rows.extend(result["frontier_rows"])
        deferral_rows.extend(result["deferral_rows"])
        credit_scenario_rows.extend(result["credit_scenario_rows"])

    print("Building summary tables...", flush=True)
    method_table = pd.DataFrame(method_rows)
    frontier = pd.DataFrame(frontier_rows)
    deferral = pd.DataFrame(deferral_rows)
    method_summary = summarize_methods(method_table)
    deferral_summary = summarize_deferral(deferral)
    stats = paired_method_tests(method_table, baseline="mondrian_cp")
    cross_stats = cross_dataset_tests(method_summary, baseline="mondrian_cp")
    credit_case = credit_default_case(method_summary, deferral_summary)
    credit_scenarios = pd.DataFrame(credit_scenario_rows)
    ablations = summarize_ablations(frontier, deferral_summary)
    failures = failure_analysis(method_summary, deferral_summary)
    model_summary = summarize_models(method_summary)

    print("Writing CSV outputs...", flush=True)
    dataset_profile.to_csv(args.output_dir / "dataset_profile.csv", index=False)
    method_table.to_csv(args.output_dir / "method_table.csv", index=False)
    method_summary.to_csv(args.output_dir / "method_summary.csv", index=False)
    frontier.to_csv(args.output_dir / "frontier.csv", index=False)
    deferral.to_csv(args.output_dir / "deferral_table.csv", index=False)
    deferral_summary.to_csv(args.output_dir / "deferral_sensitivity.csv", index=False)
    stats.to_csv(args.output_dir / "method_stat_tests.csv", index=False)
    cross_stats.to_csv(args.output_dir / "cross_dataset_stat_tests.csv", index=False)
    model_summary.to_csv(args.output_dir / "model_summary.csv", index=False)
    credit_case.to_csv(args.output_dir / "credit_default_case.csv", index=False)
    credit_scenarios.to_csv(args.output_dir / "credit_default_scenarios.csv", index=False)
    ablations.to_csv(args.output_dir / "ablation_summary.csv", index=False)
    failures.to_csv(args.output_dir / "failure_analysis.csv", index=False)

    # Post-processing: compute breakeven thresholds, noisy human review, and budget constraints
    breakeven_summary = compute_breakeven_thresholds(deferral)
    breakeven_summary.to_csv(args.output_dir / "breakeven_thresholds.csv", index=False)

    noisy_df = compute_noisy_human_review(method_table)
    noisy_df.to_csv(args.output_dir / "noisy_human_review.csv", index=False)
    noisy_summary = noisy_df.groupby(["eps_hum", "c_rev"])[["total_cost", "abstention_rate"]].mean().reset_index()
    noisy_summary.to_csv(args.output_dir / "noisy_human_review_summary.csv", index=False)

    budget_summary = compute_budget_constrained(method_table)
    budget_summary.to_csv(args.output_dir / "budget_constrained_summary.csv", index=False)

    # Fixed-policy cost must be non-decreasing as review cost rises.
    validate_deferral_monotonicity(deferral_summary)

    print("Rendering plots...", flush=True)
    plot_coverage_bars(
        method_table,
        args.output_dir / "coverage_bars.png",
        target_coverage=1.0 - args.alpha,
    )
    plot_cost_abstention_frontier(
        frontier,
        args.output_dir / "cost_abstention_frontier.png",
        method_summary=method_summary,
    )
    plot_minority_coverage_by_dataset(
        method_table,
        args.output_dir / "minority_coverage_by_dataset.png",
        target_coverage=1.0 - args.alpha,
    )
    plot_deferral_sensitivity(
        deferral_summary, args.output_dir / "deferral_sensitivity.png"
    )

    # Sync figures to KBS-CSC/figs/ if present
    kbs_figs_dir = ROOT / "KBS-CSC" / "figs"
    if kbs_figs_dir.exists():
        import shutil
        for fig_name in ["coverage_bars.png", "cost_abstention_frontier.png", "minority_coverage_by_dataset.png", "deferral_sensitivity.png"]:
            src_fig = args.output_dir / fig_name
            if src_fig.exists():
                shutil.copy(src_fig, kbs_figs_dir / fig_name)

    summary_cols = [
        "method",
        "model",
        "probability_calibration",
        "marginal_coverage",
        "coverage_class_0",
        "coverage_class_1",
        "average_set_size",
        "abstention_rate",
        "mean_cost",
    ]
    summary = method_table.groupby(["model", "probability_calibration", "method"])[
        summary_cols[3:]
    ].mean().reset_index()
    print(summary[summary_cols].round(4).to_string(index=False))
    print(f"\nWrote outputs to {args.output_dir.resolve()}")


def compute_breakeven_thresholds(deferral_df: pd.DataFrame) -> pd.DataFrame:
    """Compute dataset-specific break-even review threshold C_rev* for each dataset."""
    df_zero = deferral_df[deferral_df["defer_cost"] == 0.0].copy()
    df_zero = df_zero[df_zero["mondrian_abstain"] > 0.001].copy()
    df_zero["c_rev_star"] = (
        df_zero["threshold_cost"] - df_zero["mondrian_cost"]
    ) / df_zero["mondrian_abstain"]
    
    grouped = df_zero.groupby("dataset")["c_rev_star"]
    summary = grouped.agg(
        c_rev_star_mean="mean",
        c_rev_star_std="std",
        c_rev_star_median="median",
        c_rev_star_q25=lambda x: x.quantile(0.25),
        c_rev_star_q75=lambda x: x.quantile(0.75),
        n_samples="count"
    ).reset_index()
    
    summary["c_rev_star_ci_lo"] = summary["c_rev_star_mean"] - 1.96 * (summary["c_rev_star_std"] / np.sqrt(summary["n_samples"]))
    summary["c_rev_star_ci_hi"] = summary["c_rev_star_mean"] + 1.96 * (summary["c_rev_star_std"] / np.sqrt(summary["n_samples"]))
    return summary.sort_values("c_rev_star_mean", ascending=False)


def compute_noisy_human_review(method_df: pd.DataFrame, eps_values: list[float] = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20]) -> pd.DataFrame:
    """Evaluate decision cost under noisy human review error rates (eps_hum)."""
    rows = []
    mondrian = method_df[method_df["method"] == "mondrian_cp"].copy()
    c_err_avg = 0.9 * 1.0 + 0.1 * 10.0  # = 1.9
    
    for _, row in mondrian.iterrows():
        base_cost = row["mean_cost"]
        abstain = row["abstention_rate"]
        for eps in eps_values:
            for c_rev in [0.0, 0.25, 0.5, 1.0, 2.0]:
                total_cost = base_cost + abstain * c_rev + abstain * eps * c_err_avg
                rows.append({
                    "dataset": row["dataset"],
                    "seed": row["seed"],
                    "model": row["model"],
                    "probability_calibration": row["probability_calibration"],
                    "method": "mondrian_cp",
                    "eps_hum": eps,
                    "c_rev": c_rev,
                    "abstention_rate": abstain,
                    "base_cost": base_cost,
                    "total_cost": total_cost,
                })
    return pd.DataFrame(rows)


def validate_deferral_monotonicity(deferral_df: pd.DataFrame) -> None:
    """Reject impossible decreasing cost curves before figure generation."""
    required = {"dataset", "defer_cost", "mondrian_cost", "threshold_cost"}
    missing = required.difference(deferral_df.columns)
    if missing:
        raise ValueError(f"deferral output missing columns: {sorted(missing)}")
    policy_keys = ["dataset"]
    for key in ("model", "probability_calibration"):
        if key in deferral_df.columns:
            policy_keys.append(key)
    for policy, group in deferral_df.groupby(policy_keys, dropna=False):
        ordered = group.sort_values("defer_cost")
        for column in ("mondrian_cost", "threshold_cost"):
            values = ordered[column].to_numpy(dtype=float)
            if np.any(np.diff(values) < -1e-10):
                raise ValueError(
                    f"{column} decreases with review cost for policy={policy}; "
                    "check evaluation before plotting"
                )


def compute_budget_constrained(method_df: pd.DataFrame) -> pd.DataFrame:
    """Compute performance under strict review capacity constraints (r_abs <= 5%, 10%, 20%)."""
    capacity_rows = method_df[method_df["method"].isin(["mondrian_cp", "capacity_limited_mondrian", "cost_controlled_mondrian", "cost_tuned_threshold"])].copy()
    return capacity_rows.groupby(["method", "review_capacity"])[["marginal_coverage", "coverage_class_1", "abstention_rate", "mean_cost"]].mean().reset_index()


def write_manifest(path: Path, args: argparse.Namespace, datasets: list[Dataset]) -> None:
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sklearn": sklearn.__version__,
        },
        "args": vars(args) | {
            "csv": str(args.csv) if args.csv else None,
            "output_dir": str(args.output_dir),
        },
        "datasets": [
            {
                "name": dataset.name,
                "n_samples": int(dataset.X.shape[0]),
                "n_features": int(dataset.X.shape[1]),
                "positive_rate": float(np.mean(dataset.y)),
            }
            for dataset in datasets
        ],
    }
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def profile_datasets(
    datasets: list[Dataset],
    seeds: list[int],
    calibrations: list[str],
) -> pd.DataFrame:
    rows = []
    for dataset in datasets:
        for seed in seeds:
            X_train, X_tmp, y_train, y_tmp = train_test_split(
                dataset.X,
                dataset.y,
                test_size=0.40,
                stratify=dataset.y,
                random_state=seed,
            )
            X_cal, X_test, y_cal, y_test = train_test_split(
                X_tmp,
                y_tmp,
                test_size=0.50,
                stratify=y_tmp,
                random_state=seed + 1,
            )
            for calibration in calibrations:
                if calibration == "none":
                    y_model_train = y_train
                    y_proba_cal = np.array([], dtype=int)
                else:
                    _, _, y_model_train, y_proba_cal = train_test_split(
                        X_train,
                        y_train,
                        test_size=0.20,
                        stratify=y_train,
                        random_state=seed + 2,
                    )
                rows.append(
                    {
                        "dataset": dataset.name,
                        "seed": seed,
                        "probability_calibration": calibration,
                        "n_samples": int(dataset.X.shape[0]),
                        "n_features": int(dataset.X.shape[1]),
                        "positive_rate": float(np.mean(dataset.y)),
                        **split_profile("model_train", y_model_train),
                        **split_profile("probability_calibration", y_proba_cal),
                        **split_profile("conformal_calibration", y_cal),
                        **split_profile("test", y_test),
                    }
                )
    return pd.DataFrame(rows)


def split_profile(prefix: str, y: np.ndarray) -> dict[str, float | int]:
    y = np.asarray(y, dtype=int)
    return {
        f"{prefix}_n": int(y.size),
        f"{prefix}_class_0": int(np.sum(y == 0)),
        f"{prefix}_class_1": int(np.sum(y == 1)),
        f"{prefix}_positive_rate": float(np.mean(y)) if y.size else float("nan"),
    }


def checkpoint_path(
    checkpoint_dir: Path,
    dataset_name: str,
    seed: int,
    model_name: str,
    calibration_name: str,
    args: argparse.Namespace,
) -> Path:
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in dataset_name)
    key = checkpoint_key(dataset_name, seed, model_name, calibration_name, args)
    return checkpoint_dir / f"{safe_name}_{model_name}_{calibration_name}_seed{seed}_{key}.pkl"


def checkpoint_key(
    dataset_name: str,
    seed: int,
    model_name: str,
    calibration_name: str,
    args: argparse.Namespace,
) -> str:
    payload = {
        "dataset": dataset_name,
        "seed": seed,
        "csv": str(args.csv) if args.csv else None,
        "target": args.target,
        "positive_label": args.positive_label,
        "dataset_name": args.dataset_name,
        "alpha": args.alpha,
        "model": model_name,
        "probability_calibration": calibration_name,
        "fp_cost": args.fp_cost,
        "fn_cost": args.fn_cost,
        "defer_cost": args.defer_cost,
        "defer_costs": args.defer_costs,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
        "raps_penalty": args.raps_penalty,
        "review_capacity": args.review_capacity,
        "frontier_alpha_min": args.frontier_alpha_min,
        "frontier_alpha_max": args.frontier_alpha_max,
        "frontier_steps": args.frontier_steps,
    }
    raw = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha1(raw).hexdigest()[:12]


def load_checkpoint(path: Path) -> dict[str, list[dict[str, float | str | int | bool]]]:
    with path.open("rb") as handle:
        return pickle.load(handle)


def save_checkpoint(
    path: Path,
    result: dict[str, list[dict[str, float | str | int | bool]]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    with tmp_path.open("wb") as handle:
        pickle.dump(result, handle)
    tmp_path.replace(path)


def log_step(dataset_name: str, seed: int, message: str) -> None:
    print(f"[{dataset_name} seed={seed}] {message}", flush=True)


def run_one_dataset(
    dataset: Dataset,
    seed: int,
    alpha: float,
    costs: CostMatrix,
    defer_costs: list[float],
    bootstrap_samples: int,
    bootstrap_seed: int,
    raps_penalty: float,
    review_capacity: float,
    model_name: str,
    probability_calibration: str,
    frontier_alpha_min: float,
    frontier_alpha_max: float,
    frontier_steps: int,
) -> dict[str, list[dict[str, float | str | int | bool]]]:
    log_step(dataset.name, seed, "Splitting train/calibration/test data")
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        dataset.X,
        dataset.y,
        test_size=0.40,
        stratify=dataset.y,
        random_state=seed,
    )
    X_cal, X_test, y_cal, y_test = train_test_split(
        X_tmp,
        y_tmp,
        test_size=0.50,
        stratify=y_tmp,
        random_state=seed + 1,
    )
    # Keep alpha selection data independent from final conformal calibration.
    X_select, X_cal_final, y_select, y_cal_final = train_test_split(
        X_cal,
        y_cal,
        test_size=0.50,
        stratify=y_cal,
        random_state=seed + 3,
    )

    if probability_calibration == "none":
        X_model_train, y_model_train = X_train, y_train
        X_proba_cal, y_proba_cal = None, None
    else:
        X_model_train, X_proba_cal, y_model_train, y_proba_cal = train_test_split(
            X_train,
            y_train,
            test_size=0.20,
            stratify=y_train,
            random_state=seed + 2,
        )

    log_step(dataset.name, seed, f"Fitting base {model_name} model")
    model, _ = make_model(model_name, seed)
    model.fit(X_model_train, y_model_train)

    log_step(dataset.name, seed, "Predicting calibration/test probabilities")
    select_proba = calibrate_proba(
        model,
        X_proba_cal,
        y_proba_cal,
        X_select,
        probability_calibration,
    )
    cal_proba = calibrate_proba(
        model,
        X_proba_cal,
        y_proba_cal,
        X_cal_final,
        probability_calibration,
    )
    test_proba = calibrate_proba(
        model,
        X_proba_cal,
        y_proba_cal,
        X_test,
        probability_calibration,
    )

    method_rows = []
    mondrian_mask = None
    common = {
        "dataset": dataset.name,
        "seed": seed,
        "model": model_name,
        "probability_calibration": probability_calibration,
    }
    for method_name, mode in [
        ("marginal_cp", "marginal"),
        ("mondrian_cp", "mondrian"),
    ]:
        log_step(dataset.name, seed, f"Evaluating {method_name}")
        cp = ConformalPredictor.fit(cal_proba, y_cal_final, alpha=alpha, mode=mode)
        prediction_mask = cp.predict_mask(test_proba)
        if mode == "mondrian":
            mondrian_mask = prediction_mask
        row = evaluate_prediction_mask(y_test, prediction_mask, costs)
        row.update(common | {"method": method_name, "alpha": alpha})
        method_rows.append(row)

    for method_name, penalty in [
        ("aps_cp", 0.0),
        ("raps_cp", raps_penalty),
    ]:
        log_step(dataset.name, seed, f"Evaluating {method_name}")
        cp = AdaptiveConformalPredictor.fit(
            cal_proba, y_cal_final, alpha=alpha, penalty=penalty, seed=seed
        )
        adaptive_mask = cp.predict_mask(test_proba)
        row = evaluate_prediction_mask(y_test, adaptive_mask, costs)
        row.update(common | {"method": method_name, "alpha": alpha})
        method_rows.append(row)
        if method_name == "aps_cp":
            aps_mask = adaptive_mask
        else:
            raps_mask = adaptive_mask

    log_step(dataset.name, seed, "Tuning cost-controlled Mondrian alpha")
    tuned_alpha = tune_cost_controlled_mondrian_alpha(
        select_proba,
        y_select,
        costs,
        target_coverage=1.0 - alpha,
        alpha_grid=np.linspace(frontier_alpha_min, frontier_alpha_max, frontier_steps),
        seed=seed,
    )
    tuned_cp = ConformalPredictor.fit(
        cal_proba, y_cal_final, alpha=tuned_alpha, mode="mondrian"
    )
    tuned_row = evaluate_prediction_mask(y_test, tuned_cp.predict_mask(test_proba), costs)
    tuned_row.update(
        {
            "method": "cost_controlled_mondrian",
            "alpha": tuned_alpha,
        } | common
    )
    method_rows.append(tuned_row)

    log_step(dataset.name, seed, "Tuning cost threshold baseline")
    threshold = tune_cost_threshold(y_cal_final, cal_proba[:, 1], costs)
    threshold_pred = (test_proba[:, 1] >= threshold).astype(int)
    threshold_row = evaluate_hard_predictions(y_test, threshold_pred, costs)
    threshold_row.update(
        {
            "method": "cost_tuned_threshold",
            "alpha": np.nan,
            "threshold": threshold,
        } | common
    )
    method_rows.append(threshold_row)

    log_step(dataset.name, seed, "Fitting class-weighted baseline")
    weighted_model, weighted_final_step = make_model(model_name, seed)
    sample_weight = balanced_sample_weight(y_model_train)
    weighted_model.fit(
        X_model_train,
        y_model_train,
        **{f"{weighted_final_step}__sample_weight": sample_weight},
    )
    weighted_cal_proba = calibrate_proba(
        weighted_model,
        X_proba_cal,
        y_proba_cal,
        X_cal_final,
        probability_calibration,
    )
    weighted_test_proba = calibrate_proba(
        weighted_model,
        X_proba_cal,
        y_proba_cal,
        X_test,
        probability_calibration,
    )
    weighted_threshold = tune_cost_threshold(y_cal_final, weighted_cal_proba[:, 1], costs)
    weighted_pred = (weighted_test_proba[:, 1] >= weighted_threshold).astype(int)
    weighted_row = evaluate_hard_predictions(y_test, weighted_pred, costs)
    weighted_row.update(
        {
            "method": "class_weighted_threshold",
            "alpha": np.nan,
            "threshold": weighted_threshold,
        } | common
    )
    method_rows.append(weighted_row)

    log_step(dataset.name, seed, "Fitting random-oversampling baseline")
    over_model, over_final_step = make_model(model_name, seed)
    over_X, over_y = random_oversample(X_model_train, y_model_train, seed)
    over_model.fit(over_X, over_y)
    over_cal_proba = calibrate_proba(
        over_model,
        X_proba_cal,
        y_proba_cal,
        X_cal_final,
        probability_calibration,
    )
    over_test_proba = calibrate_proba(
        over_model,
        X_proba_cal,
        y_proba_cal,
        X_test,
        probability_calibration,
    )
    over_threshold = tune_cost_threshold(y_cal_final, over_cal_proba[:, 1], costs)
    over_pred = (over_test_proba[:, 1] >= over_threshold).astype(int)
    over_row = evaluate_hard_predictions(y_test, over_pred, costs)
    over_row.update(
        {
            "method": "random_oversampling_threshold",
            "alpha": np.nan,
            "threshold": over_threshold,
        } | common
    )
    method_rows.append(over_row)

    log_step(dataset.name, seed, "Fitting SMOTE baseline")
    smote_model, _ = make_model(model_name, seed)
    smote_X, smote_y = smote(X_model_train, y_model_train, seed)
    smote_model.fit(smote_X, smote_y)
    smote_cal_proba = calibrate_proba(
        smote_model, X_proba_cal, y_proba_cal, X_cal_final, probability_calibration
    )
    smote_test_proba = calibrate_proba(
        smote_model, X_proba_cal, y_proba_cal, X_test, probability_calibration
    )
    smote_threshold = tune_cost_threshold(y_cal_final, smote_cal_proba[:, 1], costs)
    smote_pred = (smote_test_proba[:, 1] >= smote_threshold).astype(int)
    smote_row = evaluate_hard_predictions(y_test, smote_pred, costs)
    smote_row.update(
        {
            "method": "smote_threshold",
            "alpha": np.nan,
            "threshold": smote_threshold,
        }
        | common
    )
    method_rows.append(smote_row)

    log_step(dataset.name, seed, "Tuning confidence rejector")
    confidence_tau = tune_confidence_tau(y_cal_final, cal_proba, costs)
    confidence_mask = confidence_reject_mask(test_proba, confidence_tau)
    confidence_row = evaluate_prediction_mask(y_test, confidence_mask, costs)
    confidence_row.update(
        {
            "method": "confidence_rejector",
            "alpha": np.nan,
            "threshold": confidence_tau,
        } | common
    )
    method_rows.append(confidence_row)

    log_step(dataset.name, seed, "Tuning risk-controlled rejector")
    risk_tau = tune_risk_controlled_tau(
        y_cal_final, cal_proba, target_coverage=1.0 - alpha
    )
    risk_row = evaluate_prediction_mask(y_test, confidence_reject_mask(test_proba, risk_tau), costs)
    risk_row.update(
        {
            "method": "risk_controlled_rejector",
            "alpha": alpha,
            "threshold": risk_tau,
        } | common
    )
    method_rows.append(risk_row)

    log_step(dataset.name, seed, "Tuning class-conditional rejector")
    class_taus = tune_class_conditional_tau(
        y_cal_final, cal_proba, target_coverage=1.0 - alpha
    )
    class_reject_row = evaluate_prediction_mask(
        y_test, class_conditional_reject_mask(test_proba, class_taus), costs
    )
    class_reject_row.update(
        {
            "method": "class_conditional_rejector",
            "alpha": alpha,
            "threshold": float(np.mean(class_taus)),
        } | common
    )
    method_rows.append(class_reject_row)

    # Match every selective baseline to the same review budget.
    for method_name, prediction_mask in [
        ("capacity_limited_confidence_rejector", confidence_mask),
        (
            "capacity_limited_risk_controlled_rejector",
            confidence_reject_mask(test_proba, risk_tau),
        ),
        (
            "capacity_limited_class_conditional_rejector",
            class_conditional_reject_mask(test_proba, class_taus),
        ),
        ("capacity_limited_aps", aps_mask),
        ("capacity_limited_raps", raps_mask),
    ]:
        budget_mask = capacity_limited_mask(
            prediction_mask, test_proba, max_defer_rate=review_capacity
        )
        budget_row = evaluate_prediction_mask(y_test, budget_mask, costs)
        budget_row.update(
            {"method": method_name, "alpha": alpha, "review_capacity": review_capacity}
            | common
        )
        method_rows.append(budget_row)

    if mondrian_mask is None:
        raise RuntimeError("Mondrian prediction mask was not computed")
    log_step(dataset.name, seed, "Applying capacity-limited Mondrian")
    capacity_mask = capacity_limited_mask(
        mondrian_mask,
        test_proba,
        max_defer_rate=review_capacity,
    )
    capacity_row = evaluate_prediction_mask(y_test, capacity_mask, costs)
    capacity_row.update(
        {
            "method": "capacity_limited_mondrian",
            "alpha": alpha,
            "review_capacity": review_capacity,
        } | common
    )
    method_rows.append(capacity_row)

    deferral_rows = []
    for defer_cost in defer_costs:
        log_step(dataset.name, seed, f"Bootstrapping deferral cost={defer_cost}")
        defer_cost_matrix = CostMatrix(
            false_positive=costs.false_positive,
            false_negative=costs.false_negative,
            defer=defer_cost,
        )
        mondrian_metrics = evaluate_prediction_mask(
            y_test, mondrian_mask, defer_cost_matrix
        )
        threshold_metrics = evaluate_hard_predictions(
            y_test, threshold_pred, defer_cost_matrix
        )
        mondrian_cost = prediction_mask_costs(y_test, mondrian_mask, defer_cost_matrix)
        threshold_cost = hard_prediction_costs(y_test, threshold_pred, defer_cost_matrix)
        boot = paired_bootstrap_diff(
            mondrian_cost - threshold_cost,
            samples=bootstrap_samples,
            seed=bootstrap_seed + seed,
        )
        deferral_rows.append(
            {
                **common,
                "defer_cost": defer_cost,
                "mondrian_cost": mondrian_metrics["mean_cost"],
                "mondrian_abstain": mondrian_metrics["abstention_rate"],
                "threshold_cost": threshold_metrics["mean_cost"],
                "cost_diff": boot["observed"],
                "cost_diff_ci_lo": boot["ci_lo"],
                "cost_diff_ci_hi": boot["ci_hi"],
                "p_mondrian_cheaper": boot["p_less_than_zero"],
                "mondrian_wins": mondrian_metrics["mean_cost"]
                < threshold_metrics["mean_cost"],
            }
        )

    frontier_rows = []
    log_step(dataset.name, seed, f"Building frontier ({frontier_steps} alpha steps)")
    alpha_grid = np.linspace(frontier_alpha_min, frontier_alpha_max, frontier_steps)
    for frontier_alpha in alpha_grid:
        for method_name, mode in [
            ("marginal_cp", "marginal"),
            ("mondrian_cp", "mondrian"),
        ]:
            cp = ConformalPredictor.fit(
                cal_proba, y_cal_final, alpha=float(frontier_alpha), mode=mode
            )
            row = evaluate_prediction_mask(y_test, cp.predict_mask(test_proba), costs)
            row.update(
                {
                    **common,
                    "method": method_name,
                    "alpha": float(frontier_alpha),
                }
            )
            frontier_rows.append(row)

    threshold_frontier = threshold_row.copy()
    threshold_frontier.update({"method": "cost_tuned_threshold", "alpha": np.nan})
    frontier_rows.append(threshold_frontier)
    credit_scenario_rows = credit_default_scenarios(
        dataset.name,
        seed,
        model_name,
        probability_calibration,
        y_cal_final,
        y_test,
        cal_proba,
        test_proba,
        mondrian_mask,
    )

    return {
        "method_rows": method_rows,
        "frontier_rows": frontier_rows,
        "deferral_rows": deferral_rows,
        "credit_scenario_rows": credit_scenario_rows,
    }


def make_model(model_name: str, seed: int):
    if model_name == "hgb":
        return (
            make_pipeline(
                StandardScaler(with_mean=False),
                HistGradientBoostingClassifier(random_state=seed),
            ),
            "histgradientboostingclassifier",
        )
    if model_name == "rf":
        return (
            make_pipeline(
                RandomForestClassifier(
                    n_estimators=300,
                    min_samples_leaf=2,
                    n_jobs=-1,
                    class_weight=None,
                    random_state=seed,
                )
            ),
            "randomforestclassifier",
        )
    if model_name == "et":
        return (
            make_pipeline(
                ExtraTreesClassifier(
                    n_estimators=300,
                    min_samples_leaf=2,
                    n_jobs=-1,
                    random_state=seed,
                )
            ),
            "extratreesclassifier",
        )
    if model_name == "gb":
        return (
            make_pipeline(GradientBoostingClassifier(random_state=seed)),
            "gradientboostingclassifier",
        )
    if model_name == "ada":
        return (
            make_pipeline(AdaBoostClassifier(random_state=seed)),
            "adaboostclassifier",
        )
    if model_name == "gnb":
        return (
            make_pipeline(StandardScaler(), GaussianNB()),
            "gaussiannb",
        )
    return (
        make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=3000,
                solver="lbfgs",
                random_state=seed,
            ),
        ),
        "logisticregression",
    )


def calibrate_proba(
    model,
    X_proba_cal: np.ndarray | None,
    y_proba_cal: np.ndarray | None,
    X: np.ndarray,
    method: str,
) -> np.ndarray:
    proba = model.predict_proba(X)
    if method == "none":
        return proba
    if X_proba_cal is None or y_proba_cal is None:
        raise ValueError("probability calibration split is required")
    if np.unique(y_proba_cal).size < 2:
        return proba
    cal_positive = model.predict_proba(X_proba_cal)[:, 1]
    target_positive = proba[:, 1]
    if method == "sigmoid":
        calibrator = LogisticRegression(solver="lbfgs")
        calibrator.fit(cal_positive.reshape(-1, 1), y_proba_cal)
        positive = calibrator.predict_proba(target_positive.reshape(-1, 1))[:, 1]
    elif method == "isotonic":
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(cal_positive, y_proba_cal)
        positive = calibrator.predict(target_positive)
    else:
        raise ValueError(f"unsupported probability calibration: {method}")
    positive = np.clip(positive, 0.0, 1.0)
    return np.column_stack([1.0 - positive, positive])


def balanced_sample_weight(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=int)
    counts = np.bincount(y, minlength=2)
    weights = np.zeros(2, dtype=float)
    present = counts > 0
    weights[present] = y.size / (present.sum() * counts[present])
    return weights[y]


def random_oversample(
    X: np.ndarray, y: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Balance training data by duplicating minority rows only.

    Resampling happens after the train split, so calibration and test rows stay
    untouched. This is a simple SMOTE-free baseline with no new dependency.
    """
    y = np.asarray(y, dtype=int)
    classes, counts = np.unique(y, return_counts=True)
    if classes.size != 2 or counts[0] == counts[1]:
        return X, y
    rng = np.random.default_rng(seed)
    minority = classes[np.argmin(counts)]
    majority_count = int(counts.max())
    minority_idx = np.flatnonzero(y == minority)
    extra_idx = rng.choice(
        minority_idx,
        size=majority_count - int(counts.min()),
        replace=True,
    )
    idx = np.concatenate([np.arange(y.size), extra_idx])
    rng.shuffle(idx)
    return X[idx], y[idx]


def smote(
    X: np.ndarray, y: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Balance numeric training data with simple minority interpolation."""
    y = np.asarray(y, dtype=int)
    classes, counts = np.unique(y, return_counts=True)
    if classes.size != 2 or counts[0] == counts[1]:
        return X, y
    rng = np.random.default_rng(seed)
    minority = classes[np.argmin(counts)]
    minority_X = np.asarray(X)[y == minority]
    n_extra = int(counts.max() - counts.min())
    left = rng.integers(0, minority_X.shape[0], size=n_extra)
    right = rng.integers(0, minority_X.shape[0], size=n_extra)
    mix = rng.random(n_extra)[:, None]
    synthetic = minority_X[left] + mix * (minority_X[right] - minority_X[left])
    return np.concatenate([X, synthetic]), np.concatenate([y, np.full(n_extra, minority)])


def tune_confidence_tau(
    cal_y: np.ndarray,
    cal_proba: np.ndarray,
    costs: CostMatrix,
    grid_size: int = 101,
) -> float:
    thresholds = np.linspace(0.5, 1.0, grid_size)
    losses = []
    for tau in thresholds:
        losses.append(
            evaluate_prediction_mask(
                cal_y, confidence_reject_mask(cal_proba, tau), costs
            )["mean_cost"]
        )
    return float(thresholds[int(np.argmin(losses))])


def tune_risk_controlled_tau(
    cal_y: np.ndarray,
    cal_proba: np.ndarray,
    target_coverage: float,
    grid_size: int = 101,
) -> float:
    thresholds = np.linspace(0.5, 1.0, grid_size)
    candidates = []
    for tau in thresholds:
        metrics = evaluate_prediction_mask(
            cal_y, confidence_reject_mask(cal_proba, tau), CostMatrix()
        )
        if metrics["marginal_coverage"] >= target_coverage:
            candidates.append((metrics["abstention_rate"], float(tau)))
    return min(candidates)[1] if candidates else 1.0


def tune_class_conditional_tau(
    cal_y: np.ndarray,
    cal_proba: np.ndarray,
    target_coverage: float,
    grid_size: int = 101,
) -> tuple[float, float]:
    pred = np.argmax(cal_proba, axis=1)
    conf = np.max(cal_proba, axis=1)
    thresholds = np.linspace(0.5, 1.0, grid_size)
    taus = []
    for label in (0, 1):
        best = 1.0
        for tau in thresholds:
            automated = (pred == label) & (conf >= tau)
            if not automated.any():
                continue
            if np.mean(cal_y[automated] == label) >= target_coverage:
                best = float(tau)
                break
        taus.append(best)
    return tuple(taus)


def tune_cost_controlled_mondrian_alpha(
    cal_proba: np.ndarray,
    cal_y: np.ndarray,
    costs: CostMatrix,
    target_coverage: float,
    alpha_grid: np.ndarray,
    seed: int,
) -> float:
    idx = np.arange(cal_y.size)
    if np.min(np.bincount(cal_y, minlength=2)) < 2:
        return float(alpha_grid[0])
    fit_idx, tune_idx = train_test_split(
        idx,
        test_size=0.5,
        stratify=cal_y,
        random_state=seed + 17,
    )
    candidates = []
    for alpha in alpha_grid:
        cp = ConformalPredictor.fit(
            cal_proba[fit_idx], cal_y[fit_idx], alpha=float(alpha), mode="mondrian"
        )
        metrics = evaluate_prediction_mask(
            cal_y[tune_idx], cp.predict_mask(cal_proba[tune_idx]), costs
        )
        if metrics["coverage_class_1"] >= target_coverage:
            candidates.append((metrics["mean_cost"], float(alpha)))
    if candidates:
        return min(candidates)[1]
    return float(alpha_grid[0])


def confidence_reject_mask(proba: np.ndarray, tau: float) -> np.ndarray:
    pred = np.argmax(proba, axis=1)
    confident = np.max(proba, axis=1) >= tau
    mask = np.ones_like(proba, dtype=bool)
    mask[confident] = False
    mask[np.flatnonzero(confident), pred[confident]] = True
    return mask


def class_conditional_reject_mask(
    proba: np.ndarray,
    taus: tuple[float, float],
) -> np.ndarray:
    pred = np.argmax(proba, axis=1)
    confidence = np.max(proba, axis=1)
    tau_by_pred = np.asarray(taus, dtype=float)[pred]
    confident = confidence >= tau_by_pred
    mask = np.ones_like(proba, dtype=bool)
    mask[confident] = False
    mask[np.flatnonzero(confident), pred[confident]] = True
    return mask


def capacity_limited_mask(
    prediction_mask: np.ndarray,
    proba: np.ndarray,
    max_defer_rate: float,
) -> np.ndarray:
    mask = np.asarray(prediction_mask, dtype=bool).copy()
    deferred = np.flatnonzero(mask.sum(axis=1) != 1)
    max_deferred = int(np.floor(max_defer_rate * mask.shape[0]))
    overflow = deferred.size - max_deferred
    if overflow <= 0:
        return mask

    confidence = np.max(proba[deferred], axis=1)
    automate = deferred[np.argsort(-confidence)[:overflow]]
    pred = np.argmax(proba[automate], axis=1)
    mask[automate] = False
    mask[automate, pred] = True
    return mask


def credit_default_scenarios(
    dataset_name: str,
    seed: int,
    model_name: str,
    probability_calibration: str,
    y_cal: np.ndarray,
    y_test: np.ndarray,
    cal_proba: np.ndarray,
    test_proba: np.ndarray,
    mondrian_mask: np.ndarray,
) -> list[dict[str, float | str | int]]:
    if dataset_name not in CREDIT_DATASETS:
        return []
    rows = []
    for fn_cost in CREDIT_FN_COSTS:
        for defer_cost in CREDIT_DEFER_COSTS:
            costs = CostMatrix(false_positive=1.0, false_negative=fn_cost, defer=defer_cost)
            threshold = tune_cost_threshold(y_cal, cal_proba[:, 1], costs)
            threshold_pred = (test_proba[:, 1] >= threshold).astype(int)
            for method, metrics in [
                ("mondrian_cp", evaluate_prediction_mask(y_test, mondrian_mask, costs)),
                ("cost_tuned_threshold", evaluate_hard_predictions(y_test, threshold_pred, costs)),
            ]:
                rows.append(
                    {
                        "dataset": dataset_name,
                        "seed": seed,
                        "model": model_name,
                        "probability_calibration": probability_calibration,
                        "method": method,
                        "fn_cost": fn_cost,
                        "fp_cost": 1.0,
                        "defer_cost": defer_cost,
                        "threshold": threshold if method == "cost_tuned_threshold" else np.nan,
                        **metrics,
                    }
                )
    return rows


def summarize_methods(method_table: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "marginal_coverage",
        "coverage_class_0",
        "coverage_class_1",
        "average_set_size",
        "abstention_rate",
        "mean_cost",
    ]
    rows = []
    for (dataset, model, calibration, method), group in method_table.groupby(
        ["dataset", "model", "probability_calibration", "method"]
    ):
        row = {
            "dataset": dataset,
            "model": model,
            "probability_calibration": calibration,
            "method": method,
            "n_seeds": int(group["seed"].nunique()),
        }
        for metric in metric_cols:
            mean, lo, hi = mean_ci(group[metric].to_numpy(dtype=float))
            row[f"{metric}_mean"] = mean
            row[f"{metric}_ci_lo"] = lo
            row[f"{metric}_ci_hi"] = hi
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_deferral(deferral: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, model, calibration, defer_cost), group in deferral.groupby(
        ["dataset", "model", "probability_calibration", "defer_cost"]
    ):
        diff_mean, diff_lo, diff_hi = mean_ci(group["cost_diff"].to_numpy(dtype=float))
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "probability_calibration": calibration,
                "defer_cost": defer_cost,
                "n_seeds": int(group["seed"].nunique()),
                "mondrian_cost": float(group["mondrian_cost"].mean()),
                "mondrian_abstain": float(group["mondrian_abstain"].mean()),
                "threshold_cost": float(group["threshold_cost"].mean()),
                "cost_diff": diff_mean,
                "cost_diff_ci_lo": diff_lo,
                "cost_diff_ci_hi": diff_hi,
                "p_mondrian_cheaper": float(group["p_mondrian_cheaper"].mean()),
                "mondrian_wins": bool(diff_mean < 0.0),
            }
        )
    return pd.DataFrame(rows)


def paired_method_tests(method_table: pd.DataFrame, baseline: str) -> pd.DataFrame:
    metrics = ["coverage_class_1", "mean_cost", "abstention_rate"]
    rows = []
    keys = ["dataset", "model", "probability_calibration", "seed"]
    for (dataset, model, calibration), group in method_table.groupby(
        ["dataset", "model", "probability_calibration"]
    ):
        wide = group.pivot_table(index=keys, columns="method", values=metrics)
        if baseline not in wide["mean_cost"]:
            continue
        for method in group["method"].drop_duplicates():
            if method == baseline:
                continue
            for metric in metrics:
                if method not in wide[metric]:
                    continue
                diff = (wide[metric][baseline] - wide[metric][method]).dropna().to_numpy()
                if diff.size == 0:
                    continue
                mean, lo, hi = mean_ci(diff)
                rows.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "probability_calibration": calibration,
                        "baseline": baseline,
                        "method": method,
                        "metric": metric,
                        "n_pairs": int(diff.size),
                        "baseline_minus_method_mean": mean,
                        "ci_lo": lo,
                        "ci_hi": hi,
                        "sign_test_p": sign_test_p(diff),
                    }
                )
    return pd.DataFrame(rows)


def cross_dataset_tests(method_summary: pd.DataFrame, baseline: str) -> pd.DataFrame:
    metrics = ["coverage_class_1_mean", "mean_cost_mean", "abstention_rate_mean"]
    rows = []
    keys = ["dataset", "model", "probability_calibration"]
    wide = method_summary.pivot_table(index=keys, columns="method", values=metrics)
    if baseline not in wide["mean_cost_mean"]:
        return pd.DataFrame()
    for method in method_summary["method"].drop_duplicates():
        if method == baseline:
            continue
        for metric in metrics:
            if method not in wide[metric]:
                continue
            diff = (wide[metric][baseline] - wide[metric][method]).dropna().to_numpy()
            if diff.size == 0:
                continue
            mean, lo, hi = mean_ci(diff)
            rows.append(
                {
                    "baseline": baseline,
                    "method": method,
                    "metric": metric.removesuffix("_mean"),
                    "n_dataset_model_pairs": int(diff.size),
                    "baseline_minus_method_mean": mean,
                    "ci_lo": lo,
                    "ci_hi": hi,
                    "sign_test_p": sign_test_p(diff),
                    "baseline_mean_rank": mean_rank(method_summary, baseline, metric),
                    "method_mean_rank": mean_rank(method_summary, method, metric),
                }
            )
    return pd.DataFrame(rows)


def summarize_models(method_summary: pd.DataFrame) -> pd.DataFrame:
    frame = method_summary.copy()
    frame["cost_rank"] = frame.groupby(
        ["dataset", "probability_calibration", "method"]
    )["mean_cost_mean"].rank(method="average")
    frame["minority_coverage_rank"] = frame.groupby(
        ["dataset", "probability_calibration", "method"]
    )["coverage_class_1_mean"].rank(method="average", ascending=False)
    cols = [
        "marginal_coverage_mean",
        "coverage_class_1_mean",
        "average_set_size_mean",
        "abstention_rate_mean",
        "mean_cost_mean",
        "cost_rank",
        "minority_coverage_rank",
    ]
    return frame.groupby(["model", "probability_calibration", "method"])[cols].mean().reset_index()


def mean_rank(method_summary: pd.DataFrame, method: str, metric: str) -> float:
    ascending = metric != "coverage_class_1_mean"
    frame = method_summary.copy()
    frame["rank"] = frame.groupby(["dataset", "model", "probability_calibration"])[metric].rank(
        method="average",
        ascending=ascending,
    )
    values = frame.loc[frame["method"] == method, "rank"].to_numpy(dtype=float)
    return float(np.nanmean(values)) if values.size else float("nan")


def credit_default_case(
    method_summary: pd.DataFrame,
    deferral_summary: pd.DataFrame,
) -> pd.DataFrame:
    frame = method_summary[method_summary["dataset"].isin(["uci_default", "credit_g"])].copy()
    if frame.empty:
        return frame
    defer = deferral_summary[deferral_summary["dataset"].isin(["uci_default", "credit_g"])]
    break_even = defer.groupby(
        ["dataset", "model", "probability_calibration"]
    ).apply(estimate_break_even).rename("break_even_defer_cost")
    frame = frame.merge(
        break_even.reset_index(),
        on=["dataset", "model", "probability_calibration"],
        how="left",
    )
    frame["case_note"] = np.where(
        frame["dataset"].eq("uci_default"),
        "UCI credit default; class 1 = default; false negative is missed default.",
        "German credit; class 1 follows configured positive mapping.",
    )
    return frame


def summarize_ablations(frontier: pd.DataFrame, deferral_summary: pd.DataFrame) -> pd.DataFrame:
    alpha_rows = (
        frontier.groupby(
            ["dataset", "model", "probability_calibration", "method", "alpha"],
            dropna=False,
        )
        [["coverage_class_1", "abstention_rate", "mean_cost"]]
        .mean()
        .reset_index()
    )
    alpha_rows["ablation"] = "alpha_frontier"
    alpha_rows = alpha_rows.rename(columns={"alpha": "setting"})
    defer_rows = deferral_summary[
        [
            "dataset",
            "model",
            "probability_calibration",
            "defer_cost",
            "mondrian_abstain",
            "cost_diff",
        ]
    ].copy()
    defer_rows["method"] = "mondrian_cp_vs_cost_tuned_threshold"
    defer_rows["coverage_class_1"] = np.nan
    defer_rows = defer_rows.rename(
        columns={
            "defer_cost": "setting",
            "mondrian_abstain": "abstention_rate",
            "cost_diff": "mean_cost",
        }
    )
    defer_rows["ablation"] = "defer_cost"
    return pd.concat(
        [alpha_rows, defer_rows[alpha_rows.columns]],
        ignore_index=True,
        sort=False,
    )


def failure_analysis(method_summary: pd.DataFrame, deferral_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, model, calibration), group in method_summary.groupby(
        ["dataset", "model", "probability_calibration"]
    ):
        marginal = one_row(group, "marginal_cp")
        mondrian = one_row(group, "mondrian_cp")
        if marginal is None or mondrian is None:
            continue
        defer = deferral_summary[
            (deferral_summary["dataset"] == dataset)
            & (deferral_summary["model"] == model)
            & (deferral_summary["probability_calibration"] == calibration)
        ]
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "probability_calibration": calibration,
                "marginal_minority_coverage": marginal["coverage_class_1_mean"],
                "mondrian_minority_coverage": mondrian["coverage_class_1_mean"],
                "coverage_gain": mondrian["coverage_class_1_mean"] - marginal["coverage_class_1_mean"],
                "mondrian_abstention": mondrian["abstention_rate_mean"],
                "mondrian_cost": mondrian["mean_cost_mean"],
                "break_even_defer_cost": estimate_break_even(defer),
                "failure_mode": classify_failure(marginal, mondrian, defer),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["failure_mode", "mondrian_abstention"], ascending=[True, False]
    )


def one_row(group: pd.DataFrame, method: str) -> pd.Series | None:
    rows = group[group["method"] == method]
    if rows.empty:
        return None
    return rows.iloc[0]


def classify_failure(
    marginal: pd.Series,
    mondrian: pd.Series,
    deferral: pd.DataFrame,
) -> str:
    if mondrian["abstention_rate_mean"] > 0.75:
        return "high_abstention"
    if estimate_break_even(deferral) < 0.25:
        return "low_review_cost_tolerance"
    if marginal["coverage_class_1_mean"] < 0.2:
        return "marginal_undercoverage"
    return "watch"


def estimate_break_even(group: pd.DataFrame) -> float:
    if group.empty:
        return float("nan")
    ordered = group.sort_values("defer_cost")
    xs = ordered["defer_cost"].to_numpy(dtype=float)
    ys = ordered["cost_diff"].to_numpy(dtype=float)
    if ys[0] >= 0.0:
        return float(xs[0])
    for idx in range(1, ys.size):
        if ys[idx] >= 0.0:
            x0, x1 = xs[idx - 1], xs[idx]
            y0, y1 = ys[idx - 1], ys[idx]
            return float(x0 + (0.0 - y0) * (x1 - x0) / (y1 - y0))
    return float("inf")


def sign_test_p(values: np.ndarray) -> float:
    nonzero = values[values != 0.0]
    n = nonzero.size
    if n == 0:
        return 1.0
    wins = int(np.sum(nonzero > 0.0))
    tail = min(wins, n - wins)
    p = 2.0 * sum(math.comb(n, k) for k in range(tail + 1)) / (2.0**n)
    return float(min(1.0, p))


def mean_ci(values: np.ndarray) -> tuple[float, float, float]:
    values = values[np.isfinite(values)]
    mean = float(np.mean(values))
    if values.size < 2:
        return mean, float("nan"), float("nan")
    half_width = 1.96 * float(np.std(values, ddof=1)) / np.sqrt(values.size)
    return mean, mean - half_width, mean + half_width


def paired_bootstrap_diff(
    paired_diffs: np.ndarray,
    samples: int,
    seed: int,
) -> dict[str, float]:
    paired_diffs = np.asarray(paired_diffs, dtype=float)
    observed = float(np.mean(paired_diffs))
    if samples <= 0 or paired_diffs.size == 0:
        return {
            "observed": observed,
            "ci_lo": float("nan"),
            "ci_hi": float("nan"),
            "p_less_than_zero": float("nan"),
        }
    rng = np.random.default_rng(seed)
    boot_means = np.empty(samples, dtype=float)
    for idx in range(samples):
        boot_means[idx] = np.mean(
            paired_diffs[rng.integers(0, paired_diffs.size, size=paired_diffs.size)]
        )
    lo, hi = np.quantile(boot_means, [0.025, 0.975])
    return {
        "observed": observed,
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "p_less_than_zero": float(np.mean(boot_means < 0.0)),
    }


if __name__ == "__main__":
    main()
