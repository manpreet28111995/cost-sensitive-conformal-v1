from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_digits, make_classification


@dataclass(frozen=True)
class Dataset:
    name: str
    X: np.ndarray
    y: np.ndarray


# Large, public, imbalanced binary datasets served via OpenML (no manual download).
# Existing datasets keep their paper-facing positive class. New generic binary
# benchmarks use the minority label as the costly positive class.
_OPENML_SPECS: dict[str, dict] = {
    "uci_default":    dict(kw=dict(data_id=42477),               pos=lambda y: y.astype(str) == "1"),
    "bank_marketing": dict(kw=dict(data_id=1461),                pos=lambda y: y.astype(str) == "2"),
    "adult":          dict(kw=dict(data_id=1590),                pos=lambda y: y.astype(str) == ">50K"),
    "aps_failure":    dict(kw=dict(data_id=41138),               pos=lambda y: y.astype(str) == "pos"),
    "diabetes130us":  dict(kw=dict(data_id=4541),                pos=lambda y: y.astype(str) == "<30"),
    "miniboone":      dict(kw=dict(data_id=41150),               pos=lambda y: y.astype(str) == "True"),
    "fraud":          dict(kw=dict(data_id=1597),                pos=lambda y: y.astype(str) == "1"),
    "mammography":    dict(kw=dict(data_id=310)),
    "sick_numeric":   dict(kw=dict(data_id=41946),                pos=lambda y: y.astype(str) == "sick"),
    "wilt":           dict(kw=dict(data_id=40983)),
    "ozone_level_8hr": dict(kw=dict(data_id=1487)),
    "seismic_bumps":  dict(kw=dict(data_id=45562)),
    "pc1":            dict(kw=dict(data_id=1068)),
    "oil_spill":      dict(kw=dict(data_id=311)),
    "credit_g":       dict(kw=dict(data_id=31)),
}

OPENML_DATASETS = tuple(_OPENML_SPECS)


def load_openml_dataset(name: str) -> Dataset:
    """Fetch a large imbalanced binary dataset from OpenML and binarize its target."""
    from sklearn.datasets import fetch_openml

    spec = _OPENML_SPECS[name]
    print(
        f"Downloading/loading OpenML dataset {name} "
        f"(id={spec['kw']['data_id']})...",
        flush=True,
    )
    bundle = fetch_openml(as_frame=True, parser="auto", **spec["kw"])
    X_frame = bundle.data.copy()
    for column in X_frame.columns:
        if str(X_frame[column].dtype) in ("object", "category"):
            X_frame[column] = X_frame[column].astype("category").cat.codes
    X = X_frame.fillna(0).to_numpy(dtype=float)
    y = _positive_mask(bundle.target, spec.get("pos")).astype(int).to_numpy()
    if len(np.unique(y)) != 2:
        raise ValueError(f"OpenML dataset {name!r} did not resolve to a binary target")
    print(f"Loaded {name}: {X.shape[0]} rows, {X.shape[1]} features", flush=True)
    return Dataset(name=name, X=X, y=y)


def _positive_mask(target: pd.Series, pos_rule) -> pd.Series:
    if target is None:
        raise ValueError("OpenML dataset did not include a target column")
    target = target.astype(str)
    if pos_rule is not None:
        return pos_rule(target)
    counts = target.value_counts()
    if counts.size != 2:
        raise ValueError("minority-positive mapping requires a binary target")
    return target == counts.idxmin()


def load_builtin_dataset(name: str, random_state: int) -> Dataset:
    if name == "synthetic":
        print(f"Generating synthetic dataset (random_state={random_state})...", flush=True)
        X, y = make_classification(
            n_samples=6000,
            n_features=30,
            n_informative=10,
            n_redundant=5,
            n_clusters_per_class=2,
            weights=[0.95, 0.05],
            class_sep=0.9,
            flip_y=0.02,
            random_state=random_state,
        )
        return Dataset(name=name, X=X, y=y.astype(int))

    if name == "breast_cancer":
        print("Loading sklearn breast_cancer dataset...", flush=True)
        raw = load_breast_cancer()
        # sklearn labels malignant as 0 and benign as 1. Positive = malignant.
        y = (raw.target == 0).astype(int)
        return Dataset(name=name, X=raw.data, y=y)

    if name == "digits_9_vs_rest":
        print("Loading sklearn digits_9_vs_rest dataset...", flush=True)
        raw = load_digits()
        y = (raw.target == 9).astype(int)
        return Dataset(name=name, X=raw.data, y=y)

    if name in _OPENML_SPECS:
        return load_openml_dataset(name)

    raise ValueError(
        f"unknown dataset {name!r}; choose synthetic, breast_cancer, digits_9_vs_rest, "
        f"or an OpenML set: {', '.join(OPENML_DATASETS)}"
    )


def load_csv_dataset(
    csv_path: str | Path,
    target: str,
    positive_label: str,
    dataset_name: str | None = None,
) -> Dataset:
    print(f"Loading CSV dataset {csv_path}...", flush=True)
    frame = pd.read_csv(csv_path)
    if target not in frame.columns:
        raise ValueError(f"target column {target!r} not found")

    y_raw = frame[target].astype(str)
    y = (y_raw == str(positive_label)).astype(int).to_numpy()
    X_frame = frame.drop(columns=[target])
    X_frame = pd.get_dummies(X_frame, drop_first=False)
    X = X_frame.to_numpy(dtype=float)

    if len(np.unique(y)) != 2:
        raise ValueError("CSV target must become binary after positive-label mapping")

    name = dataset_name or Path(csv_path).stem
    print(f"Loaded {name}: {X.shape[0]} rows, {X.shape[1]} features", flush=True)
    return Dataset(name=name, X=X, y=y)
