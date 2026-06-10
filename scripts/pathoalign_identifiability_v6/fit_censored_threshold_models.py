from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

INPUT = Path(
    r"results/pathoalign_two_resource_analysis/"
    r"threshold_sensitivity.csv"
)
OUTPUT = INPUT.parent / "censored_threshold_models.csv"

PAIR_GRID = np.array(
    [0, 2, 5, 10, 15, 20, 30, 40, 50, 60, 75,
     100, 150, 200, 250, 300],
    dtype=float,
)

df = pd.read_csv(INPUT)

def nonlinear_indicator(series):
    mapped = (
        series.astype(str)
        .str.lower()
        .map({
            "false": 0.0,
            "true": 1.0,
            "0": 0.0,
            "1": 1.0,
        })
    )
    if mapped.isna().any():
        raise ValueError(
            f"Unknown nonlinear values: "
            f"{series[mapped.isna()].unique().tolist()}"
        )
    return mapped.to_numpy(dtype=float)

def boundary_interval(value):
    if pd.isna(value):
        return np.log(PAIR_GRID[-1]), np.inf

    value = float(value)
    positive = PAIR_GRID[PAIR_GRID > 0]

    position = np.where(positive == value)[0]
    if len(position) != 1:
        raise ValueError(
            f"Boundary {value} is not in the configured pair grid."
        )

    index = int(position[0])
    lower_count = (
        1e-6 if index == 0 else positive[index - 1]
    )

    return np.log(lower_count), np.log(value)

def fit_threshold(column):
    X = np.column_stack([
        np.ones(len(df)),
        np.log(df["n"].astype(float).to_numpy()),
        df["overlap"].astype(float).to_numpy(),
        nonlinear_indicator(df["nonlinear"]),
    ])

    intervals = [
        boundary_interval(value)
        for value in pd.to_numeric(
            df[column],
            errors="coerce",
        )
    ]

    lower = np.array([item[0] for item in intervals])
    upper = np.array([item[1] for item in intervals])

    def negative_log_likelihood(params):
        beta = params[:-1]
        sigma = np.exp(params[-1])
        mu = X @ beta

        z_lower = (lower - mu) / sigma
        cdf_lower = norm.cdf(z_lower)

        finite = np.isfinite(upper)
        probability = np.empty(len(df), dtype=float)

        z_upper = (upper[finite] - mu[finite]) / sigma
        probability[finite] = (
            norm.cdf(z_upper) - cdf_lower[finite]
        )

        probability[~finite] = (
            1.0 - cdf_lower[~finite]
        )

        probability = np.clip(
            probability,
            1e-12,
            1.0,
        )

        return -np.log(probability).sum()

    observed = pd.to_numeric(
        df[column],
        errors="coerce",
    )

    usable = observed.notna() & observed.gt(0)
    X_start = X[usable]
    y_start = np.log(observed[usable].to_numpy())

    beta_start, *_ = np.linalg.lstsq(
        X_start,
        y_start,
        rcond=None,
    )

    initial = np.concatenate([
        beta_start,
        [np.log(0.5)],
    ])

    result = minimize(
        negative_log_likelihood,
        initial,
        method="BFGS",
    )

    beta = result.x[:-1]

    return {
        "threshold": float(column),
        "intercept": beta[0],
        "log_n": beta[1],
        "overlap": beta[2],
        "nonlinear": beta[3],
        "sigma": np.exp(result.x[-1]),
        "negative_log_likelihood": result.fun,
        "converged": result.success,
        "n_conditions": len(df),
        "n_censored": int(observed.isna().sum()),
    }

threshold_columns = [
    column
    for column in df.columns
    if column not in {
        "n",
        "overlap",
        "nonlinear",
        "method",
    }
]

result = pd.DataFrame([
    fit_threshold(column)
    for column in threshold_columns
]).sort_values("threshold")

result.to_csv(OUTPUT, index=False)

print(result.to_string(index=False))
print(f"\nWrote {OUTPUT}")
