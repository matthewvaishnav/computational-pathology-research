"""
Causal validation and sensitivity analysis.

Refutation tests verify causal estimates are not spurious.
Sensitivity analysis quantifies robustness to unmeasured confounding.
"""

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
from sklearn.base import clone

logger = logging.getLogger(__name__)


def refutation_random_cause(
    estimator,
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    n_simulations: int = 100,
    significance_level: float = 0.05,
) -> Dict[str, Any]:
    """
    Refutation test: add a random independent cause.

    If the ATE changes significantly when a random variable is added as
    a confounder, the estimate was not stable. A valid causal estimate
    should be robust to adding irrelevant variables.

    Returns:
        Dict with original_ate, mean_new_ate, p_value, passed
    """
    original_ate = estimator.fit(X, T, Y).predict_ate(X)
    new_ates = []
    for _ in range(n_simulations):
        random_cause = np.random.randn(len(X), 1)
        X_aug = np.hstack([X, random_cause])
        try:
            ate_new = clone(estimator).fit(X_aug, T, Y).predict_ate(X_aug)
            new_ates.append(ate_new)
        except Exception as e:
            logger.debug(f"Simulation failed: {e}")

    new_ates_arr = np.array(new_ates)
    # Check if original ATE falls within null distribution
    mean_new = float(np.mean(new_ates_arr))
    std_new = float(np.std(new_ates_arr))
    z_score = abs(original_ate - mean_new) / (std_new + 1e-8)
    # Two-tailed p-value under normality
    from scipy import stats as scipy_stats

    p_value = float(2 * (1 - scipy_stats.norm.cdf(z_score)))
    passed = p_value > significance_level

    logger.info(
        f"Random cause refutation: original_ATE={original_ate:.4f}, "
        f"mean_new_ATE={mean_new:.4f}, p={p_value:.4f}, passed={passed}"
    )
    return {
        "original_ate": original_ate,
        "mean_new_ate": mean_new,
        "std_new_ate": std_new,
        "p_value": p_value,
        "passed": passed,
        "n_simulations": len(new_ates),
    }


def refutation_data_subset(
    estimator,
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    subset_fraction: float = 0.8,
    n_simulations: int = 100,
    significance_level: float = 0.05,
) -> Dict[str, Any]:
    """
    Refutation test: ATE should be stable under data subsampling.

    Repeatedly subsample the data and re-estimate. A robust causal
    estimate should not change substantially with small data changes.
    """
    original_ate = estimator.fit(X, T, Y).predict_ate(X)
    n = len(X)
    subset_size = int(n * subset_fraction)
    new_ates = []

    for _ in range(n_simulations):
        idx = np.random.choice(n, size=subset_size, replace=False)
        try:
            ate_new = clone(estimator).fit(X[idx], T[idx], Y[idx]).predict_ate(X[idx])
            new_ates.append(ate_new)
        except Exception as e:
            logger.debug(f"Subset simulation failed: {e}")

    new_ates_arr = np.array(new_ates)
    mean_new = float(np.mean(new_ates_arr))
    std_new = float(np.std(new_ates_arr))
    relative_change = abs(original_ate - mean_new) / (abs(original_ate) + 1e-8)
    passed = relative_change < 0.1  # <10% change considered stable

    logger.info(
        f"Data subset refutation: original={original_ate:.4f}, "
        f"mean_subset={mean_new:.4f}, rel_change={relative_change:.2%}, passed={passed}"
    )
    return {
        "original_ate": original_ate,
        "mean_subset_ate": mean_new,
        "std_subset_ate": std_new,
        "relative_change": relative_change,
        "passed": passed,
        "subset_fraction": subset_fraction,
        "n_simulations": len(new_ates),
    }


def compute_evalue(
    ate: float,
    se: float,
    outcome_type: str = "continuous",
) -> Dict[str, float]:
    """
    Compute E-value for sensitivity to unmeasured confounding (VanderWeele & Ding 2017).

    The E-value is the minimum strength of association (on the risk ratio scale)
    that an unmeasured confounder would need to have with BOTH treatment and outcome
    to fully explain away the observed effect.

    For continuous outcomes, converts effect size to approximate RR using:
        RR ≈ exp(0.91 * d)  where d = ate / se (Cohen's d approximation)

    Returns:
        Dict with evalue, evalue_ci (for lower confidence bound), rr
    """
    if se <= 0:
        return {"evalue": float("nan"), "evalue_ci": float("nan"), "rr": float("nan")}

    # Convert to risk ratio scale (approximation for continuous outcomes)
    d = abs(ate) / (se + 1e-8)
    rr = float(np.exp(0.91 * d))  # Chinn (2000) conversion

    def _evalue_from_rr(r: float) -> float:
        if r <= 1.0:
            return 1.0
        return r + np.sqrt(r * (r - 1))

    evalue = _evalue_from_rr(rr)
    # E-value for lower confidence bound (more conservative)
    rr_lower = float(np.exp(0.91 * max(0, d - 1.96)))
    evalue_ci = _evalue_from_rr(rr_lower)

    logger.info(
        f"E-value: ATE={ate:.4f}, SE={se:.4f}, RR≈{rr:.2f}, "
        f"E-value={evalue:.2f}, E-value(CI)={evalue_ci:.2f}"
    )
    return {
        "ate": ate,
        "se": se,
        "rr": rr,
        "evalue": evalue,
        "evalue_ci": evalue_ci,
        "interpretation": (
            f"Unmeasured confounder would need RR≥{evalue:.2f} with both "
            f"treatment and outcome to explain away this effect"
        ),
    }


def check_positivity(
    X: np.ndarray,
    T: np.ndarray,
    propensity_model=None,
    threshold: float = 0.05,
) -> Dict[str, Any]:
    """
    Check overlap/positivity assumption: P(T=1|X) ∈ (threshold, 1-threshold).

    Positivity violations indicate regions of X-space where the treatment
    effect is not identifiable from observational data.

    Returns:
        Dict with n_violations, violation_fraction, propensity_scores, summary
    """
    from sklearn.linear_model import LogisticRegression

    model = propensity_model or LogisticRegression(max_iter=1000)
    model.fit(X, T)
    e = model.predict_proba(X)[:, 1]

    violations = (e < threshold) | (e > 1 - threshold)
    violation_fraction = float(violations.mean())

    summary = "PASS" if violation_fraction < 0.05 else "FAIL"
    logger.info(
        f"Positivity check: {violations.sum()}/{len(T)} violations "
        f"({violation_fraction:.1%}) — {summary}"
    )
    return {
        "propensity_scores": e,
        "n_violations": int(violations.sum()),
        "violation_fraction": violation_fraction,
        "threshold": threshold,
        "summary": summary,
        "min_propensity": float(e.min()),
        "max_propensity": float(e.max()),
        "mean_propensity": float(e.mean()),
    }
