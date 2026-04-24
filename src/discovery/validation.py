"""
Statistical validation of discovered cancer subtypes.

All tests use established survival analysis methods to verify that
discovered subtypes have genuine prognostic value.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)


def log_rank_test(
    survival_times: np.ndarray,
    events: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, Any]:
    """
    Global log-rank test for survival difference between subtypes.

    Args:
        survival_times: Observed survival times [n]
        events: Event indicators [n] (1=event)
        labels: Subtype labels [n]

    Returns:
        Dict with statistic, p_value, n_groups, significant
    """
    groups = np.unique(labels)
    if len(groups) < 2:
        return {"statistic": 0.0, "p_value": 1.0, "n_groups": 1, "significant": False}

    # Collect observed and expected events per group
    all_times = np.sort(np.unique(survival_times[events == 1]))
    O = np.zeros(len(groups))
    E = np.zeros(len(groups))

    for t in all_times:
        # At-risk counts at time t
        at_risk = np.array([np.sum((survival_times >= t) & (labels == g)) for g in groups])
        n_total = at_risk.sum()
        if n_total == 0:
            continue
        # Events at time t
        observed = np.array(
            [np.sum((survival_times == t) & (events == 1) & (labels == g)) for g in groups]
        )
        d_total = observed.sum()
        expected = at_risk * d_total / n_total
        O += observed
        E += expected

    # Log-rank statistic (Mantel-Cox)
    numerator = O - E
    # Variance: sum over time points
    variance = np.zeros(len(groups))
    for t in all_times:
        at_risk = np.array([np.sum((survival_times >= t) & (labels == g)) for g in groups])
        n_total = at_risk.sum()
        if n_total <= 1:
            continue
        d_total = np.sum((survival_times == t) & (events == 1))
        for j in range(len(groups)):
            v = at_risk[j] * (n_total - at_risk[j]) * d_total * (n_total - d_total)
            variance[j] += v / (n_total**2 * (n_total - 1) + 1e-8)

    # Chi-squared statistic (df = k-1)
    denom = variance[:-1].sum()
    if denom <= 0:
        return {"statistic": 0.0, "p_value": 1.0, "n_groups": len(groups), "significant": False}

    stat = float((numerator[:-1] ** 2 / (variance[:-1] + 1e-8)).sum())
    df = len(groups) - 1
    p_value = float(1 - scipy_stats.chi2.cdf(stat, df=df))

    logger.info(f"Log-rank test: χ²={stat:.3f}, df={df}, p={p_value:.4e}")
    return {
        "statistic": stat,
        "p_value": p_value,
        "df": df,
        "n_groups": len(groups),
        "observed_events": O.tolist(),
        "expected_events": E.tolist(),
        "significant": p_value < 0.05,
    }


def concordance_index(
    risk_scores: np.ndarray,
    survival_times: np.ndarray,
    events: np.ndarray,
) -> float:
    """
    Harrell's C-index: fraction of concordant pairs among comparable pairs.

    A pair (i, j) is comparable if patient i had an event before j.
    It's concordant if patient i has a higher predicted risk than j.

    Returns C-index in [0, 1]; 0.5 = random, 1.0 = perfect.
    """
    n = len(survival_times)
    concordant = 0
    discordant = 0
    tied_risk = 0

    for i in range(n):
        if events[i] == 0:
            continue  # Censored: not a reference event
        for j in range(n):
            if i == j:
                continue
            if survival_times[j] <= survival_times[i]:
                continue  # j not comparable (died before or at same time)
            if risk_scores[i] > risk_scores[j]:
                concordant += 1
            elif risk_scores[i] < risk_scores[j]:
                discordant += 1
            else:
                tied_risk += 0.5

    total = concordant + discordant + tied_risk
    if total == 0:
        return 0.5
    c = (concordant + 0.5 * tied_risk) / total
    logger.info(f"C-index: {c:.4f} (concordant={concordant}, discordant={discordant})")
    return float(c)


def bootstrap_stability(
    features: np.ndarray,
    labels: np.ndarray,
    survival_times: np.ndarray,
    events: np.ndarray,
    n_bootstrap: int = 50,
    n_epochs: int = 30,
) -> Dict[str, float]:
    """
    Cluster label stability under bootstrap resampling (Jaccard index).

    High stability (Jaccard > 0.75) indicates subtypes are robust,
    not artifacts of the specific dataset sample.
    """
    from sklearn.metrics import adjusted_rand_score
    from .subtype import SurvivalAwareClusterer

    n = len(labels)
    k = len(np.unique(labels))
    ari_scores = []

    for b in range(n_bootstrap):
        idx = np.random.choice(n, size=n, replace=True)
        X_boot = features[idx]
        T_boot = survival_times[idx]
        E_boot = events[idx]
        try:
            clusterer = SurvivalAwareClusterer(
                input_dim=features.shape[1],
                k_min=k,
                k_max=k,  # Fix k to compare
                n_epochs=n_epochs,
            )
            labels_boot = clusterer.fit(X_boot, T_boot, E_boot)
            ari = adjusted_rand_score(labels[idx], labels_boot)
            ari_scores.append(ari)
        except Exception as e:
            logger.debug(f"Bootstrap {b} failed: {e}")

    if not ari_scores:
        return {"mean_ari": float("nan"), "std_ari": float("nan"), "stable": False}

    mean_ari = float(np.mean(ari_scores))
    std_ari = float(np.std(ari_scores))
    logger.info(
        f"Bootstrap stability: ARI={mean_ari:.3f} ± {std_ari:.3f} over {len(ari_scores)} resamples"
    )
    return {
        "mean_ari": mean_ari,
        "std_ari": std_ari,
        "n_bootstrap": len(ari_scores),
        "stable": mean_ari > 0.6,
    }


def subtype_enrichment(
    labels: np.ndarray,
    clinical_variable: np.ndarray,
    variable_name: str = "clinical_var",
) -> Dict[str, Any]:
    """
    Fisher's exact test for enrichment of a binary clinical variable
    within each discovered subtype vs all others.

    Args:
        labels: Subtype labels [n]
        clinical_variable: Binary clinical variable [n] (e.g., ER+/-)
        variable_name: Name for logging

    Returns:
        Dict with per-subtype odds ratios and p-values
    """
    groups = np.unique(labels)
    results = {}
    for g in groups:
        in_group = labels == g
        # 2x2 contingency table
        a = np.sum(in_group & (clinical_variable == 1))  # in group, has var
        b = np.sum(in_group & (clinical_variable == 0))  # in group, no var
        c = np.sum(~in_group & (clinical_variable == 1))  # not in group, has var
        d = np.sum(~in_group & (clinical_variable == 0))  # not in group, no var
        table = np.array([[a, b], [c, d]])
        odds_ratio, p_value = scipy_stats.fisher_exact(table)
        results[f"subtype_{g}"] = {
            "odds_ratio": float(odds_ratio),
            "p_value": float(p_value),
            "n_in_group": int(in_group.sum()),
            "n_with_var": int(a),
            "significant": p_value < 0.05,
        }
        logger.info(
            f"Subtype {g} enrichment for {variable_name}: " f"OR={odds_ratio:.2f}, p={p_value:.4f}"
        )
    return results
