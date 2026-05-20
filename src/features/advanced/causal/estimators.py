"""
Treatment effect estimators for causal inference in pathology.

Implements ATE (Average Treatment Effect) and CATE (Conditional ATE)
using propensity score methods and meta-learners.

Reference:
    Künzel et al. "Metalearners for estimating heterogeneous treatment
    effects" (PNAS 2019)
    Robins et al. "Estimation of regression coefficients when some
    regressors are not always observed" (JASA 1994)
"""

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import cross_val_predict

logger = logging.getLogger(__name__)


def _default_outcome_model() -> BaseEstimator:
    return Ridge(alpha=1.0)


def _default_propensity_model() -> BaseEstimator:
    return LogisticRegression(max_iter=1000, C=1.0)


class IPWEstimator:
    """
    Inverse Propensity Weighting (IPW) estimator for ATE.

    ATE = E[Y(1) - Y(0)] estimated via:
        ATE_IPW = mean(T*Y/e(X) - (1-T)*Y/(1-e(X)))

    where e(X) = P(T=1|X) is the propensity score.
    """

    def __init__(
        self,
        propensity_model: Optional[BaseEstimator] = None,
        trim_quantile: float = 0.01,
    ):
        self.propensity_model = propensity_model or _default_propensity_model()
        self.trim_quantile = trim_quantile
        self._fitted = False

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> "IPWEstimator":
        self.propensity_model_ = clone(self.propensity_model)
        self.propensity_model_.fit(X, T)
        self._fitted = True
        return self

    def predict_ate(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> Tuple[float, float]:
        """
        Returns (ATE estimate, standard error).

        Uses trimming + stabilization to prevent variance explosion.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before predict_ate()")
        e = self.propensity_model_.predict_proba(X)[:, 1]

        # Trim extreme propensity scores to reduce variance
        lo = np.quantile(e, self.trim_quantile)
        hi = np.quantile(e, 1 - self.trim_quantile)
        mask = (e >= lo) & (e <= hi)
        e, T, Y = e[mask], T[mask], Y[mask]

        # Additional stabilization: clip to [0.05, 0.95] to prevent
        # division by near-zero
        e = np.clip(e, 0.05, 0.95)

        ipw = T * Y / e - (1 - T) * Y / (1 - e)
        ate = float(np.mean(ipw))
        se = float(np.std(ipw) / np.sqrt(len(ipw)))
        logger.info(f"IPW ATE={ate:.4f} ± {se:.4f} (n={mask.sum()})")
        return ate, se


class DoublyRobustEstimator:
    """
    Doubly Robust (AIPW) estimator for ATE.

    Consistent if either the outcome model OR propensity model is correctly
    specified — provides insurance against misspecification of one model.

    AIPW = mean(
        μ1(X) - μ0(X)
        + T(Y - μ1(X))/e(X)
        - (1-T)(Y - μ0(X))/(1-e(X))
    )
    """

    def __init__(
        self,
        outcome_model: Optional[BaseEstimator] = None,
        propensity_model: Optional[BaseEstimator] = None,
        n_folds: int = 5,
        trim_quantile: float = 0.01,
    ):
        self.outcome_model = outcome_model or _default_outcome_model()
        self.propensity_model = propensity_model or _default_propensity_model()
        self.n_folds = n_folds
        self.trim_quantile = trim_quantile

    def fit_predict_ate(
        self, X: np.ndarray, T: np.ndarray, Y: np.ndarray
    ) -> Tuple[float, float, np.ndarray]:
        """
        Fit models via cross-fitting and return (ATE, SE, per-sample CATE).
        """
        T = T.astype(float)

        # Cross-fitted propensity scores
        prop_model = clone(self.propensity_model)
        e = cross_val_predict(prop_model, X, T, cv=self.n_folds, method="predict_proba")[:, 1]

        # Cross-fitted outcome models for treated and control
        mu1 = np.zeros(len(Y))
        mu0 = np.zeros(len(Y))
        from sklearn.model_selection import KFold

        kf = KFold(n_splits=self.n_folds)
        for train_idx, test_idx in kf.split(X):
            om1 = clone(self.outcome_model)
            om0 = clone(self.outcome_model)
            treated_train = train_idx[T[train_idx] == 1]
            control_train = train_idx[T[train_idx] == 0]
            if len(treated_train) > 0:
                om1.fit(X[treated_train], Y[treated_train])
                mu1[test_idx] = om1.predict(X[test_idx])
            if len(control_train) > 0:
                om0.fit(X[control_train], Y[control_train])
                mu0[test_idx] = om0.predict(X[test_idx])

        # Trim propensity scores
        lo, hi = np.quantile(e, self.trim_quantile), np.quantile(e, 1 - self.trim_quantile)
        e = np.clip(e, lo, hi)

        # AIPW scores
        aipw = mu1 - mu0 + T * (Y - mu1) / e - (1 - T) * (Y - mu0) / (1 - e)
        ate = float(np.mean(aipw))
        se = float(np.std(aipw) / np.sqrt(len(aipw)))
        cate = mu1 - mu0  # Plugin CATE from outcome model difference
        logger.info(f"AIPW ATE={ate:.4f} ± {se:.4f}")
        return ate, se, cate


class TLearner:
    """
    T-Learner: fit separate outcome models for treated and control.
    CATE(x) = μ1(x) - μ0(x)
    """

    def __init__(self, base_learner: Optional[BaseEstimator] = None):
        self.base_learner = base_learner or _default_outcome_model()
        self._mu1: Optional[BaseEstimator] = None
        self._mu0: Optional[BaseEstimator] = None

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> "TLearner":
        self._mu1 = clone(self.base_learner)
        self._mu0 = clone(self.base_learner)
        treated = T == 1
        self._mu1.fit(X[treated], Y[treated])
        self._mu0.fit(X[~treated], Y[~treated])
        return self

    def predict_cate(self, X: np.ndarray) -> np.ndarray:
        if self._mu1 is None or self._mu0 is None:
            raise RuntimeError("Call fit() first")
        return self._mu1.predict(X) - self._mu0.predict(X)

    def predict_ate(self, X: np.ndarray) -> float:
        return float(np.mean(self.predict_cate(X)))


class XLearner:
    """
    X-Learner (Künzel et al. 2019): improved CATE for imbalanced treatment.

    Step 1: Fit μ1, μ0 (T-learner)
    Step 2: Impute counterfactual outcomes, fit CATE models τ1, τ0
    Step 3: Weighted combination via propensity score
    """

    def __init__(
        self,
        outcome_learner: Optional[BaseEstimator] = None,
        cate_learner: Optional[BaseEstimator] = None,
        propensity_model: Optional[BaseEstimator] = None,
    ):
        self.outcome_learner = outcome_learner or _default_outcome_model()
        self.cate_learner = cate_learner or _default_outcome_model()
        self.propensity_model = propensity_model or _default_propensity_model()

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> "XLearner":
        T = T.astype(int)
        treated, control = T == 1, T == 0

        # Step 1: outcome models
        self._mu1 = clone(self.outcome_learner).fit(X[treated], Y[treated])
        self._mu0 = clone(self.outcome_learner).fit(X[control], Y[control])

        # Step 2: impute counterfactuals
        pseudo_treated = Y[treated] - self._mu0.predict(X[treated])
        pseudo_control = self._mu1.predict(X[control]) - Y[control]

        self._tau1 = clone(self.cate_learner).fit(X[treated], pseudo_treated)
        self._tau0 = clone(self.cate_learner).fit(X[control], pseudo_control)

        # Step 3: propensity for weighting
        self._prop = clone(self.propensity_model).fit(X, T)
        return self

    def predict_cate(self, X: np.ndarray) -> np.ndarray:
        e = self._prop.predict_proba(X)[:, 1]
        tau1 = self._tau1.predict(X)
        tau0 = self._tau0.predict(X)
        return e * tau0 + (1 - e) * tau1

    def predict_ate(self, X: np.ndarray) -> float:
        return float(np.mean(self.predict_cate(X)))


def compute_ate(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    method: str = "doubly_robust",
) -> Dict[str, Any]:
    """
    Estimate Average Treatment Effect.

    Args:
        X: Covariates [n_samples, n_features]
        T: Binary treatment assignment [n_samples]
        Y: Observed outcome [n_samples]
        method: "ipw", "doubly_robust", "t_learner"

    Returns:
        Dict with ate, se, ci_lower, ci_upper, method
    """
    T = np.asarray(T, dtype=float)
    Y = np.asarray(Y, dtype=float)

    if method == "ipw":
        est = IPWEstimator()
        est.fit(X, T, Y)
        ate, se = est.predict_ate(X, T, Y)
    elif method == "doubly_robust":
        est = DoublyRobustEstimator()
        ate, se, _ = est.fit_predict_ate(X, T, Y)
    elif method == "t_learner":
        est = TLearner()
        est.fit(X, T, Y)
        ate = est.predict_ate(X)
        se = float("nan")
    else:
        raise ValueError(f"Unknown method: {method}. Choose ipw/doubly_robust/t_learner")

    return {
        "ate": ate,
        "se": se,
        "ci_lower": ate - 1.96 * se,
        "ci_upper": ate + 1.96 * se,
        "method": method,
    }


def compute_cate(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    X_test: Optional[np.ndarray] = None,
    method: str = "x_learner",
) -> np.ndarray:
    """
    Estimate Conditional Average Treatment Effect at X_test.

    Args:
        X: Training covariates
        T: Binary treatment [n_samples]
        Y: Observed outcome [n_samples]
        X_test: Test covariates (defaults to X)
        method: "t_learner" or "x_learner"

    Returns:
        CATE estimates [n_test_samples]
    """
    X_test = X if X_test is None else X_test
    if method == "t_learner":
        return TLearner().fit(X, T, Y).predict_cate(X_test)
    elif method == "x_learner":
        return XLearner().fit(X, T, Y).predict_cate(X_test)
    else:
        raise ValueError(f"Unknown method: {method}")
