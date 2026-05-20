"""
Statistical Tests for Clinical Validation

Implements comprehensive statistical testing framework for medical AI validation,
including significance tests, power analysis, and clinical trial methodology.
"""

import logging
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import (
    anderson,
    bartlett,
    chi2_contingency,
    fisher_exact,
    friedmanchisquare,
    kruskal,
    kstest,
    levene,
    mannwhitneyu,
    mcnemar,
    shapiro,
    ttest_ind,
    ttest_rel,
    wilcoxon,
)
from statsmodels.stats.contingency_tables import mcnemar as mcnemar_test
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.inter_rater import cohens_kappa, fleiss_kappa
from statsmodels.stats.power import chisquare_power, ttest_power, zt_ind_solve_power
from statsmodels.stats.proportion import proportion_confint, proportions_ztest

logger = logging.getLogger(__name__)


class TestType(Enum):
    """Types of statistical tests"""

    T_TEST_INDEPENDENT = "t_test_independent"
    T_TEST_PAIRED = "t_test_paired"
    MANN_WHITNEY_U = "mann_whitney_u"
    WILCOXON_SIGNED_RANK = "wilcoxon_signed_rank"
    CHI_SQUARE = "chi_square"
    FISHER_EXACT = "fisher_exact"
    MCNEMAR = "mcnemar"
    KRUSKAL_WALLIS = "kruskal_wallis"
    FRIEDMAN = "friedman"
    PROPORTION_TEST = "proportion_test"
    EQUIVALENCE_TEST = "equivalence_test"
    NON_INFERIORITY_TEST = "non_inferiority_test"


@dataclass
class TestResult:
    """Statistical test result"""

    test_name: str
    statistic: float
    p_value: float
    effect_size: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    power: Optional[float] = None
    sample_size: Optional[int] = None
    assumptions_met: Optional[Dict[str, bool]] = None
    interpretation: Optional[str] = None
    clinical_significance: Optional[bool] = None


@dataclass
class PowerAnalysisResult:
    """Power analysis result"""

    test_type: str
    effect_size: float
    alpha: float
    power: float
    sample_size: int
    alternative: str = "two-sided"


class ClinicalStatisticalTests:
    """Comprehensive statistical testing for clinical validation"""

    def __init__(self, alpha: float = 0.05, power_threshold: float = 0.8):
        """Initialize statistical testing framework"""
        self.alpha = alpha
        self.power_threshold = power_threshold

    def check_normality(self, data: np.ndarray) -> Dict[str, Any]:
        """Check normality assumptions using multiple tests"""
        results = {}

        # Shapiro-Wilk test (best for small samples)
        if len(data) <= 5000:
            shapiro_stat, shapiro_p = shapiro(data)
            results["shapiro_wilk"] = {
                "statistic": shapiro_stat,
                "p_value": shapiro_p,
                "normal": shapiro_p > self.alpha,
            }

        # Kolmogorov-Smirnov test
        ks_stat, ks_p = kstest(data, "norm", args=(np.mean(data), np.std(data)))
        results["kolmogorov_smirnov"] = {
            "statistic": ks_stat,
            "p_value": ks_p,
            "normal": ks_p > self.alpha,
        }

        # Anderson-Darling test
        ad_result = anderson(data, dist="norm")
        results["anderson_darling"] = {
            "statistic": ad_result.statistic,
            "critical_values": ad_result.critical_values,
            "significance_levels": ad_result.significance_levels,
            "normal": ad_result.statistic < ad_result.critical_values[2],  # 5% level
        }

        return results

    def check_equal_variances(self, *groups: np.ndarray) -> Dict[str, Any]:
        """Check equal variance assumptions"""
        results = {}

        # Levene's test (robust to non-normality)
        levene_stat, levene_p = levene(*groups)
        results["levene"] = {
            "statistic": levene_stat,
            "p_value": levene_p,
            "equal_variances": levene_p > self.alpha,
        }

        # Bartlett's test (assumes normality)
        if len(groups) >= 2:
            bartlett_stat, bartlett_p = bartlett(*groups)
            results["bartlett"] = {
                "statistic": bartlett_stat,
                "p_value": bartlett_p,
                "equal_variances": bartlett_p > self.alpha,
            }

        return results

    def calculate_effect_size(
        self, group1: np.ndarray, group2: np.ndarray, test_type: str = "cohen_d"
    ) -> float:
        """Calculate effect size measures"""
        if test_type == "cohen_d":
            # Cohen's d for t-tests
            pooled_std = np.sqrt(
                (
                    (len(group1) - 1) * np.var(group1, ddof=1)
                    + (len(group2) - 1) * np.var(group2, ddof=1)
                )
                / (len(group1) + len(group2) - 2)
            )
            return (np.mean(group1) - np.mean(group2)) / pooled_std

        elif test_type == "glass_delta":
            # Glass's delta (uses control group SD)
            return (np.mean(group1) - np.mean(group2)) / np.std(group2, ddof=1)

        elif test_type == "hedges_g":
            # Hedges' g (bias-corrected Cohen's d)
            cohen_d = self.calculate_effect_size(group1, group2, "cohen_d")
            n = len(group1) + len(group2)
            correction = 1 - (3 / (4 * n - 9))
            return cohen_d * correction

        elif test_type == "cliff_delta":
            # Cliff's delta for non-parametric effect size
            n1, n2 = len(group1), len(group2)
            dominance = 0
            for x1 in group1:
                for x2 in group2:
                    if x1 > x2:
                        dominance += 1
                    elif x1 < x2:
                        dominance -= 1
            return dominance / (n1 * n2)

        else:
            raise ValueError(f"Unknown effect size type: {test_type}")

    def independent_t_test(
        self, group1: np.ndarray, group2: np.ndarray, equal_var: Optional[bool] = None
    ) -> TestResult:
        """Independent samples t-test"""
        # Check assumptions
        normality1 = self.check_normality(group1)
        normality2 = self.check_normality(group2)

        if equal_var is None:
            variance_test = self.check_equal_variances(group1, group2)
            equal_var = variance_test["levene"]["equal_variances"]

        # Perform test
        statistic, p_value = ttest_ind(group1, group2, equal_var=equal_var)

        # Calculate effect size
        effect_size = self.calculate_effect_size(group1, group2, "cohen_d")

        # Calculate confidence interval for difference in means
        diff_mean = np.mean(group1) - np.mean(group2)
        if equal_var:
            pooled_se = np.sqrt(
                (np.var(group1, ddof=1) / len(group1)) + (np.var(group2, ddof=1) / len(group2))
            )
        else:
            pooled_se = np.sqrt(
                (np.var(group1, ddof=1) / len(group1)) + (np.var(group2, ddof=1) / len(group2))
            )

        df = len(group1) + len(group2) - 2 if equal_var else min(len(group1), len(group2)) - 1
        t_critical = stats.t.ppf(1 - self.alpha / 2, df)
        ci_lower = diff_mean - t_critical * pooled_se
        ci_upper = diff_mean + t_critical * pooled_se

        # Check assumptions
        assumptions = {
            "normality_group1": normality1.get("shapiro_wilk", {}).get("normal", True),
            "normality_group2": normality2.get("shapiro_wilk", {}).get("normal", True),
            "equal_variances": equal_var,
        }

        return TestResult(
            test_name="Independent t-test",
            statistic=statistic,
            p_value=p_value,
            effect_size=effect_size,
            confidence_interval=(ci_lower, ci_upper),
            sample_size=len(group1) + len(group2),
            assumptions_met=assumptions,
            interpretation=self._interpret_effect_size(effect_size, "cohen_d"),
        )

    def paired_t_test(self, before: np.ndarray, after: np.ndarray) -> TestResult:
        """Paired samples t-test"""
        if len(before) != len(after):
            raise ValueError("Paired samples must have equal length")

        differences = after - before

        # Check normality of differences
        normality = self.check_normality(differences)

        # Perform test
        statistic, p_value = ttest_rel(after, before)

        # Calculate effect size (Cohen's d for paired samples)
        effect_size = np.mean(differences) / np.std(differences, ddof=1)

        # Confidence interval for mean difference
        se_diff = np.std(differences, ddof=1) / np.sqrt(len(differences))
        df = len(differences) - 1
        t_critical = stats.t.ppf(1 - self.alpha / 2, df)
        ci_lower = np.mean(differences) - t_critical * se_diff
        ci_upper = np.mean(differences) + t_critical * se_diff

        assumptions = {
            "normality_differences": normality.get("shapiro_wilk", {}).get("normal", True)
        }

        return TestResult(
            test_name="Paired t-test",
            statistic=statistic,
            p_value=p_value,
            effect_size=effect_size,
            confidence_interval=(ci_lower, ci_upper),
            sample_size=len(before),
            assumptions_met=assumptions,
            interpretation=self._interpret_effect_size(effect_size, "cohen_d"),
        )

    def mann_whitney_u_test(
        self, group1: np.ndarray, group2: np.ndarray, alternative: str = "two-sided"
    ) -> TestResult:
        """Mann-Whitney U test (non-parametric alternative to t-test)"""
        statistic, p_value = mannwhitneyu(group1, group2, alternative=alternative)

        # Calculate effect size (Cliff's delta)
        effect_size = self.calculate_effect_size(group1, group2, "cliff_delta")

        return TestResult(
            test_name="Mann-Whitney U test",
            statistic=statistic,
            p_value=p_value,
            effect_size=effect_size,
            sample_size=len(group1) + len(group2),
            interpretation=self._interpret_effect_size(effect_size, "cliff_delta"),
        )

    def wilcoxon_signed_rank_test(self, before: np.ndarray, after: np.ndarray) -> TestResult:
        """Wilcoxon signed-rank test (non-parametric paired test)"""
        if len(before) != len(after):
            raise ValueError("Paired samples must have equal length")

        statistic, p_value = wilcoxon(after, before)

        # Effect size (r = Z / sqrt(N))
        z_score = stats.norm.ppf(1 - p_value / 2)  # Approximate Z from p-value
        effect_size = z_score / np.sqrt(len(before))

        return TestResult(
            test_name="Wilcoxon signed-rank test",
            statistic=statistic,
            p_value=p_value,
            effect_size=effect_size,
            sample_size=len(before),
            interpretation=self._interpret_effect_size(effect_size, "r"),
        )

    def chi_square_test(self, contingency_table: np.ndarray) -> TestResult:
        """Chi-square test of independence"""
        chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)

        # Calculate effect size (Cramér's V)
        n = np.sum(contingency_table)
        min_dim = min(contingency_table.shape) - 1
        cramers_v = np.sqrt(chi2_stat / (n * min_dim))

        return TestResult(
            test_name="Chi-square test",
            statistic=chi2_stat,
            p_value=p_value,
            effect_size=cramers_v,
            sample_size=int(n),
            interpretation=self._interpret_effect_size(cramers_v, "cramers_v"),
        )

    def fisher_exact_test(self, contingency_table: np.ndarray) -> TestResult:
        """Fisher's exact test (for 2x2 tables)"""
        if contingency_table.shape != (2, 2):
            raise ValueError("Fisher's exact test requires 2x2 contingency table")

        odds_ratio, p_value = fisher_exact(contingency_table)

        # Calculate effect size (odds ratio)
        effect_size = np.log(odds_ratio)  # Log odds ratio

        return TestResult(
            test_name="Fisher's exact test",
            statistic=odds_ratio,
            p_value=p_value,
            effect_size=effect_size,
            sample_size=int(np.sum(contingency_table)),
            interpretation=f"Odds ratio: {odds_ratio:.3f}",
        )

    def mcnemar_test(self, contingency_table: np.ndarray) -> TestResult:
        """McNemar's test for paired categorical data"""
        if contingency_table.shape != (2, 2):
            raise ValueError("McNemar's test requires 2x2 contingency table")

        result = mcnemar_test(contingency_table)

        return TestResult(
            test_name="McNemar's test",
            statistic=result.statistic,
            p_value=result.pvalue,
            sample_size=int(np.sum(contingency_table)),
        )

    def proportion_test(
        self, successes: List[int], totals: List[int], alternative: str = "two-sided"
    ) -> TestResult:
        """Test for equality of proportions"""
        z_stat, p_value = proportions_ztest(successes, totals)

        # Calculate effect size (Cohen's h)
        props = [s / t for s, t in zip(successes, totals)]
        if len(props) == 2:
            h = 2 * (np.arcsin(np.sqrt(props[0])) - np.arcsin(np.sqrt(props[1])))
            effect_size = h
        else:
            effect_size = None

        return TestResult(
            test_name="Proportion test",
            statistic=z_stat,
            p_value=p_value,
            effect_size=effect_size,
            sample_size=sum(totals),
            interpretation=(
                self._interpret_effect_size(effect_size, "cohen_h") if effect_size else None
            ),
        )

    def equivalence_test(
        self, group1: np.ndarray, group2: np.ndarray, equivalence_margin: float
    ) -> TestResult:
        """Two one-sided tests (TOST) for equivalence"""
        diff_mean = np.mean(group1) - np.mean(group2)
        pooled_se = np.sqrt(
            (np.var(group1, ddof=1) / len(group1)) + (np.var(group2, ddof=1) / len(group2))
        )

        # Test H0: diff >= margin vs H1: diff < margin
        t1 = (diff_mean - equivalence_margin) / pooled_se
        # Test H0: diff <= -margin vs H1: diff > -margin
        t2 = (diff_mean + equivalence_margin) / pooled_se

        df = len(group1) + len(group2) - 2
        p1 = stats.t.cdf(t1, df)
        p2 = 1 - stats.t.cdf(t2, df)

        # Equivalence if both null hypotheses are rejected
        p_value = max(p1, p2)
        equivalent = p_value < self.alpha

        return TestResult(
            test_name="Equivalence test (TOST)",
            statistic=min(abs(t1), abs(t2)),
            p_value=p_value,
            sample_size=len(group1) + len(group2),
            clinical_significance=equivalent,
            interpretation=(
                f"Equivalent within ±{equivalence_margin}" if equivalent else "Not equivalent"
            ),
        )

    def non_inferiority_test(
        self, treatment: np.ndarray, control: np.ndarray, non_inferiority_margin: float
    ) -> TestResult:
        """Non-inferiority test"""
        diff_mean = np.mean(treatment) - np.mean(control)
        pooled_se = np.sqrt(
            (np.var(treatment, ddof=1) / len(treatment)) + (np.var(control, ddof=1) / len(control))
        )

        # Test H0: diff <= -margin vs H1: diff > -margin
        t_stat = (diff_mean + non_inferiority_margin) / pooled_se
        df = len(treatment) + len(control) - 2
        p_value = 1 - stats.t.cdf(t_stat, df)

        non_inferior = p_value < self.alpha

        return TestResult(
            test_name="Non-inferiority test",
            statistic=t_stat,
            p_value=p_value,
            sample_size=len(treatment) + len(control),
            clinical_significance=non_inferior,
            interpretation=(
                f"Non-inferior (margin: {non_inferiority_margin})"
                if non_inferior
                else "Inferiority not ruled out"
            ),
        )

    def power_analysis_t_test(
        self, effect_size: float, alpha: float = None, power: float = None, sample_size: int = None
    ) -> PowerAnalysisResult:
        """Power analysis for t-test"""
        if alpha is None:
            alpha = self.alpha

        if power is None and sample_size is not None:
            # Calculate power given sample size
            power = ttest_power(effect_size, sample_size, alpha)
        elif sample_size is None and power is not None:
            # Calculate sample size given power
            sample_size = int(
                np.ceil(zt_ind_solve_power(effect_size, power, alpha, alternative="two-sided"))
            )
        else:
            raise ValueError("Must specify either power or sample_size")

        return PowerAnalysisResult(
            test_type="t-test",
            effect_size=effect_size,
            alpha=alpha,
            power=power,
            sample_size=sample_size,
        )

    def multiple_comparisons_correction(
        self, p_values: List[float], method: str = "bonferroni"
    ) -> Dict[str, Any]:
        """Apply multiple comparisons correction"""
        p_values = np.array(p_values)

        if method == "bonferroni":
            corrected_alpha = self.alpha / len(p_values)
            corrected_p = p_values * len(p_values)
            corrected_p = np.minimum(corrected_p, 1.0)

        elif method == "holm":
            # Holm-Bonferroni method
            sorted_indices = np.argsort(p_values)
            corrected_p = np.zeros_like(p_values)

            for i, idx in enumerate(sorted_indices):
                corrected_p[idx] = p_values[idx] * (len(p_values) - i)

            corrected_p = np.minimum(corrected_p, 1.0)
            corrected_alpha = self.alpha

        elif method == "fdr_bh":
            # Benjamini-Hochberg FDR control
            sorted_indices = np.argsort(p_values)
            corrected_p = np.zeros_like(p_values)

            for i, idx in enumerate(sorted_indices):
                corrected_p[idx] = p_values[idx] * len(p_values) / (i + 1)

            corrected_p = np.minimum(corrected_p, 1.0)
            corrected_alpha = self.alpha

        else:
            raise ValueError(f"Unknown correction method: {method}")

        significant = corrected_p < corrected_alpha

        return {
            "method": method,
            "original_p_values": p_values.tolist(),
            "corrected_p_values": corrected_p.tolist(),
            "corrected_alpha": corrected_alpha,
            "significant": significant.tolist(),
            "n_significant": int(np.sum(significant)),
        }

    def _interpret_effect_size(self, effect_size: float, measure: str) -> str:
        """Interpret effect size magnitude"""
        if effect_size is None:
            return "Effect size not calculated"

        abs_effect = abs(effect_size)

        if measure == "cohen_d":
            if abs_effect < 0.2:
                return f"Negligible effect (d = {effect_size:.3f})"
            elif abs_effect < 0.5:
                return f"Small effect (d = {effect_size:.3f})"
            elif abs_effect < 0.8:
                return f"Medium effect (d = {effect_size:.3f})"
            else:
                return f"Large effect (d = {effect_size:.3f})"

        elif measure == "cliff_delta":
            if abs_effect < 0.147:
                return f"Negligible effect (δ = {effect_size:.3f})"
            elif abs_effect < 0.33:
                return f"Small effect (δ = {effect_size:.3f})"
            elif abs_effect < 0.474:
                return f"Medium effect (δ = {effect_size:.3f})"
            else:
                return f"Large effect (δ = {effect_size:.3f})"

        elif measure == "cramers_v":
            if abs_effect < 0.1:
                return f"Negligible association (V = {effect_size:.3f})"
            elif abs_effect < 0.3:
                return f"Small association (V = {effect_size:.3f})"
            elif abs_effect < 0.5:
                return f"Medium association (V = {effect_size:.3f})"
            else:
                return f"Large association (V = {effect_size:.3f})"

        elif measure == "cohen_h":
            if abs_effect < 0.2:
                return f"Small difference (h = {effect_size:.3f})"
            elif abs_effect < 0.5:
                return f"Medium difference (h = {effect_size:.3f})"
            else:
                return f"Large difference (h = {effect_size:.3f})"

        elif measure == "r":
            if abs_effect < 0.1:
                return f"Small effect (r = {effect_size:.3f})"
            elif abs_effect < 0.3:
                return f"Medium effect (r = {effect_size:.3f})"
            else:
                return f"Large effect (r = {effect_size:.3f})"

        else:
            return f"Effect size: {effect_size:.3f}"


# Example usage and testing
if __name__ == "__main__":
    # Initialize statistical testing framework
    stats_tests = ClinicalStatisticalTests(alpha=0.05)

    # Generate sample data
    np.random.seed(42)

    # Test data for AI vs human pathologist accuracy
    ai_accuracy = np.random.normal(0.92, 0.05, 100)  # AI accuracy
    human_accuracy = np.random.normal(0.88, 0.08, 100)  # Human accuracy

    print("Clinical AI Validation Statistical Tests")
    print("=" * 50)

    # Independent t-test
    t_result = stats_tests.independent_t_test(ai_accuracy, human_accuracy)
    print(f"\n{t_result.test_name}:")
    print(f"  Statistic: {t_result.statistic:.4f}")
    print(f"  P-value: {t_result.p_value:.6f}")
    print(f"  Effect size: {t_result.effect_size:.4f}")
    print(f"  Interpretation: {t_result.interpretation}")
    print(f"  Significant: {t_result.p_value < 0.05}")

    # Mann-Whitney U test (non-parametric alternative)
    mw_result = stats_tests.mann_whitney_u_test(ai_accuracy, human_accuracy)
    print(f"\n{mw_result.test_name}:")
    print(f"  Statistic: {mw_result.statistic:.4f}")
    print(f"  P-value: {mw_result.p_value:.6f}")
    print(f"  Effect size: {mw_result.effect_size:.4f}")
    print(f"  Interpretation: {mw_result.interpretation}")

    # Power analysis
    power_result = stats_tests.power_analysis_t_test(effect_size=0.5, power=0.8)
    print(f"\nPower Analysis:")
    print(f"  Required sample size: {power_result.sample_size}")
    print(f"  Effect size: {power_result.effect_size}")
    print(f"  Power: {power_result.power}")

    # Multiple comparisons example
    p_values = [0.01, 0.03, 0.08, 0.12, 0.45]
    mc_result = stats_tests.multiple_comparisons_correction(p_values, "bonferroni")
    print(f"\nMultiple Comparisons (Bonferroni):")
    print(f"  Original p-values: {mc_result['original_p_values']}")
    print(f"  Corrected p-values: {mc_result['corrected_p_values']}")
    print(f"  Significant tests: {mc_result['n_significant']}/{len(p_values)}")

    # Equivalence test example
    equiv_result = stats_tests.equivalence_test(
        ai_accuracy, human_accuracy, equivalence_margin=0.02
    )
    print(f"\n{equiv_result.test_name}:")
    print(f"  P-value: {equiv_result.p_value:.6f}")
    print(f"  Interpretation: {equiv_result.interpretation}")
    print(f"  Clinically equivalent: {equiv_result.clinical_significance}")
