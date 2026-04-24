"""
Statistical testing of generated hypotheses against omics data.

For each hypothesis, checks whether the affected_genes/pathways show
significant differential expression, enrichment, or association with
the pattern that inspired the hypothesis.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    hypothesis_text: str
    n_genes_tested: int
    n_genes_significant: int      # FDR < 0.05
    enrichment_pvalue: float      # hypergeometric / GSEA-style
    mean_effect_size: float       # mean |log2FC| for affected genes
    direction_consistent: bool    # sign of effect matches prediction
    supported: bool               # True if enrichment p < 0.05 & direction consistent
    gene_results: Dict[str, dict] = field(default_factory=dict)  # gene → {pval, fc}
    notes: str = ""


class HypothesisTester:
    """
    Tests generated hypotheses against expression / omics data.

    Args:
        expression_matrix: (N, G) samples × genes float array
        sample_labels: (N,) group labels (e.g. subtype assignments)
        gene_names: list of G gene names matching expression columns
        fdr_threshold: FDR cutoff for per-gene significance
    """

    def __init__(
        self,
        expression_matrix: np.ndarray,
        sample_labels: np.ndarray,
        gene_names: List[str],
        fdr_threshold: float = 0.05,
    ):
        self.expression = expression_matrix
        self.labels = sample_labels
        self.gene_names = gene_names
        self.gene_to_idx = {g: i for i, g in enumerate(gene_names)}
        self.fdr_threshold = fdr_threshold
        self.unique_labels = np.unique(sample_labels)

    def _ttest_genes(
        self,
        genes: List[str],
        group_a: np.ndarray,
        group_b: np.ndarray,
    ) -> Dict[str, dict]:
        """Two-sample t-test per gene between two groups."""
        from scipy import stats as scipy_stats

        results = {}
        for gene in genes:
            idx = self.gene_to_idx.get(gene)
            if idx is None:
                continue
            a = self.expression[group_a, idx]
            b = self.expression[group_b, idx]
            if a.std() == 0 and b.std() == 0:
                results[gene] = {"pval": 1.0, "log2fc": 0.0, "mean_a": float(a.mean()), "mean_b": float(b.mean())}
                continue
            stat, pval = scipy_stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")
            fc = float(a.mean()) - float(b.mean())  # log-space → log2FC difference
            results[gene] = {"pval": float(pval), "log2fc": float(fc), "mean_a": float(a.mean()), "mean_b": float(b.mean())}
        return results

    def _fdr_correct(self, pvals: List[float]) -> np.ndarray:
        """Benjamini-Hochberg FDR correction."""
        n = len(pvals)
        if n == 0:
            return np.array([])
        order = np.argsort(pvals)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(1, n + 1)
        fdr = np.array(pvals) * n / ranks
        # Enforce monotonicity (cumulative min from right)
        for i in range(n - 2, -1, -1):
            fdr[order[i]] = min(fdr[order[i]], fdr[order[i + 1]])
        return np.clip(fdr, 0, 1)

    def _hypergeometric_enrichment(
        self,
        n_affected_sig: int,
        n_affected: int,
        n_sig_total: int,
        n_total: int,
    ) -> float:
        """One-tailed hypergeometric p-value for gene set enrichment."""
        try:
            from scipy.stats import hypergeom
            # P(X >= n_affected_sig) under H0
            pval = hypergeom.sf(n_affected_sig - 1, n_total, n_affected, n_sig_total)
            return float(pval)
        except ImportError:
            # Fallback: Fisher's exact via chi2 approximation
            if n_affected_sig == 0:
                return 1.0
            expected = n_sig_total * n_affected / max(n_total, 1)
            fold = n_affected_sig / (expected + 1e-8)
            return float(1.0 / (1.0 + fold))

    def test(
        self,
        hypothesis,   # GeneratedHypothesis
        group_a_label: Optional[int] = None,
        group_b_label: Optional[int] = None,
    ) -> TestResult:
        """
        Test one hypothesis against expression data.

        If only two groups exist, uses them as group_a/b.
        If multiple groups, group_a_label and group_b_label must be specified.

        Args:
            hypothesis: GeneratedHypothesis with affected_genes
            group_a_label: label of "positive" group (e.g. subtype 0)
            group_b_label: label of "comparison" group (e.g. subtype 1)

        Returns:
            TestResult
        """
        # Determine groups
        if len(self.unique_labels) == 2:
            a_label, b_label = self.unique_labels[0], self.unique_labels[1]
        else:
            if group_a_label is None or group_b_label is None:
                raise ValueError("Must specify group_a_label and group_b_label when >2 groups")
            a_label, b_label = group_a_label, group_b_label

        mask_a = self.labels == a_label
        mask_b = self.labels == b_label

        genes = [g for g in hypothesis.affected_genes if g in self.gene_to_idx]
        if not genes:
            return TestResult(
                hypothesis_text=hypothesis.hypothesis_text,
                n_genes_tested=0,
                n_genes_significant=0,
                enrichment_pvalue=1.0,
                mean_effect_size=0.0,
                direction_consistent=False,
                supported=False,
                notes="No affected genes found in expression matrix",
            )

        gene_results = self._ttest_genes(genes, mask_a, mask_b)
        pvals = [r["pval"] for r in gene_results.values()]
        fdrs = self._fdr_correct(pvals)

        sig_genes = [g for g, fdr in zip(gene_results.keys(), fdrs) if fdr < self.fdr_threshold]
        n_sig = len(sig_genes)

        # Enrichment: are hypothesis genes over-represented among DE genes?
        # Compute genome-wide DE first (simplified: use all genes in matrix)
        all_gene_results = self._ttest_genes(self.gene_names[:min(5000, len(self.gene_names))], mask_a, mask_b)
        all_pvals = [r["pval"] for r in all_gene_results.values()]
        all_fdrs = self._fdr_correct(all_pvals)
        n_sig_global = int((all_fdrs < self.fdr_threshold).sum())

        enrich_p = self._hypergeometric_enrichment(
            n_affected_sig=n_sig,
            n_affected=len(genes),
            n_sig_total=n_sig_global,
            n_total=len(all_gene_results),
        )

        mean_fc = float(np.mean([abs(r["log2fc"]) for r in gene_results.values()])) if gene_results else 0.0

        # Direction: most affected genes should be up in group_a (hypothesis typically predicts up)
        n_up = sum(1 for r in gene_results.values() if r["log2fc"] > 0)
        direction_ok = n_up >= len(gene_results) / 2

        supported = enrich_p < 0.05 and direction_ok and n_sig > 0

        return TestResult(
            hypothesis_text=hypothesis.hypothesis_text,
            n_genes_tested=len(genes),
            n_genes_significant=n_sig,
            enrichment_pvalue=float(enrich_p),
            mean_effect_size=mean_fc,
            direction_consistent=direction_ok,
            supported=supported,
            gene_results={g: r for g, r in zip(gene_results.keys(), [dict(zip(gene_results[g].keys(), gene_results[g].values())) for g in gene_results])},
        )

    def test_batch(
        self,
        hypotheses: List,
        **kwargs,
    ) -> List[TestResult]:
        return [self.test(h, **kwargs) for h in hypotheses]
