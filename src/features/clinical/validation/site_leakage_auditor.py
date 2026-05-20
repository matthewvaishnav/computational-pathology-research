"""
Site Leakage Auditor for Computational Pathology Models.

Research contribution: Systematic framework for detecting and quantifying
site-level leakage in pathology AI models — the phenomenon where models
learn to identify hospitals rather than pathological features.

Key finding this module enables:
    "We don't know if AI pathology models are diagnosing cancer or diagnosing hospitals."

Three leakage mechanisms detected:
    1. Feature leakage: model features encode site identity
    2. Attention leakage: attention weights predict site better than chance
    3. Performance leakage: AUC degrades when site information is removed

Integrates with:
    - src/clinical_validation/bias_detection.BiasDetector
    - src/interpretability/feature_importance.FeatureImportanceCalculator
    - src/clinical_validation/subgroup_analysis.ClinicalSubgroupAnalyzer

Usage:
    auditor = SiteLeakageAuditor(model, feature_extractor)
    report = auditor.audit(slides_by_site)
    print(report.summary())

    # Compare federated vs centralized
    comparison = SiteLeakageAuditor.compare_training_regimes(
        models={"federated": fed_model, "centralized": cent_model},
        slides_by_site=by_site,
    )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ── Data structures ───────────────────────────────────────────────────────────


@dataclass
class LeakageScore:
    """Quantifies one leakage mechanism."""

    mechanism: str  # "feature" | "attention" | "performance"
    score: float  # 0 = no leakage, 1 = complete leakage
    baseline: float  # random-chance baseline (1/num_sites)
    excess: float  # score - baseline (the actual leakage)
    severity: str  # "none" | "mild" | "moderate" | "severe"
    description: str

    @classmethod
    def compute(
        cls, mechanism: str, score: float, num_sites: int, description: str
    ) -> "LeakageScore":
        baseline = 1.0 / num_sites
        excess = max(0.0, score - baseline)
        if excess < 0.05:
            severity = "none"
        elif excess < 0.15:
            severity = "mild"
        elif excess < 0.30:
            severity = "moderate"
        else:
            severity = "severe"
        return cls(mechanism, score, baseline, excess, severity, description)


@dataclass
class SiteLeakageReport:
    """Full audit report for one model."""

    model_name: str
    num_sites: int
    num_slides: int

    feature_leakage: LeakageScore
    attention_leakage: Optional[LeakageScore]
    performance_leakage: LeakageScore

    # AUC with and without site information
    auc_with_site_info: float
    auc_without_site_info: float
    auc_degradation: float

    # Per-site performance
    per_site_auc: Dict[int, float]
    per_site_accuracy: Dict[int, float]

    # Cross-site attention consistency (higher = more site-invariant)
    cross_site_attention_correlation: Optional[float] = None

    def overall_leakage_score(self) -> float:
        """Composite leakage score 0-1."""
        scores = [self.feature_leakage.excess, self.performance_leakage.excess]
        if self.attention_leakage:
            scores.append(self.attention_leakage.excess)
        return float(np.mean(scores))

    def summary(self) -> str:
        lines = [
            f"\n{'='*60}",
            f"SITE LEAKAGE AUDIT: {self.model_name}",
            f"{'='*60}",
            f"Slides: {self.num_slides} across {self.num_sites} sites",
            f"",
            f"LEAKAGE SCORES (excess above {1/self.num_sites:.0%} random baseline):",
            f"  Feature leakage:     {self.feature_leakage.score:.1%} "
            f"(+{self.feature_leakage.excess:.1%}) [{self.feature_leakage.severity}]",
            f"  Performance leakage: {self.performance_leakage.score:.1%} "
            f"(+{self.performance_leakage.excess:.1%}) [{self.performance_leakage.severity}]",
        ]
        if self.attention_leakage:
            lines.append(
                f"  Attention leakage:   {self.attention_leakage.score:.1%} "
                f"(+{self.attention_leakage.excess:.1%}) [{self.attention_leakage.severity}]"
            )
        if self.cross_site_attention_correlation is not None:
            lines.append(
                f"  Cross-site attn corr: {self.cross_site_attention_correlation:.3f} "
                f"(1.0 = perfectly consistent)"
            )
        lines += [
            f"",
            f"AUC with site info:    {self.auc_with_site_info:.3f}",
            f"AUC without site info: {self.auc_without_site_info:.3f}",
            f"AUC degradation:       {self.auc_degradation:+.3f}",
            f"",
            f"Overall leakage score: {self.overall_leakage_score():.3f}",
            f"",
            f"VERDICT:",
        ]
        score = self.overall_leakage_score()
        if score < 0.05:
            lines.append("  ✅ No significant site leakage detected.")
            lines.append("     Model appears to learn site-invariant pathological features.")
        elif score < 0.15:
            lines.append("  🔶 Mild site leakage detected.")
            lines.append("     Model may be partially relying on scanner characteristics.")
        elif score < 0.30:
            lines.append("  ⚠️  Moderate site leakage detected.")
            lines.append("     Reported AUC likely inflated by site-specific shortcuts.")
        else:
            lines.append("  🚨 Severe site leakage detected.")
            lines.append("     Model is diagnosing hospitals, not pathology.")
            lines.append(
                "     Published results should not be trusted without site-controlled validation."
            )
        lines.append("=" * 60)
        return "\n".join(lines)


@dataclass
class RegimeComparison:
    """Comparison of leakage across training regimes."""

    reports: Dict[str, SiteLeakageReport]

    def summary(self) -> str:
        lines = [
            "\n" + "=" * 70,
            "TRAINING REGIME COMPARISON: Site Leakage",
            "=" * 70,
            f"\n{'Regime':<25} {'Feature':>10} {'Attention':>10} {'Perf':>10} {'Overall':>10} {'AUC drop':>10}",
        ]
        lines.append("-" * 70)
        for name, report in self.reports.items():
            attn = f"{report.attention_leakage.excess:.1%}" if report.attention_leakage else "N/A"
            lines.append(
                f"{name:<25} {report.feature_leakage.excess:>10.1%} "
                f"{attn:>10} {report.performance_leakage.excess:>10.1%} "
                f"{report.overall_leakage_score():>10.3f} {report.auc_degradation:>+10.3f}"
            )
        lines.append("=" * 70)

        # Key finding
        scores = {k: v.overall_leakage_score() for k, v in self.reports.items()}
        best = min(scores, key=scores.get)
        worst = max(scores, key=scores.get)
        if scores[worst] - scores[best] > 0.05:
            lines.append(f"\n📊 KEY FINDING: {best} shows {scores[worst]-scores[best]:.1%} less")
            lines.append(f"   site leakage than {worst}.")
            if "federated" in best.lower():
                lines.append(
                    "   Federated training is structurally more resistant to site shortcuts."
                )
                lines.append(
                    "   This supports federated learning beyond privacy — as a debiasing mechanism."
                )
        else:
            lines.append(
                "\n📊 KEY FINDING: No significant difference in site leakage across regimes."
            )
            lines.append("   All training approaches show similar shortcut learning behavior.")
        return "\n".join(lines)


# ── Core auditor ──────────────────────────────────────────────────────────────


class SiteLeakageAuditor:
    """
    Audits pathology AI models for site-level leakage.

    Site leakage occurs when a model learns to identify the hospital/scanner
    that produced a slide rather than the underlying pathological features.
    This inflates reported AUC on within-distribution test sets and causes
    catastrophic failure on new sites.

    Three mechanisms are tested:
        1. Feature leakage: Can a linear probe predict site from model embeddings?
        2. Attention leakage: Do attention weights encode site identity?
        3. Performance leakage: Does AUC drop when site-correlated features are removed?
    """

    def __init__(
        self,
        model: nn.Module,
        model_name: str = "model",
        get_embeddings: Optional[Callable] = None,
        get_attention: Optional[Callable] = None,
        device: str = "cpu",
    ):
        """
        Args:
            model: Trained pathology model
            model_name: Name for reporting
            get_embeddings: fn(model, features) -> embedding tensor. If None,
                            assumes model returns (logits, attention) tuple.
            get_attention: fn(model, features) -> attention weights. If None,
                           assumes model returns (logits, attention) tuple.
            device: torch device
        """
        self.model = model.to(device)
        self.model_name = model_name
        self.device = device

        self._get_embeddings = get_embeddings or self._default_embeddings
        self._get_attention = get_attention or self._default_attention

    def _default_embeddings(self, model: nn.Module, features: torch.Tensor) -> torch.Tensor:
        """Extract slide-level embedding from (logits, attention) model."""
        model.eval()
        with torch.no_grad():
            logits, attention = model(features.to(self.device))
            # Weighted aggregation = slide embedding
            embedding = (attention.unsqueeze(-1) * features.to(self.device)).sum(0)
        return embedding.cpu()

    def _default_attention(self, model: nn.Module, features: torch.Tensor) -> torch.Tensor:
        model.eval()
        with torch.no_grad():
            _, attention = model(features.to(self.device))
        return attention.cpu()

    # ── Leakage mechanisms ────────────────────────────────────────────────────

    def _feature_leakage(self, slides_by_site: Dict[int, List]) -> Tuple[float, np.ndarray]:
        """
        Train linear probe to predict site from slide embeddings.
        Returns (accuracy, embeddings_array).
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold, cross_val_score

        X, y = [], []
        for site_id, slides in slides_by_site.items():
            for slide in slides:
                emb = self._get_embeddings(self.model, slide.features)
                X.append(emb.numpy())
                y.append(site_id)

        X, y = np.array(X), np.array(y)
        probe = LogisticRegression(max_iter=500, random_state=42, C=1.0)
        cv = StratifiedKFold(n_splits=min(3, len(set(y))), shuffle=True, random_state=42)
        try:
            scores = cross_val_score(probe, X, y, cv=cv, scoring="accuracy")
            return float(scores.mean()), X
        except Exception as e:
            logger.warning(f"Feature leakage probe failed: {e}")
            return 1.0 / len(slides_by_site), np.array(X)

    def _attention_leakage(self, slides_by_site: Dict[int, List]) -> Tuple[float, float]:
        """
        Train linear probe to predict site from attention weights.
        Also computes cross-site attention correlation.
        Returns (site_pred_accuracy, cross_site_correlation).
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold, cross_val_score

        X, y = [], []
        site_mean_attentions = {}

        for site_id, slides in slides_by_site.items():
            site_attentions = []
            for slide in slides:
                attn = self._get_attention(self.model, slide.features)
                X.append(attn.numpy())
                y.append(site_id)
                site_attentions.append(attn.numpy())
            site_mean_attentions[site_id] = np.mean(site_attentions, axis=0)

        X, y = np.array(X), np.array(y)

        # Site predictability
        probe = LogisticRegression(max_iter=500, random_state=42)
        cv = StratifiedKFold(n_splits=min(3, len(set(y))), shuffle=True, random_state=42)
        try:
            scores = cross_val_score(probe, X, y, cv=cv, scoring="accuracy")
            site_pred = float(scores.mean())
        except Exception as e:
            logger.warning(f"Site prediction failed, using baseline: {e}")
            site_pred = 1.0 / len(slides_by_site)

        # Cross-site attention correlation
        site_ids = list(site_mean_attentions.keys())
        correlations = []
        for i in range(len(site_ids)):
            for j in range(i + 1, len(site_ids)):
                a = site_mean_attentions[site_ids[i]]
                b = site_mean_attentions[site_ids[j]]
                if a.std() > 1e-6 and b.std() > 1e-6:
                    corr = float(np.corrcoef(a, b)[0, 1])
                    correlations.append(corr)

        cross_site_corr = float(np.mean(correlations)) if correlations else 0.0
        return site_pred, cross_site_corr

    def _performance_leakage(
        self, slides_by_site: Dict[int, List], embeddings: np.ndarray
    ) -> Tuple[float, float, float]:
        """
        Measure AUC with vs without site-correlated dimensions removed.
        Uses PCA to identify site-predictive directions, then projects them out.
        Returns (auc_with, auc_without, degradation).
        """
        from sklearn.decomposition import PCA
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import StratifiedKFold, cross_val_predict

        all_slides = [s for slides in slides_by_site.values() for s in slides]
        y_labels = np.array([s.label for s in all_slides])
        site_labels = np.array([s.site_id for s in all_slides])
        X = embeddings

        if len(set(y_labels)) < 2:
            return 0.5, 0.5, 0.0

        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        clf = LogisticRegression(max_iter=500, random_state=42)

        try:
            # AUC with full embeddings (includes site info)
            probs_with = cross_val_predict(clf, X, y_labels, cv=cv, method="predict_proba")
            auc_with = roc_auc_score(y_labels, probs_with[:, 1])
        except Exception as e:
            logger.warning(f"AUC calculation with site info failed: {e}")
            auc_with = 0.5

        try:
            # Remove site-correlated directions via PCA on site labels
            # Find top-k PCA components that are most correlated with site
            pca = PCA(n_components=min(20, X.shape[1] - 1))
            X_pca = pca.fit_transform(X)

            # Identify site-correlated components
            site_correlations = [
                abs(np.corrcoef(X_pca[:, i], site_labels)[0, 1]) for i in range(X_pca.shape[1])
            ]
            # Remove top-3 most site-correlated components
            site_dims = np.argsort(site_correlations)[-3:]
            keep_dims = [i for i in range(X_pca.shape[1]) if i not in site_dims]
            X_debiased = X_pca[:, keep_dims]

            probs_without = cross_val_predict(
                clf, X_debiased, y_labels, cv=cv, method="predict_proba"
            )
            auc_without = roc_auc_score(y_labels, probs_without[:, 1])
        except Exception as e:
            logger.warning(f"AUC calculation without site info failed: {e}")
            auc_without = auc_with

        return auc_with, auc_without, auc_without - auc_with

    def _per_site_metrics(
        self, slides_by_site: Dict[int, List]
    ) -> Tuple[Dict[int, float], Dict[int, float]]:
        """Compute per-site AUC and accuracy."""
        from sklearn.metrics import roc_auc_score

        per_site_auc = {}
        per_site_acc = {}

        self.model.eval()
        with torch.no_grad():
            for site_id, slides in slides_by_site.items():
                preds, labels, probs = [], [], []
                for slide in slides:
                    logits, _ = self.model(slide.features.to(self.device))
                    prob = torch.softmax(logits, dim=0)[1].item()
                    pred = int(logits.argmax().item())
                    preds.append(pred)
                    labels.append(slide.label)
                    probs.append(prob)

                per_site_acc[site_id] = float(np.mean(np.array(preds) == np.array(labels)))
                try:
                    if len(set(labels)) > 1:
                        per_site_auc[site_id] = roc_auc_score(labels, probs)
                    else:
                        per_site_auc[site_id] = float("nan")
                except Exception as e:
                    logger.debug(f"AUC calculation failed for site {site_id}: {e}")
                    per_site_auc[site_id] = float("nan")

        return per_site_auc, per_site_acc

    # ── Public API ────────────────────────────────────────────────────────────

    def audit(self, slides_by_site: Dict[int, List]) -> SiteLeakageReport:
        """
        Run full site leakage audit.

        Args:
            slides_by_site: Dict mapping site_id -> list of SlideData objects.
                            Each SlideData must have: .features, .label, .site_id

        Returns:
            SiteLeakageReport with all leakage metrics
        """
        num_sites = len(slides_by_site)
        num_slides = sum(len(s) for s in slides_by_site.values())
        logger.info(f"Auditing {self.model_name}: {num_slides} slides, {num_sites} sites")

        # 1. Feature leakage
        logger.info("  Computing feature leakage...")
        feat_score, embeddings = self._feature_leakage(slides_by_site)
        feature_leakage = LeakageScore.compute(
            "feature",
            feat_score,
            num_sites,
            "Linear probe accuracy predicting site from slide embeddings",
        )

        # 2. Attention leakage (if model returns attention)
        attention_leakage = None
        cross_site_corr = None
        try:
            logger.info("  Computing attention leakage...")
            attn_score, cross_site_corr = self._attention_leakage(slides_by_site)
            attention_leakage = LeakageScore.compute(
                "attention",
                attn_score,
                num_sites,
                "Linear probe accuracy predicting site from attention weights",
            )
        except Exception as e:
            logger.warning(f"  Attention leakage skipped: {e}")

        # 3. Performance leakage
        logger.info("  Computing performance leakage...")
        auc_with, auc_without, auc_degradation = self._performance_leakage(
            slides_by_site, embeddings
        )
        # Performance leakage = how much AUC drops when site info removed
        # Normalize: large drop = high leakage
        perf_score = max(0.0, -auc_degradation)  # degradation is negative
        performance_leakage = LeakageScore.compute(
            "performance",
            min(1.0, perf_score * 5),  # scale to 0-1
            0.0,
            "AUC degradation when site-correlated embedding dimensions removed",
        )

        # 4. Per-site metrics
        logger.info("  Computing per-site metrics...")
        per_site_auc, per_site_acc = self._per_site_metrics(slides_by_site)

        return SiteLeakageReport(
            model_name=self.model_name,
            num_sites=num_sites,
            num_slides=num_slides,
            feature_leakage=feature_leakage,
            attention_leakage=attention_leakage,
            performance_leakage=performance_leakage,
            auc_with_site_info=auc_with,
            auc_without_site_info=auc_without,
            auc_degradation=auc_degradation,
            per_site_auc=per_site_auc,
            per_site_accuracy=per_site_acc,
            cross_site_attention_correlation=cross_site_corr,
        )

    @staticmethod
    def compare_training_regimes(
        models: Dict[str, nn.Module],
        slides_by_site: Dict[int, List],
        device: str = "cpu",
        **auditor_kwargs,
    ) -> RegimeComparison:
        """
        Compare site leakage across multiple training regimes.

        Args:
            models: Dict of name -> trained model
            slides_by_site: Site-split data
            device: torch device

        Returns:
            RegimeComparison with per-regime reports and key finding
        """
        reports = {}
        for name, model in models.items():
            auditor = SiteLeakageAuditor(model, model_name=name, device=device, **auditor_kwargs)
            reports[name] = auditor.audit(slides_by_site)
        return RegimeComparison(reports=reports)
