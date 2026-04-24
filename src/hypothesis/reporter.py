"""
Scientific report generation for hypothesis testing results.

Produces structured Markdown reports and JSON exports summarising:
  - Generated hypotheses
  - Statistical test results
  - Prioritised list by support strength
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


class HypothesisReporter:
    """
    Generates reports from hypothesis generation + testing results.

    Args:
        cancer_type: tissue/cancer type label for report header
        output_dir: directory for saved reports
    """

    def __init__(self, cancer_type: str = "cancer", output_dir: Optional[str] = None):
        self.cancer_type = cancer_type
        self.output_dir = Path(output_dir) if output_dir else Path("hypothesis_reports")

    def _emoji_support(self, supported: bool) -> str:
        return "SUPPORTED" if supported else "NOT SUPPORTED"

    def _confidence_rank(self, confidence: str) -> int:
        return {"HIGH": 0, "MEDIUM": 1, "LOW": 2}.get(confidence.upper(), 1)

    def _sort_hypotheses(
        self,
        hypotheses,
        test_results,
    ) -> List[Tuple]:
        pairs = list(zip(hypotheses, test_results))
        pairs.sort(key=lambda x: (
            not x[1].supported,
            x[1].enrichment_pvalue,
            self._confidence_rank(x[0].confidence),
        ))
        return pairs

    def generate_markdown(
        self,
        hypotheses: List,
        test_results: List,
        title: str = "Hypothesis Report",
    ) -> str:
        now = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
        pairs = self._sort_hypotheses(hypotheses, test_results)

        n_supported = sum(1 for _, r in pairs if r.supported)

        lines = [
            f"# {title}",
            f"**Cancer type:** {self.cancer_type}  ",
            f"**Generated:** {now}  ",
            f"**Hypotheses:** {len(pairs)} total, {n_supported} statistically supported",
            "",
            "---",
            "",
        ]

        for i, (hyp, result) in enumerate(pairs, 1):
            status = self._emoji_support(result.supported)
            lines += [
                f"## Hypothesis {i}: [{status}]",
                "",
                f"**Statement:** {hyp.hypothesis_text}",
                "",
                f"**Confidence:** {hyp.confidence}",
                "",
                f"**Mechanism:** {hyp.mechanism}",
                "",
                f"**Testable prediction:** {hyp.testable_prediction}",
                "",
            ]

            if hyp.affected_genes:
                lines.append(f"**Affected genes:** {', '.join(hyp.affected_genes[:10])}" +
                              (" ..." if len(hyp.affected_genes) > 10 else ""))
                lines.append("")

            if hyp.affected_pathways:
                lines.append(f"**Affected pathways:** {', '.join(hyp.affected_pathways)}")
                lines.append("")

            lines += [
                "**Statistical evidence:**",
                f"- Genes tested: {result.n_genes_tested}",
                f"- Significantly DE (FDR<0.05): {result.n_genes_significant}",
                f"- Enrichment p-value: {result.enrichment_pvalue:.4f}",
                f"- Mean |effect size|: {result.mean_effect_size:.3f}",
                f"- Direction consistent: {'Yes' if result.direction_consistent else 'No'}",
            ]

            if result.notes:
                lines.append(f"- Notes: {result.notes}")

            lines += ["", "---", ""]

        return "\n".join(lines)

    def generate_json(self, hypotheses: List, test_results: List) -> dict:
        pairs = self._sort_hypotheses(hypotheses, test_results)
        return {
            "cancer_type": self.cancer_type,
            "generated_at": datetime.now().isoformat(),
            "n_hypotheses": len(pairs),
            "n_supported": sum(1 for _, r in pairs if r.supported),
            "hypotheses": [
                {
                    "hypothesis": hyp.hypothesis_text,
                    "mechanism": hyp.mechanism,
                    "testable_prediction": hyp.testable_prediction,
                    "affected_genes": hyp.affected_genes,
                    "affected_pathways": hyp.affected_pathways,
                    "confidence": hyp.confidence,
                    "subtype_context": hyp.subtype_context,
                    "factor_context": hyp.factor_context,
                    "test_result": {
                        "supported": result.supported,
                        "n_genes_tested": result.n_genes_tested,
                        "n_genes_significant": result.n_genes_significant,
                        "enrichment_pvalue": result.enrichment_pvalue,
                        "mean_effect_size": result.mean_effect_size,
                        "direction_consistent": result.direction_consistent,
                    },
                }
                for hyp, result in pairs
            ],
        }

    def save(
        self,
        hypotheses: List,
        test_results: List,
        prefix: str = "hypotheses",
    ) -> Tuple[Path, Path]:
        """Save both Markdown and JSON reports. Returns (md_path, json_path)."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        md_path = self.output_dir / f"{prefix}_{ts}.md"
        json_path = self.output_dir / f"{prefix}_{ts}.json"

        md_content = self.generate_markdown(
            hypotheses, test_results, title=f"HistoCore Hypothesis Report — {self.cancer_type}"
        )
        md_path.write_text(md_content, encoding="utf-8")

        json_content = self.generate_json(hypotheses, test_results)
        json_path.write_text(json.dumps(json_content, indent=2), encoding="utf-8")

        logger.info("Saved report: %s, %s", md_path, json_path)
        return md_path, json_path

    def print_summary(self, hypotheses: List, test_results: List) -> None:
        pairs = self._sort_hypotheses(hypotheses, test_results)
        n_supported = sum(1 for _, r in pairs if r.supported)
        print(f"\n{'='*60}")
        print(f"HYPOTHESIS SUMMARY — {self.cancer_type}")
        print(f"{'='*60}")
        print(f"Total: {len(pairs)}  |  Supported: {n_supported}")
        print()
        for i, (hyp, result) in enumerate(pairs, 1):
            status = "[OK]" if result.supported else "[--]"
            print(f"{status} H{i} ({hyp.confidence}): {hyp.hypothesis_text[:80]}...")
            print(f"     enrich_p={result.enrichment_pvalue:.4f}  de_genes={result.n_genes_significant}/{result.n_genes_tested}")
        print()
