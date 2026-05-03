"""Manual approval workflow for benchmark results."""

import logging
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class QAFlag:
    """Quality assurance flag for suspicious results."""
    category: str  # 'anomaly', 'deviation', 'validation', 'error'
    severity: str  # 'info', 'warning', 'critical'
    framework: str
    message: str
    details: Optional[str] = None


class ApprovalWorkflow:
    """Manages manual approval workflow for benchmark results."""
    
    def __init__(self, output_dir: Path):
        """
        Initialize approval workflow.
        
        Args:
            output_dir: Directory for approval reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_approval_report(
        self,
        results: Dict[str, Dict[str, float]],
        qa_flags: List[QAFlag],
        deviation_flags: List,
        error_summary: Optional[str] = None
    ) -> str:
        """
        Generate comprehensive approval report with all QA flags and warnings.
        
        Args:
            results: Benchmark results dict
            qa_flags: List of QA flags
            deviation_flags: List of historical deviation flags
            error_summary: Optional error summary text
            
        Returns:
            Formatted approval report
        """
        report = []
        report.append("# Benchmark Results Approval Report")
        report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Executive summary
        report.append("## Executive Summary\n")
        report.append(f"- Frameworks tested: {len(results)}")
        report.append(f"- QA flags: {len(qa_flags)}")
        report.append(f"- Historical deviations: {len(deviation_flags)}")
        
        critical_flags = [f for f in qa_flags if f.severity == 'critical']
        warning_flags = [f for f in qa_flags if f.severity == 'warning']
        report.append(f"- Critical issues: {len(critical_flags)}")
        report.append(f"- Warnings: {len(warning_flags)}\n")
        
        # Results summary
        report.append("## Results Summary\n")
        report.append("| Framework | Accuracy | Loss | Training Time (s) | GPU Memory (GB) |")
        report.append("|-----------|----------|------|-------------------|-----------------|")
        
        for framework, metrics in results.items():
            report.append(
                f"| {framework} | "
                f"{metrics.get('accuracy', 0.0):.4f} | "
                f"{metrics.get('loss', 0.0):.4f} | "
                f"{metrics.get('training_time', 0.0):.1f} | "
                f"{metrics.get('gpu_memory_peak', 0.0):.2f} |"
            )
        report.append("")
        
        # QA Flags
        if qa_flags:
            report.append("## Quality Assurance Flags\n")
            
            if critical_flags:
                report.append(f"### Critical Issues ({len(critical_flags)})\n")
                for flag in critical_flags:
                    report.append(self._format_qa_flag(flag))
                    
            if warning_flags:
                report.append(f"### Warnings ({len(warning_flags)})\n")
                for flag in warning_flags:
                    report.append(self._format_qa_flag(flag))
                    
            info_flags = [f for f in qa_flags if f.severity == 'info']
            if info_flags:
                report.append(f"### Informational ({len(info_flags)})\n")
                for flag in info_flags:
                    report.append(self._format_qa_flag(flag))
        else:
            report.append("## Quality Assurance Flags\n")
            report.append("No QA flags detected. Results appear normal.\n")
            
        # Historical deviations
        if deviation_flags:
            report.append("## Historical Deviations\n")
            for flag in deviation_flags:
                direction = "increased" if flag.deviation_percent > 0 else "decreased"
                report.append(
                    f"- **{flag.framework}** - {flag.metric}: "
                    f"{direction} by {abs(flag.deviation_percent):.1f}% "
                    f"({flag.severity})\n"
                )
        else:
            report.append("## Historical Deviations\n")
            report.append("No significant deviations from historical baselines.\n")
            
        # Error summary
        if error_summary:
            report.append("## Error Summary\n")
            report.append(error_summary)
            report.append("")
            
        # Approval decision
        report.append("## Approval Decision\n")
        
        if critical_flags:
            report.append("**Status: REQUIRES REVIEW**\n")
            report.append(
                f"Critical issues detected ({len(critical_flags)}). "
                "Manual review required before updating PERFORMANCE_COMPARISON.md.\n"
            )
        elif warning_flags or deviation_flags:
            report.append("**Status: REVIEW RECOMMENDED**\n")
            report.append(
                f"Warnings or deviations detected ({len(warning_flags + deviation_flags)}). "
                "Review recommended before updating PERFORMANCE_COMPARISON.md.\n"
            )
        else:
            report.append("**Status: APPROVED**\n")
            report.append(
                "No critical issues or warnings detected. "
                "Results can be applied to PERFORMANCE_COMPARISON.md.\n"
            )
            
        # Next steps
        report.append("## Next Steps\n")
        report.append("1. Review this report carefully")
        report.append("2. Investigate any critical issues or warnings")
        report.append("3. If approved, run: `python experiments/benchmark_system/run_benchmark.py approve`")
        report.append("4. If rejected, re-run benchmark or investigate issues\n")
        
        return "\n".join(report)
        
    def _format_qa_flag(self, flag: QAFlag) -> str:
        """Format QA flag as string."""
        lines = [f"**[{flag.severity.upper()}]** {flag.framework} - {flag.category}"]
        lines.append(f"  - {flag.message}")
        if flag.details:
            lines.append(f"  - Details: {flag.details}")
        lines.append("")
        return "\n".join(lines)
        
    def request_approval(
        self,
        approval_report: str,
        interactive: bool = True
    ) -> bool:
        """
        Request user approval for benchmark results.
        
        Args:
            approval_report: Formatted approval report
            interactive: If True, prompt user for approval
            
        Returns:
            True if approved, False otherwise
        """
        # Save approval report
        report_file = self.output_dir / "approval_report.md"
        with open(report_file, 'w') as f:
            f.write(approval_report)
        logger.info(f"Approval report saved to {report_file}")
        
        if not interactive:
            logger.info("Non-interactive mode: approval required manually")
            return False
            
        # Print report
        print("\n" + "="*80)
        print(approval_report)
        print("="*80 + "\n")
        
        # Prompt for approval
        while True:
            response = input("Approve these results? (yes/no): ").strip().lower()
            if response in ['yes', 'y']:
                logger.info("Results approved by user")
                return True
            elif response in ['no', 'n']:
                logger.info("Results rejected by user")
                return False
            else:
                print("Please enter 'yes' or 'no'")
                
    def apply_approved_results(
        self,
        results: Dict[str, Dict[str, float]],
        performance_comparison_path: Path
    ):
        """
        Update PERFORMANCE_COMPARISON.md with approved results.
        
        Args:
            results: Approved benchmark results
            performance_comparison_path: Path to PERFORMANCE_COMPARISON.md
        """
        if not performance_comparison_path.exists():
            logger.error(f"PERFORMANCE_COMPARISON.md not found at {performance_comparison_path}")
            return
            
        # Read existing file
        with open(performance_comparison_path, 'r') as f:
            content = f.read()
            
        # Generate updated results table
        table_lines = [
            "| Framework | Accuracy | Loss | Training Time (s) | GPU Memory (GB) |",
            "|-----------|----------|------|-------------------|-----------------|"
        ]
        
        for framework, metrics in results.items():
            table_lines.append(
                f"| {framework} | "
                f"{metrics.get('accuracy', 0.0):.4f} | "
                f"{metrics.get('loss', 0.0):.4f} | "
                f"{metrics.get('training_time', 0.0):.1f} | "
                f"{metrics.get('gpu_memory_peak', 0.0):.2f} |"
            )
            
        new_table = "\n".join(table_lines)
        
        # Replace placeholder or existing table
        # Look for table marker
        marker = "<!-- BENCHMARK_RESULTS_TABLE -->"
        if marker in content:
            # Replace content after marker until next section
            parts = content.split(marker)
            if len(parts) >= 2:
                # Find next section (starts with ##)
                after_marker = parts[1]
                next_section_idx = after_marker.find("\n##")
                if next_section_idx != -1:
                    updated_content = (
                        parts[0] + marker + "\n\n" + new_table + "\n" +
                        after_marker[next_section_idx:]
                    )
                else:
                    updated_content = parts[0] + marker + "\n\n" + new_table + "\n"
            else:
                updated_content = content + "\n\n" + marker + "\n\n" + new_table + "\n"
        else:
            # Append to end
            updated_content = content + "\n\n" + marker + "\n\n" + new_table + "\n"
            
        # Write updated file
        with open(performance_comparison_path, 'w') as f:
            f.write(updated_content)
            
        logger.info(f"Updated {performance_comparison_path} with approved results")
        
        # Create backup
        backup_path = performance_comparison_path.with_suffix('.md.backup')
        with open(backup_path, 'w') as f:
            f.write(content)
        logger.info(f"Backup saved to {backup_path}")
