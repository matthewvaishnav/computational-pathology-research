"""
Clinical reporting system for generating physician-friendly diagnostic reports.

This module provides:
- Configurable report templates for different specialties
- Report generation with diagnosis, probabilities, uncertainty, and recommendations
- Multiple export formats (PDF, HTML, FHIR, DICOM SR)
- Longitudinal progression summaries
- Attention visualization integration
"""

import base64
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from jinja2 import Environment, FileSystemLoader, Template


class ReportSpecialty(Enum):
    """Medical specialties for report templates."""

    CARDIOLOGY = "cardiology"
    ONCOLOGY = "oncology"
    RADIOLOGY = "radiology"
    PATHOLOGY = "pathology"
    GENERAL = "general"


class ExportFormat(Enum):
    """Supported report export formats."""

    PDF = "pdf"
    HTML = "html"
    FHIR = "fhir"
    DICOM_SR = "dicom_sr"
    JSON = "json"


@dataclass
class DiagnosisResult:
    """Structured diagnosis result for reporting."""

    disease_id: str
    disease_name: str
    probability: float
    confidence_interval: Optional[tuple[float, float]] = None
    uncertainty_score: Optional[float] = None


@dataclass
class ReportData:
    """Complete data for clinical report generation."""

    # Patient and scan information
    patient_id: str
    scan_id: str
    scan_date: datetime

    # Primary diagnosis
    primary_diagnosis: DiagnosisResult

    # Alternative diagnoses (top-k)
    alternative_diagnoses: List[DiagnosisResult] = field(default_factory=list)

    # Probability distribution across all disease states
    probability_distribution: Dict[str, float] = field(default_factory=dict)

    # Uncertainty quantification
    uncertainty_explanation: Optional[str] = None
    ood_detected: bool = False
    ood_explanation: Optional[str] = None

    # Risk analysis
    risk_scores: Optional[Dict[str, float]] = None
    risk_time_horizons: Optional[List[str]] = None

    # Clinical recommendations
    recommendations: List[str] = field(default_factory=list)

    # Attention visualization data
    attention_weights: Optional[np.ndarray] = None
    attention_heatmap_path: Optional[str] = None

    # Longitudinal data
    longitudinal_summary: Optional[Dict[str, Any]] = None
    previous_scan_comparison: Optional[Dict[str, Any]] = None

    # Metadata
    model_version: str = "1.0.0"
    report_timestamp: datetime = field(default_factory=datetime.now)

    # Physician annotations
    physician_notes: Optional[str] = None
    amendments: List[Dict[str, Any]] = field(default_factory=list)


class ClinicalReportGenerator:
    """
    Generate clinical reports from model predictions.

    Supports configurable templates for different specialties and multiple
    export formats including PDF, HTML, and structured formats (FHIR, DICOM SR).
    """

    def __init__(
        self,
        template_dir: Optional[Union[str, Path]] = None,
        default_specialty: ReportSpecialty = ReportSpecialty.PATHOLOGY,
    ):
        """
        Initialize the clinical report generator.

        Args:
            template_dir: Directory containing Jinja2 templates. If None, uses built-in templates.
            default_specialty: Default specialty for report generation.
        """
        self.default_specialty = default_specialty

        # Set up Jinja2 environment
        if template_dir is not None:
            template_dir = Path(template_dir)
            if not template_dir.exists():
                raise ValueError(f"Template directory does not exist: {template_dir}")
            self.jinja_env = Environment(loader=FileSystemLoader(str(template_dir)))
        else:
            # Use built-in templates
            self.jinja_env = Environment(loader=FileSystemLoader(searchpath="./"))

        # Built-in templates as strings
        self._builtin_templates = self._create_builtin_templates()

    def _create_builtin_templates(self) -> Dict[str, str]:
        """Create built-in report templates for different specialties."""

        # Base template structure
        base_html = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Clinical Pathology Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }
        .header { border-bottom: 2px solid #333; padding-bottom: 10px; margin-bottom: 20px; }
        .section { margin: 20px 0; }
        .section-title { font-size: 18px; font-weight: bold; color: #2c3e50; margin-bottom: 10px; }
        .diagnosis { background-color: #e8f4f8; padding: 15px; border-left: 4px solid #3498db; margin: 10px 0; }
        .primary { background-color: #d5f4e6; border-left-color: #27ae60; }
        .warning { background-color: #fff3cd; border-left: 4px solid #ffc107; padding: 10px; margin: 10px 0; }
        .probability-bar { background-color: #ecf0f1; height: 20px; border-radius: 4px; overflow: hidden; margin: 5px 0; }
        .probability-fill { background-color: #3498db; height: 100%; transition: width 0.3s; }
        .metadata { font-size: 12px; color: #7f8c8d; margin-top: 30px; padding-top: 20px; border-top: 1px solid #bdc3c7; }
        table { border-collapse: collapse; width: 100%; margin: 10px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #34495e; color: white; }
        .attention-viz { max-width: 100%; margin: 15px 0; }
    </style>
</head>
<body>
    <div class="header">
        <h1>{{ specialty_title }} - Computational Pathology Report</h1>
        <p><strong>Patient ID:</strong> {{ patient_id }}</p>
        <p><strong>Scan ID:</strong> {{ scan_id }}</p>
        <p><strong>Scan Date:</strong> {{ scan_date.strftime('%Y-%m-%d %H:%M:%S') }}</p>
        <p><strong>Report Generated:</strong> {{ report_timestamp.strftime('%Y-%m-%d %H:%M:%S') }}</p>
    </div>

    {% if ood_detected %}
    <div class="warning">
        <strong>⚠ Out-of-Distribution Detection:</strong> {{ ood_explanation }}
        <br><strong>Recommendation:</strong> Expert pathologist review recommended before clinical use.
    </div>
    {% endif %}

    <div class="section">
        <div class="section-title">Primary Diagnosis</div>
        <div class="diagnosis primary">
            <h3>{{ primary_diagnosis.disease_name }}</h3>
            <p><strong>Probability:</strong> {{ "%.1f"|format(primary_diagnosis.probability * 100) }}%</p>
            {% if primary_diagnosis.confidence_interval %}
            <p><strong>95% Confidence Interval:</strong>
               {{ "%.1f"|format(primary_diagnosis.confidence_interval[0] * 100) }}% -
               {{ "%.1f"|format(primary_diagnosis.confidence_interval[1] * 100) }}%</p>
            {% endif %}
            {% if primary_diagnosis.uncertainty_score %}
            <p><strong>Uncertainty Score:</strong> {{ "%.3f"|format(primary_diagnosis.uncertainty_score) }}</p>
            {% endif %}
        </div>
    </div>

    {% if alternative_diagnoses %}
    <div class="section">
        <div class="section-title">Alternative Diagnoses</div>
        {% for diagnosis in alternative_diagnoses %}
        <div class="diagnosis">
            <strong>{{ diagnosis.disease_name }}</strong>: {{ "%.1f"|format(diagnosis.probability * 100) }}%
            <div class="probability-bar">
                <div class="probability-fill" style="width: {{ diagnosis.probability * 100 }}%"></div>
            </div>
        </div>
        {% endfor %}
    </div>
    {% endif %}

    <div class="section">
        <div class="section-title">Probability Distribution</div>
        <table>
            <tr><th>Disease State</th><th>Probability</th></tr>
            {% for disease, prob in probability_distribution.items() %}
            <tr>
                <td>{{ disease }}</td>
                <td>{{ "%.2f"|format(prob * 100) }}%</td>
            </tr>
            {% endfor %}
        </table>
    </div>

    {% if uncertainty_explanation %}
    <div class="section">
        <div class="section-title">Uncertainty Quantification</div>
        <p>{{ uncertainty_explanation }}</p>
    </div>
    {% endif %}

    {% if risk_scores %}
    <div class="section">
        <div class="section-title">Risk Analysis</div>
        <table>
            <tr><th>Disease State</th><th>Risk Score</th></tr>
            {% for disease, score in risk_scores.items() %}
            <tr>
                <td>{{ disease }}</td>
                <td>{{ "%.2f"|format(score * 100) }}%</td>
            </tr>
            {% endfor %}
        </table>
    </div>
    {% endif %}

    {% if recommendations %}
    <div class="section">
        <div class="section-title">Clinical Recommendations</div>
        <ul>
        {% for rec in recommendations %}
            <li>{{ rec }}</li>
        {% endfor %}
        </ul>
    </div>
    {% endif %}

    {% if attention_heatmap_path %}
    <div class="section">
        <div class="section-title">Attention Visualization</div>
        <p>The following heatmap shows tissue regions that influenced the diagnosis:</p>
        <img src="{{ attention_heatmap_path }}" alt="Attention Heatmap" class="attention-viz">
    </div>
    {% endif %}

    {% if longitudinal_summary %}
    <div class="section">
        <div class="section-title">Longitudinal Progression Summary</div>
        <p><strong>Number of Previous Scans:</strong> {{ longitudinal_summary.num_scans }}</p>
        <p><strong>Timeline Duration:</strong> {{ longitudinal_summary.duration_days }} days</p>
        {% if longitudinal_summary.progression_trend %}
        <p><strong>Progression Trend:</strong> {{ longitudinal_summary.progression_trend }}</p>
        {% endif %}
        {% if previous_scan_comparison %}
        <p><strong>Change from Previous Scan:</strong></p>
        <ul>
            <li>Previous Diagnosis: {{ previous_scan_comparison.previous_diagnosis }}</li>
            <li>Current Diagnosis: {{ previous_scan_comparison.current_diagnosis }}</li>
            <li>Probability Change: {{ previous_scan_comparison.probability_change }}</li>
        </ul>
        {% endif %}
    </div>
    {% endif %}

    {% if physician_notes %}
    <div class="section">
        <div class="section-title">Physician Notes</div>
        <p>{{ physician_notes }}</p>
    </div>
    {% endif %}

    {% if amendments %}
    <div class="section">
        <div class="section-title">Report Amendments</div>
        {% for amendment in amendments %}
        <div style="margin: 10px 0; padding: 10px; background-color: #f8f9fa; border-left: 3px solid #6c757d;">
            <p><strong>Date:</strong> {{ amendment.timestamp }}</p>
            <p><strong>Amended By:</strong> {{ amendment.user }}</p>
            <p><strong>Note:</strong> {{ amendment.note }}</p>
        </div>
        {% endfor %}
    </div>
    {% endif %}

    <div class="metadata">
        <p><strong>Model Version:</strong> {{ model_version }}</p>
        <p><strong>Report ID:</strong> {{ scan_id }}_{{ report_timestamp.strftime('%Y%m%d%H%M%S') }}</p>
        <p><em>This report was generated by an AI-assisted diagnostic system and should be reviewed by a qualified pathologist.</em></p>
    </div>
</body>
</html>
"""

        return {
            ReportSpecialty.PATHOLOGY.value: base_html,
            ReportSpecialty.ONCOLOGY.value: base_html.replace("{{ specialty_title }}", "Oncology"),
            ReportSpecialty.CARDIOLOGY.value: base_html.replace(
                "{{ specialty_title }}", "Cardiology"
            ),
            ReportSpecialty.RADIOLOGY.value: base_html.replace(
                "{{ specialty_title }}", "Radiology"
            ),
            ReportSpecialty.GENERAL.value: base_html.replace(
                "{{ specialty_title }}", "General Pathology"
            ),
        }

    def generate_report(
        self,
        report_data: ReportData,
        specialty: Optional[ReportSpecialty] = None,
    ) -> str:
        """
        Generate HTML report from report data.

        Args:
            report_data: Complete report data including diagnosis, uncertainty, etc.
            specialty: Medical specialty for template selection. Uses default if None.

        Returns:
            HTML report as string.
        """
        if specialty is None:
            specialty = self.default_specialty

        # Get template
        template_str = self._builtin_templates.get(
            specialty.value, self._builtin_templates[ReportSpecialty.PATHOLOGY.value]
        )
        template = Template(template_str)

        # Prepare template context
        context = {
            "specialty_title": specialty.value.title(),
            "patient_id": report_data.patient_id,
            "scan_id": report_data.scan_id,
            "scan_date": report_data.scan_date,
            "report_timestamp": report_data.report_timestamp,
            "primary_diagnosis": report_data.primary_diagnosis,
            "alternative_diagnoses": report_data.alternative_diagnoses,
            "probability_distribution": report_data.probability_distribution,
            "uncertainty_explanation": report_data.uncertainty_explanation,
            "ood_detected": report_data.ood_detected,
            "ood_explanation": report_data.ood_explanation,
            "risk_scores": report_data.risk_scores,
            "risk_time_horizons": report_data.risk_time_horizons,
            "recommendations": report_data.recommendations,
            "attention_heatmap_path": report_data.attention_heatmap_path,
            "longitudinal_summary": report_data.longitudinal_summary,
            "previous_scan_comparison": report_data.previous_scan_comparison,
            "physician_notes": report_data.physician_notes,
            "amendments": report_data.amendments,
            "model_version": report_data.model_version,
        }

        # Render template
        html_report = template.render(**context)
        return html_report

    def export_report(
        self,
        report_data: ReportData,
        output_path: Union[str, Path],
        export_format: ExportFormat = ExportFormat.HTML,
        specialty: Optional[ReportSpecialty] = None,
    ) -> Path:
        """
        Export report to specified format.

        Args:
            report_data: Complete report data.
            output_path: Output file path.
            export_format: Export format (PDF, HTML, FHIR, DICOM_SR, JSON).
            specialty: Medical specialty for template selection.

        Returns:
            Path to exported report file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if export_format == ExportFormat.HTML:
            return self._export_html(report_data, output_path, specialty)
        elif export_format == ExportFormat.PDF:
            return self._export_pdf(report_data, output_path, specialty)
        elif export_format == ExportFormat.JSON:
            return self._export_json(report_data, output_path)
        elif export_format == ExportFormat.FHIR:
            return self._export_fhir(report_data, output_path)
        elif export_format == ExportFormat.DICOM_SR:
            return self._export_dicom_sr(report_data, output_path)
        else:
            raise ValueError(f"Unsupported export format: {export_format}")

    def _export_html(
        self,
        report_data: ReportData,
        output_path: Path,
        specialty: Optional[ReportSpecialty] = None,
    ) -> Path:
        """Export report as HTML file."""
        html_content = self.generate_report(report_data, specialty)
        output_path.write_text(html_content, encoding="utf-8")
        return output_path

    def _export_pdf(
        self,
        report_data: ReportData,
        output_path: Path,
        specialty: Optional[ReportSpecialty] = None,
    ) -> Path:
        """
        Export report as PDF file.

        Uses reportlab for PDF generation.
        """
        try:
            from reportlab.lib import colors
            from reportlab.lib.pagesizes import letter
            from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
            from reportlab.lib.units import inch
            from reportlab.platypus import (
                Image,
                Paragraph,
                SimpleDocTemplate,
                Spacer,
                Table,
                TableStyle,
            )
        except ImportError:
            raise ImportError(
                "reportlab is required for PDF export. Install with: pip install reportlab"
            )

        # Create PDF document
        doc = SimpleDocTemplate(str(output_path), pagesize=letter)
        story = []
        styles = getSampleStyleSheet()

        # Custom styles
        title_style = ParagraphStyle(
            "CustomTitle",
            parent=styles["Heading1"],
            fontSize=18,
            textColor=colors.HexColor("#2c3e50"),
            spaceAfter=12,
        )
        heading_style = ParagraphStyle(
            "CustomHeading",
            parent=styles["Heading2"],
            fontSize=14,
            textColor=colors.HexColor("#2c3e50"),
            spaceAfter=10,
        )

        # Header
        specialty_name = (specialty or self.default_specialty).value.title()
        story.append(Paragraph(f"{specialty_name} - Computational Pathology Report", title_style))
        story.append(Spacer(1, 0.2 * inch))

        # Patient information
        patient_info = [
            ["Patient ID:", report_data.patient_id],
            ["Scan ID:", report_data.scan_id],
            ["Scan Date:", report_data.scan_date.strftime("%Y-%m-%d %H:%M:%S")],
            ["Report Generated:", report_data.report_timestamp.strftime("%Y-%m-%d %H:%M:%S")],
        ]
        patient_table = Table(patient_info, colWidths=[2 * inch, 4 * inch])
        patient_table.setStyle(
            TableStyle(
                [
                    ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 10),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )
        story.append(patient_table)
        story.append(Spacer(1, 0.3 * inch))

        # OOD warning
        if report_data.ood_detected:
            warning_text = (
                f"<b>⚠ Out-of-Distribution Detection:</b> {report_data.ood_explanation}<br/>"
                f"<b>Recommendation:</b> Expert pathologist review recommended before clinical use."
            )
            warning_para = Paragraph(warning_text, styles["Normal"])
            story.append(warning_para)
            story.append(Spacer(1, 0.2 * inch))

        # Primary diagnosis
        story.append(Paragraph("Primary Diagnosis", heading_style))
        primary = report_data.primary_diagnosis
        primary_text = (
            f"<b>{primary.disease_name}</b><br/>" f"Probability: {primary.probability * 100:.1f}%"
        )
        if primary.confidence_interval:
            ci_low, ci_high = primary.confidence_interval
            primary_text += (
                f"<br/>95% Confidence Interval: {ci_low * 100:.1f}% - {ci_high * 100:.1f}%"
            )
        if primary.uncertainty_score:
            primary_text += f"<br/>Uncertainty Score: {primary.uncertainty_score:.3f}"
        story.append(Paragraph(primary_text, styles["Normal"]))
        story.append(Spacer(1, 0.2 * inch))

        # Alternative diagnoses
        if report_data.alternative_diagnoses:
            story.append(Paragraph("Alternative Diagnoses", heading_style))
            alt_data = [["Disease", "Probability"]]
            for diag in report_data.alternative_diagnoses:
                alt_data.append([diag.disease_name, f"{diag.probability * 100:.1f}%"])
            alt_table = Table(alt_data, colWidths=[4 * inch, 2 * inch])
            alt_table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#34495e")),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("FONTSIZE", (0, 0), (-1, -1), 10),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                        ("GRID", (0, 0), (-1, -1), 1, colors.grey),
                    ]
                )
            )
            story.append(alt_table)
            story.append(Spacer(1, 0.2 * inch))

        # Probability distribution
        story.append(Paragraph("Probability Distribution", heading_style))
        prob_data = [["Disease State", "Probability"]]
        for disease, prob in report_data.probability_distribution.items():
            prob_data.append([disease, f"{prob * 100:.2f}%"])
        prob_table = Table(prob_data, colWidths=[4 * inch, 2 * inch])
        prob_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#34495e")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 10),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                    ("GRID", (0, 0), (-1, -1), 1, colors.grey),
                ]
            )
        )
        story.append(prob_table)
        story.append(Spacer(1, 0.2 * inch))

        # Uncertainty explanation
        if report_data.uncertainty_explanation:
            story.append(Paragraph("Uncertainty Quantification", heading_style))
            story.append(Paragraph(report_data.uncertainty_explanation, styles["Normal"]))
            story.append(Spacer(1, 0.2 * inch))

        # Risk analysis
        if report_data.risk_scores:
            story.append(Paragraph("Risk Analysis", heading_style))
            risk_data = [["Disease State", "Risk Score"]]
            for disease, score in report_data.risk_scores.items():
                risk_data.append([disease, f"{score * 100:.2f}%"])
            risk_table = Table(risk_data, colWidths=[4 * inch, 2 * inch])
            risk_table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#34495e")),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("FONTSIZE", (0, 0), (-1, -1), 10),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                        ("GRID", (0, 0), (-1, -1), 1, colors.grey),
                    ]
                )
            )
            story.append(risk_table)
            story.append(Spacer(1, 0.2 * inch))

        # Recommendations
        if report_data.recommendations:
            story.append(Paragraph("Clinical Recommendations", heading_style))
            for rec in report_data.recommendations:
                story.append(Paragraph(f"• {rec}", styles["Normal"]))
            story.append(Spacer(1, 0.2 * inch))

        # Attention visualization
        if report_data.attention_heatmap_path:
            story.append(Paragraph("Attention Visualization", heading_style))
            story.append(
                Paragraph(
                    "The following heatmap shows tissue regions that influenced the diagnosis:",
                    styles["Normal"],
                )
            )
            try:
                img = Image(report_data.attention_heatmap_path, width=5 * inch, height=5 * inch)
                story.append(img)
            except Exception as e:
                story.append(Paragraph(f"[Image could not be loaded: {e}]", styles["Normal"]))
            story.append(Spacer(1, 0.2 * inch))

        # Longitudinal summary
        if report_data.longitudinal_summary:
            story.append(Paragraph("Longitudinal Progression Summary", heading_style))
            long_text = (
                f"Number of Previous Scans: {report_data.longitudinal_summary.get('num_scans', 'N/A')}<br/>"
                f"Timeline Duration: {report_data.longitudinal_summary.get('duration_days', 'N/A')} days"
            )
            if report_data.longitudinal_summary.get("progression_trend"):
                long_text += f"<br/>Progression Trend: {report_data.longitudinal_summary['progression_trend']}"
            story.append(Paragraph(long_text, styles["Normal"]))

            if report_data.previous_scan_comparison:
                comp = report_data.previous_scan_comparison
                comp_text = (
                    f"<b>Change from Previous Scan:</b><br/>"
                    f"Previous Diagnosis: {comp.get('previous_diagnosis', 'N/A')}<br/>"
                    f"Current Diagnosis: {comp.get('current_diagnosis', 'N/A')}<br/>"
                    f"Probability Change: {comp.get('probability_change', 'N/A')}"
                )
                story.append(Paragraph(comp_text, styles["Normal"]))
            story.append(Spacer(1, 0.2 * inch))

        # Physician notes
        if report_data.physician_notes:
            story.append(Paragraph("Physician Notes", heading_style))
            story.append(Paragraph(report_data.physician_notes, styles["Normal"]))
            story.append(Spacer(1, 0.2 * inch))

        # Amendments
        if report_data.amendments:
            story.append(Paragraph("Report Amendments", heading_style))
            for amendment in report_data.amendments:
                amend_text = (
                    f"<b>Date:</b> {amendment.get('timestamp', 'N/A')}<br/>"
                    f"<b>Amended By:</b> {amendment.get('user', 'N/A')}<br/>"
                    f"<b>Note:</b> {amendment.get('note', 'N/A')}"
                )
                story.append(Paragraph(amend_text, styles["Normal"]))
                story.append(Spacer(1, 0.1 * inch))

        # Metadata footer
        story.append(Spacer(1, 0.3 * inch))
        metadata_text = (
            f"<b>Model Version:</b> {report_data.model_version}<br/>"
            f"<b>Report ID:</b> {report_data.scan_id}_{report_data.report_timestamp.strftime('%Y%m%d%H%M%S')}<br/>"
            f"<i>This report was generated by an AI-assisted diagnostic system and should be reviewed by a qualified pathologist.</i>"
        )
        story.append(Paragraph(metadata_text, styles["Normal"]))

        # Build PDF
        doc.build(story)
        return output_path

    def _export_json(self, report_data: ReportData, output_path: Path) -> Path:
        """Export report as structured JSON."""
        report_dict = {
            "patient_id": report_data.patient_id,
            "scan_id": report_data.scan_id,
            "scan_date": report_data.scan_date.isoformat(),
            "report_timestamp": report_data.report_timestamp.isoformat(),
            "model_version": report_data.model_version,
            "primary_diagnosis": {
                "disease_id": report_data.primary_diagnosis.disease_id,
                "disease_name": report_data.primary_diagnosis.disease_name,
                "probability": report_data.primary_diagnosis.probability,
                "confidence_interval": report_data.primary_diagnosis.confidence_interval,
                "uncertainty_score": report_data.primary_diagnosis.uncertainty_score,
            },
            "alternative_diagnoses": [
                {
                    "disease_id": d.disease_id,
                    "disease_name": d.disease_name,
                    "probability": d.probability,
                    "confidence_interval": d.confidence_interval,
                    "uncertainty_score": d.uncertainty_score,
                }
                for d in report_data.alternative_diagnoses
            ],
            "probability_distribution": report_data.probability_distribution,
            "uncertainty_explanation": report_data.uncertainty_explanation,
            "ood_detected": report_data.ood_detected,
            "ood_explanation": report_data.ood_explanation,
            "risk_scores": report_data.risk_scores,
            "risk_time_horizons": report_data.risk_time_horizons,
            "recommendations": report_data.recommendations,
            "attention_heatmap_path": report_data.attention_heatmap_path,
            "longitudinal_summary": report_data.longitudinal_summary,
            "previous_scan_comparison": report_data.previous_scan_comparison,
            "physician_notes": report_data.physician_notes,
            "amendments": report_data.amendments,
        }

        output_path.write_text(json.dumps(report_dict, indent=2), encoding="utf-8")
        return output_path

    def _export_fhir(self, report_data: ReportData, output_path: Path) -> Path:
        """
        Export report as FHIR DiagnosticReport resource.

        Creates a FHIR R4 DiagnosticReport with observations for each diagnosis.
        """
        # Create FHIR DiagnosticReport resource
        fhir_report = {
            "resourceType": "DiagnosticReport",
            "id": f"{report_data.scan_id}_{report_data.report_timestamp.strftime('%Y%m%d%H%M%S')}",
            "status": "final",
            "category": [
                {
                    "coding": [
                        {
                            "system": "http://terminology.hl7.org/CodeSystem/v2-0074",
                            "code": "PAT",
                            "display": "Pathology",
                        }
                    ]
                }
            ],
            "code": {
                "coding": [
                    {
                        "system": "http://loinc.org",
                        "code": "60567-5",
                        "display": "Comprehensive pathology report",
                    }
                ],
                "text": "Computational Pathology Diagnostic Report",
            },
            "subject": {"reference": f"Patient/{report_data.patient_id}"},
            "effectiveDateTime": report_data.scan_date.isoformat(),
            "issued": report_data.report_timestamp.isoformat(),
            "conclusion": f"Primary Diagnosis: {report_data.primary_diagnosis.disease_name} "
            f"(Probability: {report_data.primary_diagnosis.probability * 100:.1f}%)",
            "conclusionCode": [
                {
                    "coding": [
                        {
                            "system": "http://snomed.info/sct",
                            "code": report_data.primary_diagnosis.disease_id,
                            "display": report_data.primary_diagnosis.disease_name,
                        }
                    ]
                }
            ],
            "presentedForm": [
                {
                    "contentType": "application/json",
                    "data": base64.b64encode(
                        json.dumps(
                            {
                                "probability_distribution": report_data.probability_distribution,
                                "uncertainty_explanation": report_data.uncertainty_explanation,
                                "recommendations": report_data.recommendations,
                            }
                        ).encode()
                    ).decode(),
                }
            ],
            "extension": [
                {
                    "url": "http://example.org/fhir/StructureDefinition/model-version",
                    "valueString": report_data.model_version,
                },
                {
                    "url": "http://example.org/fhir/StructureDefinition/ood-detected",
                    "valueBoolean": report_data.ood_detected,
                },
            ],
        }

        # Add alternative diagnoses as observations
        if report_data.alternative_diagnoses:
            observations = []
            for i, diag in enumerate(report_data.alternative_diagnoses):
                obs = {
                    "resourceType": "Observation",
                    "id": f"alt-diagnosis-{i}",
                    "status": "final",
                    "code": {
                        "coding": [
                            {
                                "system": "http://snomed.info/sct",
                                "code": diag.disease_id,
                                "display": diag.disease_name,
                            }
                        ]
                    },
                    "valueQuantity": {
                        "value": diag.probability,
                        "unit": "probability",
                        "system": "http://unitsofmeasure.org",
                        "code": "1",
                    },
                }
                observations.append(obs)

            fhir_report["result"] = [
                {"reference": f"#alt-diagnosis-{i}"} for i in range(len(observations))
            ]
            fhir_report["contained"] = observations

        # Add longitudinal summary if available
        if report_data.longitudinal_summary:
            fhir_report["extension"].append(
                {
                    "url": "http://example.org/fhir/StructureDefinition/longitudinal-summary",
                    "valueString": json.dumps(report_data.longitudinal_summary),
                }
            )

        output_path.write_text(json.dumps(fhir_report, indent=2), encoding="utf-8")
        return output_path

    def _export_dicom_sr(self, report_data: ReportData, output_path: Path) -> Path:
        """
        Export report as DICOM Structured Report (SR).

        Creates a simplified DICOM SR representation in JSON format.
        Note: Full DICOM SR generation requires pydicom and proper DICOM tags.
        """
        # Simplified DICOM SR structure
        dicom_sr = {
            "SOPClassUID": "1.2.840.10008.5.1.4.1.1.88.11",  # Basic Text SR
            "SOPInstanceUID": f"1.2.840.10008.{report_data.scan_id}.{int(report_data.report_timestamp.timestamp())}",
            "Modality": "SR",
            "SeriesDescription": "Computational Pathology Report",
            "ContentDate": report_data.scan_date.strftime("%Y%m%d"),
            "ContentTime": report_data.scan_date.strftime("%H%M%S"),
            "PatientID": report_data.patient_id,
            "StudyInstanceUID": report_data.scan_id,
            "SeriesInstanceUID": f"{report_data.scan_id}.1",
            "CompletionFlag": "COMPLETE",
            "VerificationFlag": "UNVERIFIED",
            "ContentSequence": [
                {
                    "RelationshipType": "CONTAINS",
                    "ValueType": "TEXT",
                    "ConceptNameCodeSequence": {
                        "CodeValue": "121060",
                        "CodingSchemeDesignator": "DCM",
                        "CodeMeaning": "History",
                    },
                    "TextValue": f"Computational pathology analysis performed on {report_data.scan_date.strftime('%Y-%m-%d')}",
                },
                {
                    "RelationshipType": "CONTAINS",
                    "ValueType": "CODE",
                    "ConceptNameCodeSequence": {
                        "CodeValue": "121071",
                        "CodingSchemeDesignator": "DCM",
                        "CodeMeaning": "Finding",
                    },
                    "ConceptCodeSequence": {
                        "CodeValue": report_data.primary_diagnosis.disease_id,
                        "CodingSchemeDesignator": "SCT",
                        "CodeMeaning": report_data.primary_diagnosis.disease_name,
                    },
                },
                {
                    "RelationshipType": "CONTAINS",
                    "ValueType": "NUM",
                    "ConceptNameCodeSequence": {
                        "CodeValue": "121401",
                        "CodingSchemeDesignator": "DCM",
                        "CodeMeaning": "Probability",
                    },
                    "MeasuredValueSequence": {
                        "NumericValue": report_data.primary_diagnosis.probability,
                        "MeasurementUnitsCodeSequence": {
                            "CodeValue": "1",
                            "CodingSchemeDesignator": "UCUM",
                            "CodeMeaning": "probability",
                        },
                    },
                },
            ],
        }

        # Add alternative diagnoses
        for diag in report_data.alternative_diagnoses:
            dicom_sr["ContentSequence"].extend(
                [
                    {
                        "RelationshipType": "CONTAINS",
                        "ValueType": "CODE",
                        "ConceptNameCodeSequence": {
                            "CodeValue": "121071",
                            "CodingSchemeDesignator": "DCM",
                            "CodeMeaning": "Alternative Finding",
                        },
                        "ConceptCodeSequence": {
                            "CodeValue": diag.disease_id,
                            "CodingSchemeDesignator": "SCT",
                            "CodeMeaning": diag.disease_name,
                        },
                    },
                    {
                        "RelationshipType": "CONTAINS",
                        "ValueType": "NUM",
                        "ConceptNameCodeSequence": {
                            "CodeValue": "121401",
                            "CodingSchemeDesignator": "DCM",
                            "CodeMeaning": "Probability",
                        },
                        "MeasuredValueSequence": {
                            "NumericValue": diag.probability,
                            "MeasurementUnitsCodeSequence": {
                                "CodeValue": "1",
                                "CodingSchemeDesignator": "UCUM",
                                "CodeMeaning": "probability",
                            },
                        },
                    },
                ]
            )

        # Add model version
        dicom_sr["ContentSequence"].append(
            {
                "RelationshipType": "CONTAINS",
                "ValueType": "TEXT",
                "ConceptNameCodeSequence": {
                    "CodeValue": "121106",
                    "CodingSchemeDesignator": "DCM",
                    "CodeMeaning": "Comment",
                },
                "TextValue": f"Model Version: {report_data.model_version}",
            }
        )

        output_path.write_text(json.dumps(dicom_sr, indent=2), encoding="utf-8")
        return output_path

    def add_physician_annotation(
        self,
        report_data: ReportData,
        user: str,
        note: str,
    ) -> ReportData:
        """
        Add physician annotation/amendment to report.

        Args:
            report_data: Existing report data.
            user: Username or identifier of physician adding annotation.
            note: Annotation text.

        Returns:
            Updated report data with new amendment.
        """
        amendment = {
            "timestamp": datetime.now().isoformat(),
            "user": user,
            "note": note,
        }
        report_data.amendments.append(amendment)
        return report_data
