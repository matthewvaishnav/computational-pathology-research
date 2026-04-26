"""Clinical report generation for Real-Time WSI Streaming.

Generates PDF reports with visualizations, confidence metrics, and processing metadata
for clinical use in hospital demos and production deployments.
"""

import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import io

import numpy as np
from PIL import Image

# PDF generation
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.units import inch
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Image as RLImage,
        Table, TableStyle, PageBreak, KeepTogether
    )
    from reportlab.lib import colors
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class StreamingReportData:
    """Data for streaming WSI clinical report."""
    # Patient/Slide info
    slide_id: str
    patient_id: Optional[str] = None
    scan_date: Optional[datetime] = None
    
    # Processing results
    prediction_class: str = ""
    confidence: float = 0.0
    processing_time: float = 0.0
    patches_processed: int = 0
    total_patches: int = 0
    
    # Attention data
    attention_heatmap_path: Optional[Path] = None
    confidence_plot_path: Optional[Path] = None
    
    # Quality metrics
    coverage_percent: float = 0.0
    throughput: float = 0.0
    memory_usage_gb: float = 0.0
    
    # Uncertainty quantification
    uncertainty_score: Optional[float] = None
    confidence_interval: Optional[tuple[float, float]] = None
    
    # Metadata
    model_version: str = "1.0.0"
    processing_mode: str = "streaming"
    report_timestamp: datetime = field(default_factory=datetime.now)
    
    # Clinical notes
    physician_notes: Optional[str] = None
    recommendations: List[str] = field(default_factory=list)


class StreamingReportGenerator:
    """Generate clinical reports for real-time WSI streaming results.
    
    Features:
    - PDF report templates with visualizations
    - Confidence metrics and uncertainty quantification
    - Processing metadata and quality metrics
    - Institutional branding support
    """
    
    def __init__(self, 
                 institution_name: str = "HistoCore Medical AI",
                 institution_logo: Optional[Path] = None,
                 template_config: Optional[Dict[str, Any]] = None):
        """Initialize report generator.
        
        Args:
            institution_name: Name of institution for branding
            institution_logo: Path to institution logo image
            template_config: Custom template configuration
        """
        if not REPORTLAB_AVAILABLE:
            raise ImportError("reportlab required for PDF generation. Install with: pip install reportlab")
        
        self.institution_name = institution_name
        self.institution_logo = institution_logo
        self.template_config = template_config or {}
        
        # Setup styles
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
        
        logger.info(f"Initialized StreamingReportGenerator for {institution_name}")
    
    def _setup_custom_styles(self):
        """Setup custom paragraph styles for reports."""
        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1a1a1a'),
            spaceAfter=30,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        ))
        
        # Section header style
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#2c5aa0'),
            spaceBefore=20,
            spaceAfter=12,
            fontName='Helvetica-Bold'
        ))
        
        # Metric style
        self.styles.add(ParagraphStyle(
            name='Metric',
            parent=self.styles['Normal'],
            fontSize=11,
            textColor=colors.HexColor('#333333'),
            spaceAfter=6
        ))
        
        # Confidence high style
        self.styles.add(ParagraphStyle(
            name='ConfidenceHigh',
            parent=self.styles['Normal'],
            fontSize=18,
            textColor=colors.HexColor('#2e7d32'),
            fontName='Helvetica-Bold'
        ))
        
        # Confidence medium style
        self.styles.add(ParagraphStyle(
            name='ConfidenceMedium',
            parent=self.styles['Normal'],
            fontSize=18,
            textColor=colors.HexColor('#f57c00'),
            fontName='Helvetica-Bold'
        ))
        
        # Confidence low style
        self.styles.add(ParagraphStyle(
            name='ConfidenceLow',
            parent=self.styles['Normal'],
            fontSize=18,
            textColor=colors.HexColor('#c62828'),
            fontName='Helvetica-Bold'
        ))
    
    def generate_pdf_report(self, 
                           report_data: StreamingReportData,
                           output_path: Path) -> Path:
        """Generate PDF clinical report.
        
        Args:
            report_data: Report data with results and visualizations
            output_path: Path to save PDF report
            
        Returns:
            Path to generated PDF file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create PDF document
        doc = SimpleDocTemplate(
            str(output_path),
            pagesize=letter,
            rightMargin=0.75*inch,
            leftMargin=0.75*inch,
            topMargin=0.75*inch,
            bottomMargin=0.75*inch
        )
        
        # Build report content
        story = []
        
        # Header
        story.extend(self._build_header(report_data))
        
        # Patient/Slide information
        story.extend(self._build_patient_info(report_data))
        
        # Primary results
        story.extend(self._build_results_section(report_data))
        
        # Visualizations
        story.extend(self._build_visualizations_section(report_data))
        
        # Quality metrics
        story.extend(self._build_quality_metrics(report_data))
        
        # Processing metadata
        story.extend(self._build_metadata_section(report_data))
        
        # Clinical notes and recommendations
        if report_data.physician_notes or report_data.recommendations:
            story.extend(self._build_clinical_notes(report_data))
        
        # Footer
        story.extend(self._build_footer(report_data))
        
        # Build PDF
        doc.build(story)
        
        logger.info(f"Generated PDF report: {output_path}")
        return output_path
    
    def _build_header(self, report_data: StreamingReportData) -> List:
        """Build report header with institution branding."""
        elements = []
        
        # Institution logo if provided
        if self.institution_logo and self.institution_logo.exists():
            try:
                logo = RLImage(str(self.institution_logo), width=2*inch, height=0.75*inch)
                elements.append(logo)
                elements.append(Spacer(1, 0.2*inch))
            except Exception as e:
                logger.warning(f"Failed to load institution logo: {e}")
        
        # Title
        title = Paragraph(
            f"<b>{self.institution_name}</b><br/>Real-Time WSI Analysis Report",
            self.styles['CustomTitle']
        )
        elements.append(title)
        elements.append(Spacer(1, 0.3*inch))
        
        return elements
    
    def _build_patient_info(self, report_data: StreamingReportData) -> List:
        """Build patient/slide information section."""
        elements = []
        
        elements.append(Paragraph("Slide Information", self.styles['SectionHeader']))
        
        # Create info table
        data = [
            ['Slide ID:', report_data.slide_id],
            ['Report Date:', report_data.report_timestamp.strftime('%Y-%m-%d %H:%M:%S')],
        ]
        
        if report_data.patient_id:
            data.insert(1, ['Patient ID:', report_data.patient_id])
        
        if report_data.scan_date:
            data.append(['Scan Date:', report_data.scan_date.strftime('%Y-%m-%d')])
        
        table = Table(data, colWidths=[2*inch, 4*inch])
        table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#555555')),
            ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
            ('ALIGN', (1, 0), (1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))
        
        elements.append(table)
        elements.append(Spacer(1, 0.3*inch))
        
        return elements
    
    def _build_results_section(self, report_data: StreamingReportData) -> List:
        """Build primary results section with confidence."""
        elements = []
        
        elements.append(Paragraph("Analysis Results", self.styles['SectionHeader']))
        
        # Prediction
        pred_text = f"<b>Prediction:</b> {report_data.prediction_class}"
        elements.append(Paragraph(pred_text, self.styles['Metric']))
        elements.append(Spacer(1, 0.1*inch))
        
        # Confidence with color coding
        confidence_pct = report_data.confidence * 100
        if report_data.confidence >= 0.90:
            style = 'ConfidenceHigh'
            interpretation = "High confidence"
        elif report_data.confidence >= 0.70:
            style = 'ConfidenceMedium'
            interpretation = "Moderate confidence"
        else:
            style = 'ConfidenceLow'
            interpretation = "Low confidence - manual review recommended"
        
        conf_text = f"<b>Confidence:</b> {confidence_pct:.1f}%"
        elements.append(Paragraph(conf_text, self.styles[style]))
        elements.append(Paragraph(f"<i>{interpretation}</i>", self.styles['Normal']))
        elements.append(Spacer(1, 0.1*inch))
        
        # Uncertainty quantification if available
        if report_data.uncertainty_score is not None:
            unc_text = f"<b>Uncertainty Score:</b> {report_data.uncertainty_score:.3f}"
            elements.append(Paragraph(unc_text, self.styles['Metric']))
        
        if report_data.confidence_interval:
            ci_low, ci_high = report_data.confidence_interval
            ci_text = f"<b>95% Confidence Interval:</b> [{ci_low:.3f}, {ci_high:.3f}]"
            elements.append(Paragraph(ci_text, self.styles['Metric']))
        
        elements.append(Spacer(1, 0.3*inch))
        
        return elements
    
    def _build_visualizations_section(self, report_data: StreamingReportData) -> List:
        """Build visualizations section with attention heatmap and confidence plot."""
        elements = []
        
        elements.append(Paragraph("Visualizations", self.styles['SectionHeader']))
        
        viz_elements = []
        
        # Attention heatmap
        if report_data.attention_heatmap_path and report_data.attention_heatmap_path.exists():
            try:
                heatmap = RLImage(
                    str(report_data.attention_heatmap_path),
                    width=5*inch,
                    height=4*inch
                )
                viz_elements.append(heatmap)
                viz_elements.append(Paragraph(
                    "<i>Figure 1: Attention heatmap showing regions of diagnostic importance</i>",
                    self.styles['Normal']
                ))
                viz_elements.append(Spacer(1, 0.2*inch))
            except Exception as e:
                logger.warning(f"Failed to load attention heatmap: {e}")
        
        # Confidence progression plot
        if report_data.confidence_plot_path and report_data.confidence_plot_path.exists():
            try:
                conf_plot = RLImage(
                    str(report_data.confidence_plot_path),
                    width=5*inch,
                    height=3*inch
                )
                viz_elements.append(conf_plot)
                viz_elements.append(Paragraph(
                    "<i>Figure 2: Confidence progression during real-time analysis</i>",
                    self.styles['Normal']
                ))
            except Exception as e:
                logger.warning(f"Failed to load confidence plot: {e}")
        
        if viz_elements:
            elements.extend(viz_elements)
        else:
            elements.append(Paragraph(
                "<i>No visualizations available</i>",
                self.styles['Normal']
            ))
        
        elements.append(Spacer(1, 0.3*inch))
        
        return elements
    
    def _build_quality_metrics(self, report_data: StreamingReportData) -> List:
        """Build quality metrics section."""
        elements = []
        
        elements.append(Paragraph("Quality Metrics", self.styles['SectionHeader']))
        
        # Create metrics table
        data = [
            ['Coverage:', f"{report_data.coverage_percent:.1f}%"],
            ['Patches Processed:', f"{report_data.patches_processed:,} / {report_data.total_patches:,}"],
            ['Processing Time:', f"{report_data.processing_time:.1f}s"],
            ['Throughput:', f"{report_data.throughput:.1f} patches/sec"],
        ]
        
        if report_data.memory_usage_gb > 0:
            data.append(['Peak Memory Usage:', f"{report_data.memory_usage_gb:.2f} GB"])
        
        table = Table(data, colWidths=[2.5*inch, 2.5*inch])
        table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#555555')),
            ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
            ('ALIGN', (1, 0), (1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ]))
        
        elements.append(table)
        elements.append(Spacer(1, 0.3*inch))
        
        return elements
    
    def _build_metadata_section(self, report_data: StreamingReportData) -> List:
        """Build processing metadata section."""
        elements = []
        
        elements.append(Paragraph("Processing Metadata", self.styles['SectionHeader']))
        
        data = [
            ['Model Version:', report_data.model_version],
            ['Processing Mode:', report_data.processing_mode.capitalize()],
            ['Analysis Timestamp:', report_data.report_timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')],
        ]
        
        table = Table(data, colWidths=[2.5*inch, 3.5*inch])
        table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#666666')),
            ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
            ('ALIGN', (1, 0), (1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        
        elements.append(table)
        elements.append(Spacer(1, 0.3*inch))
        
        return elements
    
    def _build_clinical_notes(self, report_data: StreamingReportData) -> List:
        """Build clinical notes and recommendations section."""
        elements = []
        
        elements.append(Paragraph("Clinical Notes", self.styles['SectionHeader']))
        
        if report_data.physician_notes:
            elements.append(Paragraph(
                f"<b>Physician Notes:</b><br/>{report_data.physician_notes}",
                self.styles['Normal']
            ))
            elements.append(Spacer(1, 0.1*inch))
        
        if report_data.recommendations:
            elements.append(Paragraph("<b>Recommendations:</b>", self.styles['Normal']))
            for i, rec in enumerate(report_data.recommendations, 1):
                elements.append(Paragraph(f"{i}. {rec}", self.styles['Normal']))
        
        elements.append(Spacer(1, 0.3*inch))
        
        return elements
    
    def _build_footer(self, report_data: StreamingReportData) -> List:
        """Build report footer with disclaimers."""
        elements = []
        
        elements.append(Spacer(1, 0.5*inch))
        
        disclaimer = """
        <i><font size=8>
        DISCLAIMER: This report is generated by an AI-assisted diagnostic system and should be used 
        as a decision support tool only. All results must be reviewed and validated by a qualified 
        healthcare professional. This system is not intended to replace clinical judgment.
        </font></i>
        """
        
        elements.append(Paragraph(disclaimer, self.styles['Normal']))
        
        return elements
    
    def generate_html_report(self, 
                            report_data: StreamingReportData,
                            output_path: Path) -> Path:
        """Generate HTML clinical report.
        
        Args:
            report_data: Report data with results and visualizations
            output_path: Path to save HTML report
            
        Returns:
            Path to generated HTML file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Build HTML content
        html_content = self._build_html_content(report_data)
        
        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Generated HTML report: {output_path}")
        return output_path
    
    def _build_html_content(self, report_data: StreamingReportData) -> str:
        """Build HTML report content."""
        # Confidence color
        if report_data.confidence >= 0.90:
            conf_color = '#2e7d32'
        elif report_data.confidence >= 0.70:
            conf_color = '#f57c00'
        else:
            conf_color = '#c62828'
        
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>WSI Analysis Report - {report_data.slide_id}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 900px;
            margin: 40px auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .container {{
            background: white;
            padding: 40px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .header {{
            text-align: center;
            border-bottom: 3px solid #2c5aa0;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}
        .header h1 {{
            color: #1a1a1a;
            margin: 0;
        }}
        .header h2 {{
            color: #666;
            font-weight: normal;
            margin: 10px 0 0 0;
        }}
        .section {{
            margin: 30px 0;
        }}
        .section-title {{
            color: #2c5aa0;
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 15px;
            border-bottom: 2px solid #e0e0e0;
            padding-bottom: 5px;
        }}
        .info-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }}
        .info-table td {{
            padding: 8px;
            border-bottom: 1px solid #e0e0e0;
        }}
        .info-table td:first-child {{
            font-weight: bold;
            color: #555;
            width: 200px;
        }}
        .confidence {{
            font-size: 24px;
            font-weight: bold;
            color: {conf_color};
            margin: 10px 0;
        }}
        .visualization {{
            text-align: center;
            margin: 20px 0;
        }}
        .visualization img {{
            max-width: 100%;
            border: 1px solid #ddd;
            border-radius: 4px;
        }}
        .visualization-caption {{
            font-style: italic;
            color: #666;
            margin-top: 10px;
        }}
        .disclaimer {{
            background: #fff3cd;
            border: 1px solid #ffc107;
            border-radius: 4px;
            padding: 15px;
            margin-top: 30px;
            font-size: 12px;
            color: #856404;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{self.institution_name}</h1>
            <h2>Real-Time WSI Analysis Report</h2>
        </div>
        
        <div class="section">
            <div class="section-title">Slide Information</div>
            <table class="info-table">
                <tr>
                    <td>Slide ID:</td>
                    <td>{report_data.slide_id}</td>
                </tr>
                <tr>
                    <td>Report Date:</td>
                    <td>{report_data.report_timestamp.strftime('%Y-%m-%d %H:%M:%S')}</td>
                </tr>
            </table>
        </div>
        
        <div class="section">
            <div class="section-title">Analysis Results</div>
            <p><strong>Prediction:</strong> {report_data.prediction_class}</p>
            <div class="confidence">Confidence: {report_data.confidence*100:.1f}%</div>
        </div>
        
        <div class="section">
            <div class="section-title">Quality Metrics</div>
            <table class="info-table">
                <tr>
                    <td>Coverage:</td>
                    <td>{report_data.coverage_percent:.1f}%</td>
                </tr>
                <tr>
                    <td>Patches Processed:</td>
                    <td>{report_data.patches_processed:,} / {report_data.total_patches:,}</td>
                </tr>
                <tr>
                    <td>Processing Time:</td>
                    <td>{report_data.processing_time:.1f}s</td>
                </tr>
                <tr>
                    <td>Throughput:</td>
                    <td>{report_data.throughput:.1f} patches/sec</td>
                </tr>
            </table>
        </div>
        
        <div class="disclaimer">
            <strong>DISCLAIMER:</strong> This report is generated by an AI-assisted diagnostic system 
            and should be used as a decision support tool only. All results must be reviewed and 
            validated by a qualified healthcare professional.
        </div>
    </div>
</body>
</html>
"""
        
        return html
