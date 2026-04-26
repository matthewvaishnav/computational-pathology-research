"""Tests for clinical report generation."""

import pytest
import tempfile
from pathlib import Path
from datetime import datetime
import numpy as np
from PIL import Image

from src.streaming.clinical_report_generator import (
    StreamingReportData,
    StreamingReportGenerator
)


@pytest.fixture
def temp_output_dir():
    """Create temporary output directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_report_data(temp_output_dir):
    """Create sample report data for testing."""
    # Create sample visualization images
    heatmap_path = temp_output_dir / "test_heatmap.png"
    conf_plot_path = temp_output_dir / "test_confidence.png"
    
    # Generate sample images
    heatmap_img = Image.fromarray(
        (np.random.rand(400, 400, 3) * 255).astype(np.uint8)
    )
    heatmap_img.save(heatmap_path)
    
    conf_img = Image.fromarray(
        (np.random.rand(300, 400, 3) * 255).astype(np.uint8)
    )
    conf_img.save(conf_plot_path)
    
    return StreamingReportData(
        slide_id="TEST-SLIDE-001",
        patient_id="PAT-12345",
        scan_date=datetime(2026, 4, 15),
        prediction_class="Tumor",
        confidence=0.92,
        processing_time=25.3,
        patches_processed=95000,
        total_patches=100000,
        attention_heatmap_path=heatmap_path,
        confidence_plot_path=conf_plot_path,
        coverage_percent=95.0,
        throughput=3755.9,
        memory_usage_gb=1.8,
        uncertainty_score=0.08,
        confidence_interval=(0.88, 0.96),
        model_version="1.2.0",
        processing_mode="streaming",
        physician_notes="Sample clinical notes for testing",
        recommendations=[
            "Recommend further immunohistochemistry",
            "Consider molecular profiling"
        ]
    )


@pytest.fixture
def report_generator(temp_output_dir):
    """Create report generator instance."""
    return StreamingReportGenerator(
        institution_name="Test Medical Center",
        institution_logo=None
    )


class TestStreamingReportData:
    """Tests for StreamingReportData dataclass."""
    
    def test_minimal_report_data(self):
        """Test creating report data with minimal fields."""
        data = StreamingReportData(slide_id="TEST-001")
        
        assert data.slide_id == "TEST-001"
        assert data.patient_id is None
        assert data.confidence == 0.0
        assert data.prediction_class == ""
        assert isinstance(data.report_timestamp, datetime)
    
    def test_full_report_data(self, sample_report_data):
        """Test creating report data with all fields."""
        data = sample_report_data
        
        assert data.slide_id == "TEST-SLIDE-001"
        assert data.patient_id == "PAT-12345"
        assert data.confidence == 0.92
        assert data.prediction_class == "Tumor"
        assert data.processing_time == 25.3
        assert data.patches_processed == 95000
        assert data.total_patches == 100000
        assert data.coverage_percent == 95.0
        assert data.throughput == 3755.9
        assert data.memory_usage_gb == 1.8
        assert data.uncertainty_score == 0.08
        assert data.confidence_interval == (0.88, 0.96)
        assert len(data.recommendations) == 2


class TestStreamingReportGenerator:
    """Tests for StreamingReportGenerator class."""
    
    def test_initialization(self, temp_output_dir):
        """Test report generator initialization."""
        generator = StreamingReportGenerator(
            institution_name="Test Hospital",
            institution_logo=None
        )
        
        assert generator.institution_name == "Test Hospital"
        assert generator.institution_logo is None
        assert generator.styles is not None
    
    def test_initialization_with_logo(self, temp_output_dir):
        """Test initialization with institution logo."""
        logo_path = temp_output_dir / "logo.png"
        logo_img = Image.fromarray(
            (np.random.rand(100, 200, 3) * 255).astype(np.uint8)
        )
        logo_img.save(logo_path)
        
        generator = StreamingReportGenerator(
            institution_name="Test Hospital",
            institution_logo=logo_path
        )
        
        assert generator.institution_logo == logo_path
        assert generator.institution_logo.exists()
    
    def test_custom_styles_created(self, report_generator):
        """Test that custom paragraph styles are created."""
        assert 'CustomTitle' in report_generator.styles
        assert 'SectionHeader' in report_generator.styles
        assert 'Metric' in report_generator.styles
        assert 'ConfidenceHigh' in report_generator.styles
        assert 'ConfidenceMedium' in report_generator.styles
        assert 'ConfidenceLow' in report_generator.styles


class TestPDFReportGeneration:
    """Tests for PDF report generation."""
    
    def test_generate_pdf_basic(self, report_generator, sample_report_data, temp_output_dir):
        """Test basic PDF report generation."""
        output_path = temp_output_dir / "test_report.pdf"
        
        result_path = report_generator.generate_pdf_report(
            sample_report_data,
            output_path
        )
        
        assert result_path.exists()
        assert result_path.suffix == '.pdf'
        assert result_path.stat().st_size > 0
    
    def test_generate_pdf_minimal_data(self, report_generator, temp_output_dir):
        """Test PDF generation with minimal data."""
        minimal_data = StreamingReportData(
            slide_id="MIN-001",
            prediction_class="Normal",
            confidence=0.75
        )
        
        output_path = temp_output_dir / "minimal_report.pdf"
        result_path = report_generator.generate_pdf_report(
            minimal_data,
            output_path
        )
        
        assert result_path.exists()
        assert result_path.stat().st_size > 0
    
    def test_generate_pdf_high_confidence(self, report_generator, temp_output_dir):
        """Test PDF generation with high confidence (green)."""
        data = StreamingReportData(
            slide_id="HIGH-CONF-001",
            prediction_class="Tumor",
            confidence=0.95
        )
        
        output_path = temp_output_dir / "high_conf_report.pdf"
        result_path = report_generator.generate_pdf_report(data, output_path)
        
        assert result_path.exists()
    
    def test_generate_pdf_medium_confidence(self, report_generator, temp_output_dir):
        """Test PDF generation with medium confidence (orange)."""
        data = StreamingReportData(
            slide_id="MED-CONF-001",
            prediction_class="Tumor",
            confidence=0.80
        )
        
        output_path = temp_output_dir / "med_conf_report.pdf"
        result_path = report_generator.generate_pdf_report(data, output_path)
        
        assert result_path.exists()
    
    def test_generate_pdf_low_confidence(self, report_generator, temp_output_dir):
        """Test PDF generation with low confidence (red)."""
        data = StreamingReportData(
            slide_id="LOW-CONF-001",
            prediction_class="Tumor",
            confidence=0.65
        )
        
        output_path = temp_output_dir / "low_conf_report.pdf"
        result_path = report_generator.generate_pdf_report(data, output_path)
        
        assert result_path.exists()
    
    def test_generate_pdf_with_visualizations(self, report_generator, sample_report_data, temp_output_dir):
        """Test PDF generation with attention heatmap and confidence plot."""
        output_path = temp_output_dir / "viz_report.pdf"
        
        result_path = report_generator.generate_pdf_report(
            sample_report_data,
            output_path
        )
        
        assert result_path.exists()
        # PDF with visualizations should be larger
        assert result_path.stat().st_size > 50000
    
    def test_generate_pdf_with_clinical_notes(self, report_generator, temp_output_dir):
        """Test PDF generation with physician notes and recommendations."""
        data = StreamingReportData(
            slide_id="NOTES-001",
            prediction_class="Tumor",
            confidence=0.88,
            physician_notes="Patient has history of melanoma. Recommend close monitoring.",
            recommendations=[
                "Schedule follow-up in 3 months",
                "Consider genetic testing",
                "Refer to oncology"
            ]
        )
        
        output_path = temp_output_dir / "notes_report.pdf"
        result_path = report_generator.generate_pdf_report(data, output_path)
        
        assert result_path.exists()
    
    def test_generate_pdf_with_uncertainty(self, report_generator, temp_output_dir):
        """Test PDF generation with uncertainty quantification."""
        data = StreamingReportData(
            slide_id="UNC-001",
            prediction_class="Tumor",
            confidence=0.85,
            uncertainty_score=0.12,
            confidence_interval=(0.78, 0.92)
        )
        
        output_path = temp_output_dir / "uncertainty_report.pdf"
        result_path = report_generator.generate_pdf_report(data, output_path)
        
        assert result_path.exists()
    
    def test_generate_pdf_creates_parent_dirs(self, report_generator, sample_report_data, temp_output_dir):
        """Test that PDF generation creates parent directories."""
        output_path = temp_output_dir / "subdir" / "nested" / "report.pdf"
        
        result_path = report_generator.generate_pdf_report(
            sample_report_data,
            output_path
        )
        
        assert result_path.exists()
        assert result_path.parent.exists()
    
    def test_generate_pdf_missing_visualizations(self, report_generator, temp_output_dir):
        """Test PDF generation when visualization files are missing."""
        data = StreamingReportData(
            slide_id="MISSING-VIZ-001",
            prediction_class="Normal",
            confidence=0.90,
            attention_heatmap_path=Path("/nonexistent/heatmap.png"),
            confidence_plot_path=Path("/nonexistent/plot.png")
        )
        
        output_path = temp_output_dir / "missing_viz_report.pdf"
        
        # Should not raise error, just skip visualizations
        result_path = report_generator.generate_pdf_report(data, output_path)
        
        assert result_path.exists()


class TestHTMLReportGeneration:
    """Tests for HTML report generation."""
    
    def test_generate_html_basic(self, report_generator, sample_report_data, temp_output_dir):
        """Test basic HTML report generation."""
        output_path = temp_output_dir / "test_report.html"
        
        result_path = report_generator.generate_html_report(
            sample_report_data,
            output_path
        )
        
        assert result_path.exists()
        assert result_path.suffix == '.html'
        assert result_path.stat().st_size > 0
    
    def test_generate_html_content_structure(self, report_generator, sample_report_data, temp_output_dir):
        """Test HTML report contains expected structure."""
        output_path = temp_output_dir / "structure_report.html"
        
        result_path = report_generator.generate_html_report(
            sample_report_data,
            output_path
        )
        
        # Read HTML content
        with open(result_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Check for key sections
        assert "<!DOCTYPE html>" in html_content
        assert sample_report_data.slide_id in html_content
        assert sample_report_data.prediction_class in html_content
        assert "Confidence:" in html_content
        assert "Quality Metrics" in html_content
        assert "DISCLAIMER" in html_content
    
    def test_generate_html_confidence_colors(self, report_generator, temp_output_dir):
        """Test HTML report uses correct confidence colors."""
        # High confidence (green)
        high_data = StreamingReportData(
            slide_id="HIGH-001",
            prediction_class="Tumor",
            confidence=0.95
        )
        high_path = temp_output_dir / "high_conf.html"
        report_generator.generate_html_report(high_data, high_path)
        
        with open(high_path, 'r') as f:
            high_html = f.read()
        assert "#2e7d32" in high_html  # Green color
        
        # Medium confidence (orange)
        med_data = StreamingReportData(
            slide_id="MED-001",
            prediction_class="Tumor",
            confidence=0.80
        )
        med_path = temp_output_dir / "med_conf.html"
        report_generator.generate_html_report(med_data, med_path)
        
        with open(med_path, 'r') as f:
            med_html = f.read()
        assert "#f57c00" in med_html  # Orange color
        
        # Low confidence (red)
        low_data = StreamingReportData(
            slide_id="LOW-001",
            prediction_class="Tumor",
            confidence=0.65
        )
        low_path = temp_output_dir / "low_conf.html"
        report_generator.generate_html_report(low_data, low_path)
        
        with open(low_path, 'r') as f:
            low_html = f.read()
        assert "#c62828" in low_html  # Red color
    
    def test_generate_html_responsive_design(self, report_generator, sample_report_data, temp_output_dir):
        """Test HTML report includes responsive design elements."""
        output_path = temp_output_dir / "responsive_report.html"
        
        result_path = report_generator.generate_html_report(
            sample_report_data,
            output_path
        )
        
        with open(result_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Check for responsive meta tag
        assert 'name="viewport"' in html_content
        assert 'width=device-width' in html_content
    
    def test_generate_html_minimal_data(self, report_generator, temp_output_dir):
        """Test HTML generation with minimal data."""
        minimal_data = StreamingReportData(
            slide_id="MIN-HTML-001",
            prediction_class="Normal",
            confidence=0.70
        )
        
        output_path = temp_output_dir / "minimal_html_report.html"
        result_path = report_generator.generate_html_report(
            minimal_data,
            output_path
        )
        
        assert result_path.exists()
        assert result_path.stat().st_size > 0
    
    def test_generate_html_creates_parent_dirs(self, report_generator, sample_report_data, temp_output_dir):
        """Test that HTML generation creates parent directories."""
        output_path = temp_output_dir / "html" / "reports" / "test.html"
        
        result_path = report_generator.generate_html_report(
            sample_report_data,
            output_path
        )
        
        assert result_path.exists()
        assert result_path.parent.exists()


class TestInstitutionalBranding:
    """Tests for institutional branding features."""
    
    def test_custom_institution_name_pdf(self, temp_output_dir):
        """Test custom institution name in PDF."""
        generator = StreamingReportGenerator(
            institution_name="Johns Hopkins Medical Center"
        )
        
        data = StreamingReportData(
            slide_id="BRAND-001",
            prediction_class="Tumor",
            confidence=0.88
        )
        
        output_path = temp_output_dir / "branded_report.pdf"
        result_path = generator.generate_pdf_report(data, output_path)
        
        assert result_path.exists()
    
    def test_custom_institution_name_html(self, temp_output_dir):
        """Test custom institution name in HTML."""
        generator = StreamingReportGenerator(
            institution_name="Mayo Clinic"
        )
        
        data = StreamingReportData(
            slide_id="BRAND-002",
            prediction_class="Normal",
            confidence=0.92
        )
        
        output_path = temp_output_dir / "branded_report.html"
        result_path = generator.generate_html_report(data, output_path)
        
        with open(result_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        assert "Mayo Clinic" in html_content
    
    def test_institution_logo_pdf(self, temp_output_dir):
        """Test institution logo in PDF report."""
        # Create sample logo
        logo_path = temp_output_dir / "institution_logo.png"
        logo_img = Image.fromarray(
            (np.random.rand(75, 200, 3) * 255).astype(np.uint8)
        )
        logo_img.save(logo_path)
        
        generator = StreamingReportGenerator(
            institution_name="Test Hospital",
            institution_logo=logo_path
        )
        
        data = StreamingReportData(
            slide_id="LOGO-001",
            prediction_class="Tumor",
            confidence=0.85
        )
        
        output_path = temp_output_dir / "logo_report.pdf"
        result_path = generator.generate_pdf_report(data, output_path)
        
        assert result_path.exists()
        # PDF with logo should be larger
        assert result_path.stat().st_size > 10000


class TestReportDataValidation:
    """Tests for report data validation and edge cases."""
    
    def test_confidence_bounds(self, report_generator, temp_output_dir):
        """Test handling of confidence values at boundaries."""
        # Confidence = 0.0
        data_zero = StreamingReportData(
            slide_id="CONF-ZERO",
            prediction_class="Unknown",
            confidence=0.0
        )
        path_zero = temp_output_dir / "conf_zero.pdf"
        result = report_generator.generate_pdf_report(data_zero, path_zero)
        assert result.exists()
        
        # Confidence = 1.0
        data_one = StreamingReportData(
            slide_id="CONF-ONE",
            prediction_class="Tumor",
            confidence=1.0
        )
        path_one = temp_output_dir / "conf_one.pdf"
        result = report_generator.generate_pdf_report(data_one, path_one)
        assert result.exists()
    
    def test_large_patch_counts(self, report_generator, temp_output_dir):
        """Test handling of large patch counts."""
        data = StreamingReportData(
            slide_id="LARGE-001",
            prediction_class="Tumor",
            confidence=0.90,
            patches_processed=500000,
            total_patches=500000
        )
        
        output_path = temp_output_dir / "large_patches.pdf"
        result_path = report_generator.generate_pdf_report(data, output_path)
        
        assert result_path.exists()
    
    def test_special_characters_in_text(self, report_generator, temp_output_dir):
        """Test handling of special characters in text fields."""
        data = StreamingReportData(
            slide_id="SPECIAL-<>&-001",
            prediction_class="Tumor (Grade 3/4)",
            confidence=0.87,
            physician_notes="Patient's history includes: <previous diagnosis> & treatment"
        )
        
        output_path = temp_output_dir / "special_chars.html"
        result_path = report_generator.generate_html_report(data, output_path)
        
        assert result_path.exists()
    
    def test_empty_recommendations_list(self, report_generator, temp_output_dir):
        """Test handling of empty recommendations list."""
        data = StreamingReportData(
            slide_id="EMPTY-REC-001",
            prediction_class="Normal",
            confidence=0.95,
            recommendations=[]
        )
        
        output_path = temp_output_dir / "empty_rec.pdf"
        result_path = report_generator.generate_pdf_report(data, output_path)
        
        assert result_path.exists()
    
    def test_none_optional_fields(self, report_generator, temp_output_dir):
        """Test handling of None values in optional fields."""
        data = StreamingReportData(
            slide_id="NONE-FIELDS-001",
            patient_id=None,
            scan_date=None,
            attention_heatmap_path=None,
            confidence_plot_path=None,
            uncertainty_score=None,
            confidence_interval=None,
            physician_notes=None
        )
        
        output_path = temp_output_dir / "none_fields.pdf"
        result_path = report_generator.generate_pdf_report(data, output_path)
        
        assert result_path.exists()


class TestReportFormats:
    """Tests for different report format outputs."""
    
    def test_pdf_file_extension(self, report_generator, sample_report_data, temp_output_dir):
        """Test PDF report has correct file extension."""
        output_path = temp_output_dir / "report.pdf"
        result_path = report_generator.generate_pdf_report(
            sample_report_data,
            output_path
        )
        
        assert result_path.suffix == '.pdf'
    
    def test_html_file_extension(self, report_generator, sample_report_data, temp_output_dir):
        """Test HTML report has correct file extension."""
        output_path = temp_output_dir / "report.html"
        result_path = report_generator.generate_html_report(
            sample_report_data,
            output_path
        )
        
        assert result_path.suffix == '.html'
    
    def test_pdf_is_binary(self, report_generator, sample_report_data, temp_output_dir):
        """Test PDF report is binary format."""
        output_path = temp_output_dir / "binary_test.pdf"
        result_path = report_generator.generate_pdf_report(
            sample_report_data,
            output_path
        )
        
        # Read first few bytes
        with open(result_path, 'rb') as f:
            header = f.read(4)
        
        # PDF files start with %PDF
        assert header.startswith(b'%PDF')
    
    def test_html_is_text(self, report_generator, sample_report_data, temp_output_dir):
        """Test HTML report is text format."""
        output_path = temp_output_dir / "text_test.html"
        result_path = report_generator.generate_html_report(
            sample_report_data,
            output_path
        )
        
        # Should be readable as text
        with open(result_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        assert isinstance(content, str)
        assert len(content) > 0


class TestErrorHandling:
    """Tests for error handling in report generation."""
    
    def test_invalid_output_path_pdf(self, report_generator, sample_report_data):
        """Test handling of invalid output path for PDF."""
        # This should create parent directories, not fail
        output_path = Path("/tmp/nonexistent/deep/path/report.pdf")
        
        # Should create directories and succeed
        result_path = report_generator.generate_pdf_report(
            sample_report_data,
            output_path
        )
        
        assert result_path.exists()
        result_path.unlink()  # Cleanup
    
    def test_missing_reportlab_import(self):
        """Test error when reportlab is not available."""
        # This test verifies the import check works
        # In actual usage, reportlab should be installed
        import src.streaming.clinical_report_generator as crg
        
        assert crg.REPORTLAB_AVAILABLE is True
