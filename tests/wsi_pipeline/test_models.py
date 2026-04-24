"""Unit tests for WSI pipeline data models."""

from pathlib import Path

import pytest

from src.data.wsi_pipeline.models import ProcessingResult, SlideMetadata


class TestSlideMetadata:
    """Tests for SlideMetadata dataclass."""

    def test_minimal_metadata(self):
        """Test creating SlideMetadata with minimal required fields."""
        metadata = SlideMetadata(
            slide_id="slide_001",
            patient_id="patient_001",
            file_path="/path/to/slide.svs",
            label=1,
            split="train",
        )

        assert metadata.slide_id == "slide_001"
        assert metadata.patient_id == "patient_001"
        assert metadata.file_path == "/path/to/slide.svs"
        assert metadata.label == 1
        assert metadata.split == "train"
        assert metadata.annotation_path is None
        assert metadata.width is None
        assert metadata.height is None
        assert metadata.magnification is None
        assert metadata.mpp is None
        assert metadata.scanner_model is None
        assert metadata.scan_date is None
        assert metadata.processing_timestamp is None

    def test_complete_metadata(self):
        """Test creating SlideMetadata with all fields."""
        metadata = SlideMetadata(
            slide_id="slide_001",
            patient_id="patient_001",
            file_path="/path/to/slide.svs",
            label=1,
            split="train",
            annotation_path="/path/to/annotation.xml",
            width=100000,
            height=80000,
            magnification=40.0,
            mpp=0.25,
            scanner_model="Aperio ScanScope",
            scan_date="2024-01-15",
            processing_timestamp="2024-01-16T10:30:00",
        )

        assert metadata.slide_id == "slide_001"
        assert metadata.patient_id == "patient_001"
        assert metadata.file_path == "/path/to/slide.svs"
        assert metadata.label == 1
        assert metadata.split == "train"
        assert metadata.annotation_path == "/path/to/annotation.xml"
        assert metadata.width == 100000
        assert metadata.height == 80000
        assert metadata.magnification == 40.0
        assert metadata.mpp == 0.25
        assert metadata.scanner_model == "Aperio ScanScope"
        assert metadata.scan_date == "2024-01-15"
        assert metadata.processing_timestamp == "2024-01-16T10:30:00"

    def test_metadata_with_negative_label(self):
        """Test creating SlideMetadata with negative label."""
        metadata = SlideMetadata(
            slide_id="slide_002",
            patient_id="patient_002",
            file_path="/path/to/slide.svs",
            label=0,
            split="val",
        )

        assert metadata.label == 0
        assert metadata.split == "val"

    def test_metadata_with_test_split(self):
        """Test creating SlideMetadata with test split."""
        metadata = SlideMetadata(
            slide_id="slide_003",
            patient_id="patient_003",
            file_path="/path/to/slide.svs",
            label=1,
            split="test",
        )

        assert metadata.split == "test"


class TestProcessingResult:
    """Tests for ProcessingResult dataclass."""

    def test_successful_result(self):
        """Test creating ProcessingResult for successful processing."""
        result = ProcessingResult(
            slide_id="slide_001",
            success=True,
            feature_file=Path("/path/to/features/slide_001.h5"),
            num_patches=1500,
            processing_time=120.5,
            qc_metrics={"blur_score": 150.0, "tissue_coverage": 0.75, "num_blurry_patches": 10},
        )

        assert result.slide_id == "slide_001"
        assert result.success is True
        assert result.feature_file == Path("/path/to/features/slide_001.h5")
        assert result.num_patches == 1500
        assert result.processing_time == 120.5
        assert result.error_message is None
        assert result.qc_metrics is not None
        assert result.qc_metrics["blur_score"] == 150.0
        assert result.qc_metrics["tissue_coverage"] == 0.75
        assert result.qc_metrics["num_blurry_patches"] == 10

    def test_failed_result(self):
        """Test creating ProcessingResult for failed processing."""
        result = ProcessingResult(
            slide_id="slide_002",
            success=False,
            error_message="Failed to open slide: file corrupted",
        )

        assert result.slide_id == "slide_002"
        assert result.success is False
        assert result.feature_file is None
        assert result.num_patches == 0
        assert result.processing_time == 0.0
        assert result.error_message == "Failed to open slide: file corrupted"
        assert result.qc_metrics is None

    def test_minimal_successful_result(self):
        """Test creating ProcessingResult with minimal fields."""
        result = ProcessingResult(slide_id="slide_003", success=True)

        assert result.slide_id == "slide_003"
        assert result.success is True
        assert result.feature_file is None
        assert result.num_patches == 0
        assert result.processing_time == 0.0
        assert result.error_message is None
        assert result.qc_metrics is None

    def test_result_with_zero_patches(self):
        """Test ProcessingResult with zero patches (e.g., no tissue found)."""
        result = ProcessingResult(
            slide_id="slide_004",
            success=True,
            num_patches=0,
            processing_time=5.0,
            qc_metrics={"tissue_coverage": 0.02},
        )

        assert result.success is True
        assert result.num_patches == 0
        assert result.qc_metrics["tissue_coverage"] == 0.02

    def test_result_with_long_processing_time(self):
        """Test ProcessingResult with long processing time."""
        result = ProcessingResult(
            slide_id="slide_005", success=True, num_patches=50000, processing_time=3600.0  # 1 hour
        )

        assert result.processing_time == 3600.0
        assert result.num_patches == 50000
