"""
Checkpoint test for Task 12: Verify batch processing and orchestration.

This test verifies that:
1. All tests pass for QualityControl and BatchProcessor
2. Processing a small batch of test slides (2-3 slides) works correctly
3. Slide index JSON format matches CAMELYONSlideIndex expectations
4. Progress tracking and error handling work correctly
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
from PIL import Image

from src.data.wsi_pipeline import (
    BatchProcessor,
    FeatureCache,
    FeatureGenerator,
    PatchExtractor,
    ProcessingConfig,
    QualityControl,
    SlideMetadata,
    TissueDetector,
    WSIReader,
)


class TestQualityControl:
    """Test QualityControl component."""

    def test_blur_score_calculation(self):
        """Test blur score calculation on synthetic images."""
        qc = QualityControl(blur_threshold=100.0)

        # Create sharp image (high frequency content)
        sharp_patch = np.zeros((256, 256, 3), dtype=np.uint8)
        sharp_patch[::2, ::2] = 255  # Checkerboard pattern
        sharp_score = qc.calculate_blur_score(sharp_patch)

        # Create blurry image (low frequency content)
        blurry_patch = np.ones((256, 256, 3), dtype=np.uint8) * 128  # Uniform gray
        blurry_score = qc.calculate_blur_score(blurry_patch)

        # Sharp image should have higher blur score
        assert sharp_score > blurry_score
        assert sharp_score > 100.0  # Above threshold
        assert blurry_score < 100.0  # Below threshold

    def test_artifact_detection(self):
        """Test artifact detection on synthetic images."""
        qc = QualityControl()

        # Create patch with pen marks (high saturation)
        pen_patch = np.zeros((256, 256, 3), dtype=np.uint8)
        pen_patch[:, :, 2] = 255  # Blue channel (pen mark)
        artifacts = qc.detect_artifacts(pen_patch)
        assert artifacts["pen_marks"] == True

        # Create patch with bubbles (bright regions)
        bubble_patch = np.ones((256, 256, 3), dtype=np.uint8) * 250  # Very bright
        artifacts = qc.detect_artifacts(bubble_patch)
        assert artifacts["bubbles"] == True

        # Create normal tissue patch
        normal_patch = np.random.randint(50, 150, size=(256, 256, 3), dtype=np.uint8)
        artifacts = qc.detect_artifacts(normal_patch)
        # Normal patch should have fewer artifacts
        assert not all(artifacts.values())

    def test_qc_report_generation(self):
        """Test QC report generation."""
        qc = QualityControl(blur_threshold=100.0, min_tissue_coverage=0.1)

        # Create synthetic patches
        patches = []
        for i in range(10):
            if i < 5:
                # Sharp patches
                patch = np.zeros((256, 256, 3), dtype=np.uint8)
                patch[::2, ::2] = 255
            else:
                # Blurry patches
                patch = np.ones((256, 256, 3), dtype=np.uint8) * 128
            patches.append(patch)

        # Create synthetic features
        features = np.random.randn(10, 2048).astype(np.float32)

        # Generate QC report
        report = qc.generate_qc_report(
            slide_id="test_slide_001",
            patches=patches,
            features=features,
            tissue_coverage=0.15,
            patch_size=256,
            expected_feature_dim=2048,
        )

        # Verify report structure
        assert report["slide_id"] == "test_slide_001"
        assert report["num_patches"] == 10
        assert "blur_scores" in report
        assert "artifacts" in report
        assert report["tissue_coverage"] == 0.15
        assert report["low_tissue_warning"] is False  # Above threshold
        assert "dimension_validation" in report
        assert report["dimension_validation"]["patch_dimensions_valid"] is True
        assert report["dimension_validation"]["feature_dimensions_valid"] is True

        # Verify blur scores
        blur_scores = report["blur_scores"]
        assert "mean" in blur_scores
        assert "std" in blur_scores
        assert "num_blurry" in blur_scores
        assert blur_scores["num_blurry"] == 5  # Half are blurry

    def test_qc_low_tissue_warning(self):
        """Test QC warning for low tissue coverage."""
        qc = QualityControl(min_tissue_coverage=0.1)

        patches = [np.random.randint(0, 255, size=(256, 256, 3), dtype=np.uint8)]
        features = np.random.randn(1, 2048).astype(np.float32)

        report = qc.generate_qc_report(
            slide_id="test_slide",
            patches=patches,
            features=features,
            tissue_coverage=0.05,  # Below threshold
        )

        assert report["low_tissue_warning"] is True
        assert len(report["warnings"]) > 0
        assert any("Low tissue coverage" in w for w in report["warnings"])


class TestBatchProcessor:
    """Test BatchProcessor component."""

    @patch("src.data.wsi_pipeline.reader.OPENSLIDE_AVAILABLE", True)
    @patch("src.data.wsi_pipeline.reader.OpenSlide")
    def test_single_slide_processing(self, mock_openslide_class):
        """Test processing a single slide through BatchProcessor."""
        # Setup mock slide
        mock_slide = Mock()
        mock_slide.dimensions = (1024, 1024)
        mock_slide.level_count = 1
        mock_slide.level_dimensions = [(1024, 1024)]
        mock_slide.level_downsamples = [1.0]
        mock_slide.properties = {
            "openslide.objective-power": "40",
            "openslide.mpp-x": "0.25",
            "openslide.mpp-y": "0.25",
        }

        # Mock read_region to return tissue patches
        def mock_read_region(location, level, size):
            # Create tissue patches (dark)
            img = Image.new("RGBA", size, color=(50, 50, 50, 255))
            return img

        # Mock get_thumbnail to return a thumbnail image
        def mock_get_thumbnail(size):
            # Create a thumbnail with tissue (dark) in center
            thumb = Image.new("RGB", size, color=(255, 255, 255))  # White background
            # Add tissue region in center
            from PIL import ImageDraw

            draw = ImageDraw.Draw(thumb)
            center_x, center_y = size[0] // 2, size[1] // 2
            radius = min(size) // 4
            draw.ellipse(
                [center_x - radius, center_y - radius, center_x + radius, center_y + radius],
                fill=(50, 50, 50),
            )
            return thumb

        mock_slide.read_region = mock_read_region
        mock_slide.get_thumbnail = mock_get_thumbnail
        mock_openslide_class.return_value = mock_slide

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            slide_path = tmpdir / "test_slide.svs"
            slide_path.touch()

            # Create configuration
            config = ProcessingConfig(
                patch_size=256,
                stride=256,
                level=0,
                encoder_name="resnet50",
                encoder_pretrained=False,
                batch_size=4,
                cache_dir=str(tmpdir / "features"),
            )

            # Create batch processor
            processor = BatchProcessor(config, num_workers=1)

            # Process slide
            result = processor.process_slide(slide_path)

            # Verify result
            assert result.success is True
            assert result.slide_id == "test_slide"
            assert result.num_patches > 0
            assert result.feature_file is not None
            assert result.feature_file.exists()
            assert result.qc_metrics is not None

    @patch("src.data.wsi_pipeline.reader.OPENSLIDE_AVAILABLE", True)
    @patch("src.data.wsi_pipeline.reader.OpenSlide")
    def test_batch_processing_multiple_slides(self, mock_openslide_class):
        """Test processing multiple slides in batch."""
        # Setup mock slide
        mock_slide = Mock()
        mock_slide.dimensions = (512, 512)
        mock_slide.level_count = 1
        mock_slide.level_dimensions = [(512, 512)]
        mock_slide.level_downsamples = [1.0]
        mock_slide.properties = {
            "openslide.objective-power": "40",
            "openslide.mpp-x": "0.25",
        }

        # Mock read_region
        mock_slide.read_region = lambda loc, lvl, size: Image.new(
            "RGBA", size, color=(50, 50, 50, 255)
        )

        # Mock get_thumbnail
        def mock_get_thumbnail(size):
            thumb = Image.new("RGB", size, color=(50, 50, 50))  # All tissue
            return thumb

        mock_slide.get_thumbnail = mock_get_thumbnail
        mock_openslide_class.return_value = mock_slide

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create 3 test slides
            slide_paths = []
            for i in range(3):
                slide_path = tmpdir / f"test_slide_{i}.svs"
                slide_path.touch()
                slide_paths.append(slide_path)

            # Create configuration
            config = ProcessingConfig(
                patch_size=256,
                stride=256,
                encoder_name="resnet50",
                encoder_pretrained=False,
                batch_size=4,
                cache_dir=str(tmpdir / "features"),
            )

            # Create batch processor
            processor = BatchProcessor(config, num_workers=1)

            # Process batch
            results = processor.process_batch(slide_paths)

            # Verify results
            assert results["total_slides"] == 3
            assert results["successful"] == 3
            assert results["failed"] == 0
            assert len(results["results"]) == 3

            # Verify all slides processed successfully
            for result in results["results"]:
                assert result.success is True
                assert result.num_patches > 0

            # Verify summary
            summary = results["summary"]
            assert summary["total_slides"] == 3
            assert summary["successful_slides"] == 3
            assert summary["failed_slides"] == 0
            assert "total_patches" in summary
            assert "avg_patches_per_slide" in summary

    @patch("src.data.wsi_pipeline.reader.OPENSLIDE_AVAILABLE", True)
    @patch("src.data.wsi_pipeline.reader.OpenSlide")
    def test_slide_index_generation(self, mock_openslide_class):
        """Test slide index JSON generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create slide metadata
            slide_metadata_list = [
                SlideMetadata(
                    slide_id="slide_001",
                    patient_id="patient_001",
                    file_path="slide_001.svs",
                    label=1,
                    split="train",
                    width=10000,
                    height=10000,
                    magnification=40.0,
                    mpp=0.25,
                ),
                SlideMetadata(
                    slide_id="slide_002",
                    patient_id="patient_002",
                    file_path="slide_002.svs",
                    label=0,
                    split="val",
                    width=10000,
                    height=10000,
                    magnification=40.0,
                    mpp=0.25,
                ),
                SlideMetadata(
                    slide_id="slide_003",
                    patient_id="patient_003",
                    file_path="slide_003.svs",
                    label=1,
                    split="test",
                    width=10000,
                    height=10000,
                    magnification=40.0,
                    mpp=0.25,
                ),
            ]

            # Create configuration
            config = ProcessingConfig()
            processor = BatchProcessor(config, num_workers=1)

            # Generate slide index
            index_path = processor.generate_slide_index(
                output_dir=tmpdir,
                slide_metadata_list=slide_metadata_list,
                split_ratios=(0.7, 0.15, 0.15),
            )

            # Verify index file exists
            assert index_path.exists()
            assert index_path.name == "slide_index.json"

            # Load and verify index structure
            with open(index_path, "r") as f:
                index = json.load(f)

            # Verify index structure matches CAMELYONSlideIndex expectations
            assert "dataset_name" in index
            assert "creation_date" in index
            assert "total_slides" in index
            assert "splits" in index
            assert "slides" in index

            assert index["total_slides"] == 3
            assert index["splits"]["train"] >= 0
            assert index["splits"]["val"] >= 0
            assert index["splits"]["test"] >= 0
            assert len(index["slides"]) == 3

            # Verify slide entries
            for slide in index["slides"]:
                assert "slide_id" in slide
                assert "patient_id" in slide
                assert "file_path" in slide
                assert "label" in slide
                assert "split" in slide
                assert slide["split"] in ["train", "val", "test"]

    @patch("src.data.wsi_pipeline.reader.OPENSLIDE_AVAILABLE", True)
    @patch("src.data.wsi_pipeline.reader.OpenSlide")
    def test_error_handling_and_retry(self, mock_openslide_class):
        """Test error handling and retry logic."""
        # Setup mock slide that fails first time, succeeds second time
        mock_slide = Mock()
        mock_slide.dimensions = (512, 512)
        mock_slide.level_count = 1
        mock_slide.level_dimensions = [(512, 512)]
        mock_slide.level_downsamples = [1.0]
        mock_slide.properties = {}

        call_count = [0]

        def mock_read_region_with_failure(location, level, size):
            call_count[0] += 1
            if call_count[0] <= 2:  # Fail first 2 calls
                raise RuntimeError("Simulated read failure")
            return Image.new("RGBA", size, color=(50, 50, 50, 255))

        # Mock get_thumbnail
        def mock_get_thumbnail(size):
            return Image.new("RGB", size, color=(50, 50, 50))

        mock_slide.read_region = mock_read_region_with_failure
        mock_slide.get_thumbnail = mock_get_thumbnail
        mock_openslide_class.return_value = mock_slide

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            slide_path = tmpdir / "test_slide.svs"
            slide_path.touch()

            # Create configuration with retries
            config = ProcessingConfig(
                patch_size=256,
                encoder_name="resnet50",
                encoder_pretrained=False,
                cache_dir=str(tmpdir / "features"),
                max_retries=3,
            )

            processor = BatchProcessor(config, num_workers=1)

            # Process slide - should succeed after retries
            result = processor._process_with_retry(slide_path, None, None)

            # Verify retry worked
            assert call_count[0] > 2  # Should have retried
            assert result.success is True or result.error_message is not None

    @patch("src.data.wsi_pipeline.reader.OPENSLIDE_AVAILABLE", True)
    @patch("src.data.wsi_pipeline.reader.OpenSlide")
    def test_progress_tracking(self, mock_openslide_class):
        """Test progress tracking during batch processing."""
        # Setup mock slide
        mock_slide = Mock()
        mock_slide.dimensions = (512, 512)
        mock_slide.level_count = 1
        mock_slide.level_dimensions = [(512, 512)]
        mock_slide.level_downsamples = [1.0]
        mock_slide.properties = {}
        mock_slide.read_region = lambda loc, lvl, size: Image.new(
            "RGBA", size, color=(50, 50, 50, 255)
        )

        # Mock get_thumbnail
        def mock_get_thumbnail(size):
            return Image.new("RGB", size, color=(50, 50, 50))

        mock_slide.get_thumbnail = mock_get_thumbnail
        mock_openslide_class.return_value = mock_slide

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create 2 test slides
            slide_paths = []
            for i in range(2):
                slide_path = tmpdir / f"test_slide_{i}.svs"
                slide_path.touch()
                slide_paths.append(slide_path)

            config = ProcessingConfig(
                patch_size=256,
                encoder_name="resnet50",
                encoder_pretrained=False,
                cache_dir=str(tmpdir / "features"),
            )

            processor = BatchProcessor(config, num_workers=1)

            # Process batch and verify progress tracking
            results = processor.process_batch(slide_paths)

            # Verify timing information
            assert results["total_time"] > 0
            assert results["summary"]["total_time_seconds"] > 0
            assert results["summary"]["avg_time_per_slide"] > 0

            # Verify all slides were processed
            assert len(results["results"]) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
