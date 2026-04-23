"""
End-to-end integration test for WSI processing pipeline.

This test verifies that all processing components (TissueDetector, FeatureGenerator,
FeatureCache) work together correctly for complete pipeline execution.
"""

import tempfile
import numpy as np
import pytest
import torch
from pathlib import Path
from unittest.mock import Mock, patch
from PIL import Image

from src.data.wsi_pipeline import (
    WSIReader,
    PatchExtractor,
    TissueDetector,
    FeatureGenerator,
    FeatureCache,
)


class TestEndToEndPipeline:
    """Test complete pipeline: extract patches → filter tissue → generate features → cache to HDF5."""

    @patch("src.data.wsi_pipeline.reader.OPENSLIDE_AVAILABLE", True)
    @patch("src.data.wsi_pipeline.reader.OpenSlide")
    def test_complete_pipeline_execution(self, mock_openslide_class):
        """Test complete pipeline: extract patches, filter tissue, generate features, cache to HDF5."""
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
        
        # Mock read_region to return patches with varying tissue content
        def mock_read_region(location, level, size):
            x, y = location
            # Create patches with different tissue content based on location
            if x < 512:
                # Left half: tissue (dark)
                img = Image.new("RGBA", size, color=(50, 50, 50, 255))
            else:
                # Right half: background (white)
                img = Image.new("RGBA", size, color=(255, 255, 255, 255))
            return img
        
        mock_slide.read_region = mock_read_region
        mock_openslide_class.return_value = mock_slide
        
        # Create temporary directories
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Create temporary slide file
            slide_path = tmpdir / "test_slide.svs"
            slide_path.touch()
            
            # Create cache directory
            cache_dir = tmpdir / "features"
            cache_dir.mkdir()
            
            try:
                # Step 1: Open slide with WSIReader
                with WSIReader(str(slide_path)) as reader:
                    # Step 2: Create PatchExtractor
                    extractor = PatchExtractor(patch_size=256, stride=256, level=0)
                    
                    # Step 3: Generate coordinates
                    coords = extractor.generate_coordinates(reader.dimensions)
                    assert len(coords) > 0
                    
                    # Step 4: Create TissueDetector
                    detector = TissueDetector(tissue_threshold=0.5)
                    
                    # Step 5: Create FeatureGenerator
                    generator = FeatureGenerator(
                        encoder_name="resnet50",
                        pretrained=False,  # Don't download weights for test
                        device="cpu",
                        batch_size=4
                    )
                    
                    # Step 6: Create FeatureCache
                    cache = FeatureCache(cache_dir=str(cache_dir), compression="gzip")
                    
                    # Step 7: Process patches through pipeline
                    tissue_patches = []
                    tissue_coords = []
                    
                    for patch, coord in extractor.extract_patches_streaming(reader, coords[:8]):
                        # Filter tissue patches
                        if detector.is_tissue_patch(patch):
                            tissue_patches.append(patch)
                            tissue_coords.append(coord)
                    
                    # Verify we found some tissue patches
                    assert len(tissue_patches) > 0
                    assert len(tissue_patches) < len(coords[:8])  # Not all patches should be tissue
                    
                    # Step 8: Generate features for tissue patches
                    if len(tissue_patches) > 0:
                        # Stack patches into batch
                        patches_array = np.stack(tissue_patches, axis=0)
                        
                        # Extract features
                        features = generator.extract_features(patches_array)
                        
                        # Verify features shape
                        assert features.shape[0] == len(tissue_patches)
                        assert features.shape[1] == generator.feature_dim
                        assert features.dtype == torch.float32
                        
                        # Convert to numpy for caching
                        features_np = features.cpu().numpy()
                        coords_np = np.array(tissue_coords, dtype=np.int32)
                        
                        # Step 9: Cache features to HDF5
                        metadata = {
                            "patient_id": "test_patient_001",
                            "scan_date": "2024-01-15",
                            "magnification": 40.0,
                            "mpp": 0.25,
                            "patch_size": 256,
                            "stride": 256,
                            "level": 0,
                            "encoder_name": "resnet50",
                        }
                        
                        cache_path = cache.save_features(
                            slide_id="test_slide_001",
                            features=features_np,
                            coordinates=coords_np,
                            metadata=metadata,
                        )
                        
                        # Verify cache file was created
                        assert cache_path.exists()
                        assert cache_path.suffix == ".h5"
                        
                        # Step 10: Verify HDF5 file structure
                        validation = cache.validate("test_slide_001")
                        assert validation["valid"] is True
                        assert validation["num_patches"] == len(tissue_patches)
                        assert validation["feature_dim"] == generator.feature_dim
                        
                        # Step 11: Load features back and verify
                        loaded_data = cache.load_features("test_slide_001")
                        
                        assert "features" in loaded_data
                        assert "coordinates" in loaded_data
                        assert "metadata" in loaded_data
                        
                        # Verify features match
                        np.testing.assert_array_almost_equal(
                            loaded_data["features"],
                            features_np,
                            decimal=5
                        )
                        
                        # Verify coordinates match
                        np.testing.assert_array_equal(
                            loaded_data["coordinates"],
                            coords_np
                        )
                        
                        # Verify metadata
                        assert loaded_data["metadata"]["slide_id"] == "test_slide_001"
                        assert loaded_data["metadata"]["patient_id"] == "test_patient_001"
                        assert loaded_data["metadata"]["encoder_name"] == "resnet50"
                        assert loaded_data["metadata"]["num_patches"] == len(tissue_patches)
                        
            finally:
                # Cleanup
                pass

    @patch("src.data.wsi_pipeline.reader.OPENSLIDE_AVAILABLE", True)
    @patch("src.data.wsi_pipeline.reader.OpenSlide")
    def test_pipeline_with_no_tissue_patches(self, mock_openslide_class):
        """Test pipeline behavior when no tissue patches are found."""
        # Setup mock slide with all background
        mock_slide = Mock()
        mock_slide.dimensions = (512, 512)
        mock_slide.level_count = 1
        mock_slide.level_dimensions = [(512, 512)]
        mock_slide.level_downsamples = [1.0]
        mock_slide.properties = {}
        
        # All patches are background (white)
        mock_slide.read_region = lambda loc, lvl, size: Image.new("RGBA", size, color=(255, 255, 255, 255))
        mock_openslide_class.return_value = mock_slide
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            slide_path = tmpdir / "test_slide.svs"
            slide_path.touch()
            
            with WSIReader(str(slide_path)) as reader:
                extractor = PatchExtractor(patch_size=256, stride=256, level=0)
                coords = extractor.generate_coordinates(reader.dimensions)
                
                detector = TissueDetector(tissue_threshold=0.5)
                
                tissue_patches = []
                for patch, coord in extractor.extract_patches_streaming(reader, coords[:4]):
                    if detector.is_tissue_patch(patch):
                        tissue_patches.append(patch)
                
                # Should find no tissue patches
                assert len(tissue_patches) == 0

    @patch("src.data.wsi_pipeline.reader.OPENSLIDE_AVAILABLE", True)
    @patch("src.data.wsi_pipeline.reader.OpenSlide")
    def test_pipeline_with_all_tissue_patches(self, mock_openslide_class):
        """Test pipeline behavior when all patches contain tissue."""
        # Setup mock slide with all tissue
        mock_slide = Mock()
        mock_slide.dimensions = (512, 512)
        mock_slide.level_count = 1
        mock_slide.level_dimensions = [(512, 512)]
        mock_slide.level_downsamples = [1.0]
        mock_slide.properties = {}
        
        # All patches are tissue (dark)
        mock_slide.read_region = lambda loc, lvl, size: Image.new("RGBA", size, color=(50, 50, 50, 255))
        mock_openslide_class.return_value = mock_slide
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            slide_path = tmpdir / "test_slide.svs"
            slide_path.touch()
            
            with WSIReader(str(slide_path)) as reader:
                extractor = PatchExtractor(patch_size=256, stride=256, level=0)
                coords = extractor.generate_coordinates(reader.dimensions)
                
                detector = TissueDetector(tissue_threshold=0.5)
                
                tissue_patches = []
                for patch, coord in extractor.extract_patches_streaming(reader, coords[:4]):
                    if detector.is_tissue_patch(patch):
                        tissue_patches.append(patch)
                
                # Should find all patches as tissue
                assert len(tissue_patches) == min(4, len(coords))

    def test_feature_cache_hdf5_structure(self):
        """Test that HDF5 file structure matches design specification."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "features"
            cache_dir.mkdir()
            
            cache = FeatureCache(cache_dir=str(cache_dir), compression="gzip")
            
            # Create synthetic features and coordinates
            num_patches = 100
            feature_dim = 2048
            features = np.random.randn(num_patches, feature_dim).astype(np.float32)
            coordinates = np.random.randint(0, 10000, size=(num_patches, 2), dtype=np.int32)
            
            metadata = {
                "patient_id": "patient_001",
                "scan_date": "2024-01-15",
                "scanner_model": "Aperio ScanScope",
                "magnification": 40.0,
                "mpp": 0.25,
                "patch_size": 256,
                "stride": 256,
                "level": 0,
                "encoder_name": "resnet50",
            }
            
            # Save features
            cache_path = cache.save_features(
                slide_id="test_slide",
                features=features,
                coordinates=coordinates,
                metadata=metadata,
            )
            
            # Verify file exists
            assert cache_path.exists()
            
            # Verify structure using h5py directly
            import h5py
            with h5py.File(cache_path, "r") as f:
                # Check datasets exist
                assert "features" in f
                assert "coordinates" in f
                
                # Check shapes
                assert f["features"].shape == (num_patches, feature_dim)
                assert f["coordinates"].shape == (num_patches, 2)
                
                # Check dtypes
                assert f["features"].dtype == np.float32
                assert f["coordinates"].dtype == np.int32
                
                # Check attributes
                assert "slide_id" in f.attrs
                assert "patient_id" in f.attrs
                assert "num_patches" in f.attrs
                assert f.attrs["num_patches"] == num_patches
                assert f.attrs["encoder_name"] == "resnet50"

    def test_feature_generator_different_encoders(self):
        """Test FeatureGenerator with different encoder architectures."""
        # Test with different encoders
        for encoder_name in ["resnet50", "densenet121", "efficientnet_b0"]:
            generator = FeatureGenerator(
                encoder_name=encoder_name,
                pretrained=False,
                device="cpu",
                batch_size=2
            )
            
            # Create synthetic patches
            patches = np.random.randint(0, 255, size=(4, 256, 256, 3), dtype=np.uint8)
            
            # Extract features
            features = generator.extract_features(patches)
            
            # Verify features shape
            assert features.shape[0] == 4
            assert features.shape[1] == generator.feature_dim
            assert features.dtype == torch.float32
            
            # Verify feature dimension is correct for each encoder
            expected_dims = {
                "resnet50": 2048,
                "densenet121": 1024,
                "efficientnet_b0": 1280,
            }
            assert generator.feature_dim == expected_dims[encoder_name]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
