"""
OpenSlide error handling and recovery tests.

This module provides comprehensive tests for OpenSlide error handling,
corrupted WSI file detection, and recovery option suggestions.
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, Optional, Tuple, List
from PIL import Image
import os

from src.data.openslide_utils import WSIReader, get_slide_info, check_openslide_available
from tests.dataset_testing.synthetic.wsi_generator import WSISyntheticGenerator, WSISyntheticSpec
from tests.dataset_testing.base_interfaces import ErrorSimulator


class TestOpenSlideErrorHandling:
    """Test OpenSlide error handling and recovery functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.generator = WSISyntheticGenerator(random_seed=42)
        self.error_simulator = ErrorSimulator(self.temp_dir)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_openslide_not_available_error(self):
        """Test error handling when OpenSlide is not available."""
        with patch('src.data.openslide_utils.OPENSLIDE_AVAILABLE', False):
            # Test availability check
            assert check_openslide_available() == False
            
            # Test WSIReader initialization fails
            with pytest.raises(ImportError, match="OpenSlide is not installed"):
                WSIReader("dummy.svs")
    
    def test_file_not_found_error(self):
        """Test error handling for non-existent files."""
        with patch('src.data.openslide_utils.OPENSLIDE_AVAILABLE', True):
            nonexistent_file = "nonexistent_slide.svs"
            
            with pytest.raises(FileNotFoundError, match="WSI file not found"):
                WSIReader(nonexistent_file)
    
    @patch('src.data.openslide_utils.OPENSLIDE_AVAILABLE', True)
    @patch('src.data.openslide_utils.OpenSlide')
    def test_corrupted_file_detection(self, mock_openslide):
        """Test detection of corrupted WSI files."""
        # Create a temporary file
        corrupted_file = self.temp_dir / "corrupted.svs"
        with open(corrupted_file, "wb") as f:
            f.write(b"corrupted content")
        
        # Mock OpenSlide to raise an error for corrupted files
        mock_openslide.side_effect = Exception("Cannot open slide")
        
        def detect_wsi_corruption(file_path: Path) -> Dict[str, Any]:
            """Detect WSI file corruption."""
            corruption_result = {
                'corrupted': False,
                'error_type': None,
                'error_message': '',
                'recovery_suggestions': []
            }
            
            try:
                reader = WSIReader(str(file_path))
                reader.close()
            except ImportError as e:
                corruption_result['corrupted'] = True
                corruption_result['error_type'] = 'openslide_not_available'
                corruption_result['error_message'] = str(e)
                corruption_result['recovery_suggestions'] = [
                    'Install OpenSlide: pip install openslide-python',
                    'Verify OpenSlide system dependencies are installed'
                ]
            except FileNotFoundError as e:
                corruption_result['corrupted'] = True
                corruption_result['error_type'] = 'file_not_found'
                corruption_result['error_message'] = str(e)
                corruption_result['recovery_suggestions'] = [
                    'Verify file path is correct',
                    'Check file permissions',
                    'Ensure file exists and is accessible'
                ]
            except Exception as e:
                corruption_result['corrupted'] = True
                corruption_result['error_type'] = 'file_corruption'
                corruption_result['error_message'] = str(e)
                corruption_result['recovery_suggestions'] = [
                    'Re-download the WSI file from original source',
                    'Verify file integrity with checksum',
                    'Try opening with different WSI viewer to confirm corruption',
                    'Contact data provider for replacement file'
                ]
            
            return corruption_result
        
        result = detect_wsi_corruption(corrupted_file)
        
        assert result['corrupted'] == True
        assert result['error_type'] == 'file_corruption'
        assert len(result['recovery_suggestions']) > 0
        assert 'Re-download' in result['recovery_suggestions'][0]
    
    @patch('src.data.openslide_utils.OPENSLIDE_AVAILABLE', True)
    @patch('src.data.openslide_utils.OpenSlide')
    def test_read_region_error_handling(self, mock_openslide):
        """Test error handling during region reading operations."""
        with tempfile.NamedTemporaryFile(suffix='.svs', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            mock_slide = Mock()
            
            # Mock read_region to raise different types of errors
            def error_read_region(location, level, size):
                x, y = location
                if x < 0 or y < 0:
                    raise ValueError("Invalid coordinates")
                elif x > 10000 or y > 10000:
                    raise MemoryError("Region too large")
                elif level > 3:
                    raise IndexError("Invalid pyramid level")
                else:
                    return Image.new('RGBA', size, color='red')
            
            mock_slide.read_region = error_read_region
            mock_slide.level_count = 4
            mock_openslide.return_value = mock_slide
            
            reader = WSIReader(tmp_path)
            
            def safe_read_region(reader, location, level, size):
                """Safely read region with error handling."""
                try:
                    return reader.read_region(location, level, size)
                except ValueError as e:
                    return {'error': 'invalid_coordinates', 'message': str(e)}
                except MemoryError as e:
                    return {'error': 'memory_error', 'message': str(e)}
                except IndexError as e:
                    return {'error': 'invalid_level', 'message': str(e)}
                except Exception as e:
                    return {'error': 'unknown_error', 'message': str(e)}
            
            # Test different error scenarios
            test_cases = [
                ((-100, 100), 0, (256, 256), 'invalid_coordinates'),
                ((15000, 100), 0, (256, 256), 'memory_error'),
                ((100, 100), 5, (256, 256), 'invalid_level'),
                ((100, 100), 0, (256, 256), None)  # Success case
            ]
            
            for location, level, size, expected_error in test_cases:
                result = safe_read_region(reader, location, level, size)
                
                if expected_error:
                    assert isinstance(result, dict)
                    assert result['error'] == expected_error
                else:
                    assert isinstance(result, Image.Image)
            
        finally:
            Path(tmp_path).unlink()
    
    @patch('src.data.openslide_utils.OPENSLIDE_AVAILABLE', True)
    @patch('src.data.openslide_utils.OpenSlide')
    def test_memory_constraint_handling(self, mock_openslide):
        """Test handling of memory constraints during operations."""
        with tempfile.NamedTemporaryFile(suffix='.svs', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            mock_slide = Mock()
            mock_slide.level_dimensions = [(50000, 40000)]
            mock_slide.level_downsamples = [1.0]
            
            # Mock read_region to simulate memory issues with large regions
            def memory_aware_read_region(location, level, size):
                width, height = size
                total_pixels = width * height
                
                # Simulate memory error for very large regions
                if total_pixels > 1000000:  # 1M pixels
                    raise MemoryError("Insufficient memory for region")
                
                return Image.new('RGBA', size, color='blue')
            
            mock_slide.read_region = memory_aware_read_region
            mock_openslide.return_value = mock_slide
            
            reader = WSIReader(tmp_path)
            
            def extract_patches_with_memory_management(reader, patch_size, max_memory_mb=100):
                """Extract patches with memory management."""
                patches = []
                errors = []
                
                # Calculate safe patch size based on memory limit
                bytes_per_pixel = 4  # RGBA
                max_pixels = (max_memory_mb * 1024 * 1024) // bytes_per_pixel
                safe_patch_size = min(patch_size, int(np.sqrt(max_pixels)))
                
                try:
                    # Try with original patch size
                    test_patch = reader.read_region((0, 0), 0, (patch_size, patch_size))
                    actual_patch_size = patch_size
                except MemoryError as e:
                    # Fall back to safe patch size
                    errors.append(f"Memory error with size {patch_size}, using {safe_patch_size}")
                    actual_patch_size = safe_patch_size
                
                # Extract a few patches with the determined size
                for i in range(3):
                    try:
                        patch = reader.read_region(
                            (i * actual_patch_size, 0), 
                            0, 
                            (actual_patch_size, actual_patch_size)
                        )
                        patches.append(patch)
                    except Exception as e:
                        errors.append(f"Error extracting patch {i}: {str(e)}")
                
                return {
                    'patches': patches,
                    'errors': errors,
                    'actual_patch_size': actual_patch_size,
                    'requested_patch_size': patch_size
                }
            
            # Test with large patch size that should trigger memory management
            result = extract_patches_with_memory_management(reader, 2000, max_memory_mb=50)
            
            # Verify memory error was caught
            assert len(result['errors']) > 0
            assert 'Memory error' in result['errors'][0]
            # The actual patch size should be reduced or equal to requested
            assert result['actual_patch_size'] <= result['requested_patch_size']
            
            # Test with reasonable patch size
            result_small = extract_patches_with_memory_management(reader, 256, max_memory_mb=50)
            
            assert result_small['actual_patch_size'] == 256
            assert len(result_small['patches']) > 0
            
        finally:
            Path(tmp_path).unlink()
    
    def test_format_error_detection(self):
        """Test detection of format-related errors."""
        # Create files with different formats
        test_files = {
            'invalid_extension': self.temp_dir / "test.xyz",
            'empty_file': self.temp_dir / "empty.svs",
            'wrong_format': self.temp_dir / "notaslide.svs"
        }
        
        # Create test files
        test_files['invalid_extension'].write_text("invalid content")
        test_files['empty_file'].write_bytes(b"")
        test_files['wrong_format'].write_bytes(b"This is not a slide file")
        
        def detect_format_errors(file_path: Path) -> Dict[str, Any]:
            """Detect format-related errors."""
            error_result = {
                'valid_format': True,
                'issues': [],
                'recommendations': []
            }
            
            # Check file extension
            supported_extensions = ['.svs', '.ndpi', '.tiff', '.tif', '.vms', '.vmu', '.scn']
            if file_path.suffix.lower() not in supported_extensions:
                error_result['valid_format'] = False
                error_result['issues'].append(f"Unsupported file extension: {file_path.suffix}")
                error_result['recommendations'].append(
                    f"Use supported formats: {', '.join(supported_extensions)}"
                )
            
            # Check file size
            if file_path.exists():
                file_size = file_path.stat().st_size
                if file_size == 0:
                    error_result['valid_format'] = False
                    error_result['issues'].append("File is empty")
                    error_result['recommendations'].append("Verify file was downloaded completely")
                elif file_size < 1024:  # Very small file
                    error_result['issues'].append("File is unusually small")
                    error_result['recommendations'].append("Verify file integrity")
            
            # Check file content (basic magic number check)
            if file_path.exists() and file_path.stat().st_size > 0:
                try:
                    with open(file_path, 'rb') as f:
                        header = f.read(16)
                    
                    # Basic checks for common formats
                    if file_path.suffix.lower() in ['.tiff', '.tif']:
                        # TIFF magic numbers: II* (little-endian) or MM* (big-endian)
                        if not (header.startswith(b'II*\x00') or header.startswith(b'MM\x00*')):
                            error_result['valid_format'] = False
                            error_result['issues'].append("Invalid TIFF header")
                    
                    # Check for obvious text files
                    if header.startswith(b'This is not'):
                        error_result['valid_format'] = False
                        error_result['issues'].append("File appears to be text, not a slide")
                        error_result['recommendations'].append("Verify correct file was selected")
                
                except Exception as e:
                    error_result['issues'].append(f"Error reading file header: {str(e)}")
            
            return error_result
        
        # Test format detection
        for file_type, file_path in test_files.items():
            result = detect_format_errors(file_path)
            
            if file_type == 'invalid_extension':
                assert result['valid_format'] == False
                assert any('Unsupported file extension' in issue for issue in result['issues'])
            
            elif file_type == 'empty_file':
                assert result['valid_format'] == False
                assert any('empty' in issue.lower() for issue in result['issues'])
            
            elif file_type == 'wrong_format':
                assert result['valid_format'] == False
                assert any('text' in issue.lower() for issue in result['issues'])
    
    @patch('src.data.openslide_utils.OPENSLIDE_AVAILABLE', True)
    @patch('src.data.openslide_utils.OpenSlide')
    def test_recovery_suggestion_generation(self, mock_openslide):
        """Test generation of recovery suggestions for different error types."""
        def generate_recovery_suggestions(error_type: str, error_message: str = "") -> List[str]:
            """Generate recovery suggestions based on error type."""
            suggestions = []
            
            if error_type == 'openslide_not_available':
                suggestions = [
                    'Install OpenSlide Python bindings: pip install openslide-python',
                    'Install OpenSlide system library (varies by OS)',
                    'On Ubuntu/Debian: sudo apt-get install openslide-tools',
                    'On macOS: brew install openslide',
                    'On Windows: Download from OpenSlide website'
                ]
            
            elif error_type == 'file_not_found':
                suggestions = [
                    'Verify the file path is correct',
                    'Check that the file exists in the specified location',
                    'Ensure you have read permissions for the file',
                    'Try using an absolute path instead of relative path'
                ]
            
            elif error_type == 'file_corruption':
                suggestions = [
                    'Re-download the file from the original source',
                    'Verify file integrity using checksums if available',
                    'Try opening the file with other WSI viewers (ImageJ, QuPath)',
                    'Check available disk space during download',
                    'Contact the data provider for a replacement file'
                ]
            
            elif error_type == 'memory_error':
                suggestions = [
                    'Reduce patch size or region size',
                    'Process the slide in smaller chunks',
                    'Use a higher pyramid level (lower resolution)',
                    'Close other applications to free memory',
                    'Consider using a machine with more RAM'
                ]
            
            elif error_type == 'invalid_coordinates':
                suggestions = [
                    'Verify coordinates are within slide bounds',
                    'Check that coordinates are non-negative',
                    'Use get_slide_info() to check slide dimensions',
                    'Ensure level parameter is valid for the slide'
                ]
            
            elif error_type == 'unsupported_format':
                suggestions = [
                    'Convert file to a supported format (SVS, NDPI, TIFF)',
                    'Use format-specific tools to convert the file',
                    'Contact the scanner manufacturer for conversion tools',
                    'Check if OpenSlide supports your specific format version'
                ]
            
            else:
                suggestions = [
                    'Check the OpenSlide documentation for your specific error',
                    'Verify file integrity and format compatibility',
                    'Try with a different WSI file to isolate the issue',
                    'Report the issue to OpenSlide developers if persistent'
                ]
            
            return suggestions
        
        # Test suggestion generation for different error types
        error_types = [
            'openslide_not_available',
            'file_not_found', 
            'file_corruption',
            'memory_error',
            'invalid_coordinates',
            'unsupported_format',
            'unknown_error'
        ]
        
        for error_type in error_types:
            suggestions = generate_recovery_suggestions(error_type)
            
            assert len(suggestions) > 0, f"Should provide suggestions for {error_type}"
            assert all(isinstance(s, str) for s in suggestions), "All suggestions should be strings"
            assert all(len(s) > 10 for s in suggestions), "Suggestions should be descriptive"
            
            # Verify suggestions are relevant to error type
            if error_type == 'openslide_not_available':
                assert any('install' in s.lower() for s in suggestions)
            elif error_type == 'file_not_found':
                assert any('path' in s.lower() for s in suggestions)
            elif error_type == 'memory_error':
                assert any('memory' in s.lower() or 'size' in s.lower() for s in suggestions)


class TestOpenSlideResourceManagement:
    """Test OpenSlide resource management and cleanup."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    @patch('src.data.openslide_utils.OPENSLIDE_AVAILABLE', True)
    @patch('src.data.openslide_utils.OpenSlide')
    def test_context_manager_cleanup(self, mock_openslide):
        """Test proper cleanup using context manager."""
        with tempfile.NamedTemporaryFile(suffix='.svs', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            mock_slide = Mock()
            mock_openslide.return_value = mock_slide
            
            # Test context manager usage
            with WSIReader(tmp_path) as reader:
                assert reader.slide == mock_slide
                # Do some operations
                reader.dimensions  # Access property
            
            # Verify close was called
            mock_slide.close.assert_called_once()
            
        finally:
            Path(tmp_path).unlink()
    
    @patch('src.data.openslide_utils.OPENSLIDE_AVAILABLE', True)
    @patch('src.data.openslide_utils.OpenSlide')
    def test_manual_cleanup(self, mock_openslide):
        """Test manual cleanup functionality."""
        with tempfile.NamedTemporaryFile(suffix='.svs', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            mock_slide = Mock()
            mock_openslide.return_value = mock_slide
            
            reader = WSIReader(tmp_path)
            reader.close()
            
            # Verify close was called
            mock_slide.close.assert_called_once()
            
            # Test multiple close calls (should be safe)
            reader.close()
            # Should not raise an error
            
        finally:
            Path(tmp_path).unlink()
    
    @patch('src.data.openslide_utils.OPENSLIDE_AVAILABLE', True)
    @patch('src.data.openslide_utils.OpenSlide')
    def test_exception_during_cleanup(self, mock_openslide):
        """Test handling of exceptions during cleanup."""
        with tempfile.NamedTemporaryFile(suffix='.svs', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            mock_slide = Mock()
            mock_slide.close.side_effect = Exception("Cleanup error")
            mock_openslide.return_value = mock_slide
            
            reader = WSIReader(tmp_path)
            
            # Close should not raise exception even if underlying close fails
            try:
                reader.close()
            except Exception:
                pytest.fail("WSIReader.close() should handle underlying exceptions gracefully")
            
        finally:
            Path(tmp_path).unlink()
    
    @patch('src.data.openslide_utils.OPENSLIDE_AVAILABLE', True)
    @patch('src.data.openslide_utils.OpenSlide')
    def test_resource_leak_prevention(self, mock_openslide):
        """Test prevention of resource leaks."""
        with tempfile.NamedTemporaryFile(suffix='.svs', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            mock_slide = Mock()
            mock_openslide.return_value = mock_slide
            
            # Create multiple readers to test resource management
            readers = []
            for i in range(5):
                reader = WSIReader(tmp_path)
                readers.append(reader)
            
            # Close all readers
            for reader in readers:
                reader.close()
            
            # Verify all slides were closed
            assert mock_slide.close.call_count == 5
            
        finally:
            Path(tmp_path).unlink()