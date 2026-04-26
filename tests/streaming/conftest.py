"""Pytest configuration for streaming tests."""

import sys
from unittest.mock import MagicMock

# Mock openslide before any imports to avoid DLL dependency
mock_openslide = MagicMock()
mock_openslide.deepzoom = MagicMock()
mock_openslide.deepzoom.DeepZoomGenerator = MagicMock()
sys.modules['openslide'] = mock_openslide
sys.modules['openslide.deepzoom'] = mock_openslide.deepzoom
