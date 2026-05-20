"""
Unit tests for ModelDownloadManager.

Tests model download security with revision pinning.
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from src.platform.security.exceptions import ModelSecurityError
from src.platform.security.model_download_manager import ModelDownloadManager
from src.platform.security.models import SecurityEnvironment


class TestModelDownloadManager:
    """Test ModelDownloadManager functionality."""

    def test_production_requires_pinned_revisions(self):
        """Test production requires pinned model revisions."""
        with patch.dict(os.environ, {"ENVIRONMENT": "production"}):
            manager = ModelDownloadManager()

            with pytest.raises(ModelSecurityError, match="Pinned revision required"):
                manager.download_model("owkin/phikon", revision=None)

    def test_production_allows_pinned_revisions(self):
        """Test production allows downloads with pinned revisions."""
        with patch.dict(os.environ, {"ENVIRONMENT": "production"}):
            manager = ModelDownloadManager()

            # Should not raise
            revision = manager.validate_revision("owkin/phikon", "abc123def456")
            assert revision == "abc123def456"

    def test_development_warns_on_unpinned_models(self, caplog):
        """Test development warns when using unpinned models."""
        with patch.dict(os.environ, {"ENVIRONMENT": "development"}):
            manager = ModelDownloadManager()

            manager.download_model("owkin/phikon", revision=None)
            assert "unpinned model" in caplog.text.lower()

    def test_get_pinned_revision_from_config(self):
        """Test getting pinned revision from configuration."""
        manager = ModelDownloadManager()

        # Mock config with pinned revisions
        with patch.object(
            manager,
            "_load_revision_config",
            return_value={"owkin/phikon": "abc123", "microsoft/resnet-50": "def456"},
        ):
            assert manager.get_pinned_revision("owkin/phikon") == "abc123"
            assert manager.get_pinned_revision("microsoft/resnet-50") == "def456"

    def test_get_pinned_revision_missing_model(self):
        """Test getting pinned revision for model not in config."""
        manager = ModelDownloadManager()

        with patch.object(manager, "_load_revision_config", return_value={}):
            with pytest.raises(ModelSecurityError, match="No pinned revision"):
                manager.get_pinned_revision("unknown/model")

    def test_validate_revision_format(self):
        """Test revision format validation."""
        manager = ModelDownloadManager()

        # Valid revisions (git commit hashes)
        valid_revisions = [
            "abc123def456",
            "1234567890abcdef",
            "a" * 40,  # Full SHA-1
        ]

        for revision in valid_revisions:
            assert manager.validate_revision("model/name", revision) == revision

    def test_validate_revision_invalid_format(self):
        """Test invalid revision format is rejected."""
        manager = ModelDownloadManager()

        invalid_revisions = [
            "",
            "main",  # Branch names not allowed
            "v1.0",  # Tags not allowed
            "latest",
            "abc",  # Too short
        ]

        for revision in invalid_revisions:
            with pytest.raises(ModelSecurityError, match="Invalid revision"):
                manager.validate_revision("model/name", revision)

    def test_download_model_with_revision(self):
        """Test downloading model with pinned revision."""
        manager = ModelDownloadManager()

        with patch("transformers.AutoModel.from_pretrained") as mock_download:
            manager.download_model("owkin/phikon", revision="abc123")

            mock_download.assert_called_once_with("owkin/phikon", revision="abc123")

    def test_audit_logging_for_downloads(self, caplog):
        """Test audit logging for model downloads."""
        with patch.dict(os.environ, {"ENVIRONMENT": "production"}):
            manager = ModelDownloadManager()

            with patch("transformers.AutoModel.from_pretrained"):
                manager.download_model("owkin/phikon", revision="abc123")

            assert "Model download" in caplog.text
            assert "owkin/phikon" in caplog.text
            assert "abc123" in caplog.text

    def test_revision_config_loading(self):
        """Test loading revision configuration from file."""
        config_content = """
owkin/phikon: abc123def456
microsoft/resnet-50: def456ghi789
facebook/dino-vitb16: ghi789jkl012
"""
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_content)
            config_path = f.name

        try:
            manager = ModelDownloadManager(revision_config_path=config_path)
            assert manager.get_pinned_revision("owkin/phikon") == "abc123def456"
        finally:
            os.unlink(config_path)

    def test_download_with_auto_revision_lookup(self):
        """Test download automatically looks up pinned revision."""
        manager = ModelDownloadManager()

        with patch.object(manager, "get_pinned_revision", return_value="abc123"):
            with patch("transformers.AutoModel.from_pretrained") as mock_download:
                manager.download_model_auto("owkin/phikon")

                mock_download.assert_called_once_with("owkin/phikon", revision="abc123")

    def test_multiple_model_downloads(self):
        """Test downloading multiple models with different revisions."""
        manager = ModelDownloadManager()

        models = [
            ("owkin/phikon", "abc123"),
            ("microsoft/resnet-50", "def456"),
            ("facebook/dino-vitb16", "ghi789"),
        ]

        with patch("transformers.AutoModel.from_pretrained") as mock_download:
            for model_name, revision in models:
                manager.download_model(model_name, revision=revision)

            assert mock_download.call_count == 3

    def test_revision_caching(self):
        """Test revision lookups are cached."""
        manager = ModelDownloadManager()

        with patch.object(
            manager, "_load_revision_config", return_value={"owkin/phikon": "abc123"}
        ) as mock_load:
            # First call
            rev1 = manager.get_pinned_revision("owkin/phikon")
            # Second call
            rev2 = manager.get_pinned_revision("owkin/phikon")

            assert rev1 == rev2 == "abc123"
            # Config should only be loaded once
            assert mock_load.call_count == 1

    def test_environment_specific_behavior(self):
        """Test behavior differs by environment."""
        # Production: strict
        with patch.dict(os.environ, {"ENVIRONMENT": "production"}):
            manager = ModelDownloadManager()
            with pytest.raises(ModelSecurityError):
                manager.download_model("model/name", revision=None)

        # Development: warning only
        with patch.dict(os.environ, {"ENVIRONMENT": "development"}):
            manager = ModelDownloadManager()
            with patch("transformers.AutoModel.from_pretrained"):
                manager.download_model("model/name", revision=None)  # Should not raise

    def test_download_with_additional_kwargs(self):
        """Test download passes through additional kwargs."""
        manager = ModelDownloadManager()

        with patch("transformers.AutoModel.from_pretrained") as mock_download:
            manager.download_model(
                "owkin/phikon", revision="abc123", trust_remote_code=False, cache_dir="/tmp/cache"
            )

            mock_download.assert_called_once_with(
                "owkin/phikon", revision="abc123", trust_remote_code=False, cache_dir="/tmp/cache"
            )
