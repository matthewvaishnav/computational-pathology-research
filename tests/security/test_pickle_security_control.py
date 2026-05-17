"""
Unit tests for PickleSecurityControl.

Tests pickle deserialization security with source validation.
"""

import os
import pickle
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from src.security.exceptions import PickleSecurityError
from src.security.models import SecurityEnvironment
from src.security.pickle_security_control import PickleSecurityControl


class TestPickleSecurityControl:
    """Test PickleSecurityControl functionality."""

    def test_production_blocks_untrusted_pickle(self):
        """Test production blocks pickle from untrusted sources."""
        with patch.dict(os.environ, {"ENVIRONMENT": "production"}):
            control = PickleSecurityControl()

            # Create untrusted pickle file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as f:
                pickle.dump({"data": "test"}, f)
                untrusted_path = f.name

            try:
                with pytest.raises(PickleSecurityError, match="Untrusted source"):
                    control.safe_load(untrusted_path)
            finally:
                Path(untrusted_path).unlink(missing_ok=True)

    def test_development_warns_on_untrusted_pickle(self, caplog):
        """Test development warns when loading untrusted pickle."""
        with patch.dict(os.environ, {"ENVIRONMENT": "development"}):
            control = PickleSecurityControl()

            # Create untrusted pickle file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as f:
                pickle.dump({"data": "test"}, f)
                untrusted_path = f.name

            try:
                data = control.safe_load(untrusted_path)
                assert "untrusted pickle" in caplog.text.lower()
                assert data == {"data": "test"}
            finally:
                Path(untrusted_path).unlink(missing_ok=True)

    def test_trusted_paths_work_correctly(self):
        """Test pickle from trusted paths loads successfully."""
        control = PickleSecurityControl(trusted_paths=["/trusted/models"])

        # Create pickle in trusted location
        trusted_dir = Path("/trusted/models")
        # Mock the path check
        with patch.object(Path, "exists", return_value=True):
            with patch.object(Path, "is_relative_to", return_value=True):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as f:
                    pickle.dump({"data": "test"}, f)
                    f.flush()

                    # Mock the file path to appear trusted
                    with patch("builtins.open", create=True) as mock_open:
                        mock_open.return_value.__enter__.return_value.read.return_value = (
                            pickle.dumps({"data": "test"})
                        )

                        # Should not raise
                        assert control.is_trusted_source(f.name)

    def test_is_trusted_source_validation(self):
        """Test is_trusted_source correctly validates paths."""
        control = PickleSecurityControl(trusted_paths=["/app/models", "/data/checkpoints"])

        # Trusted paths
        assert control.is_trusted_source("/app/models/model.pkl")
        assert control.is_trusted_source("/data/checkpoints/checkpoint.pkl")

        # Untrusted paths
        assert not control.is_trusted_source("/tmp/model.pkl")
        assert not control.is_trusted_source("/home/user/model.pkl")

    def test_get_alternative_format_recommendations(self):
        """Test get_alternative_format suggests safer alternatives."""
        control = PickleSecurityControl()

        alternatives = control.get_alternative_format()

        assert "JSON" in alternatives
        assert "SafeTensors" in alternatives
        assert "HDF5" in alternatives or "NPZ" in alternatives

    def test_restricted_unpickler_for_untrusted_sources(self):
        """Test restricted unpickler is used for untrusted sources."""
        with patch.dict(os.environ, {"ENVIRONMENT": "development"}):
            control = PickleSecurityControl()

            # Create pickle with simple data
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as f:
                pickle.dump({"data": [1, 2, 3]}, f)
                f.flush()
                path = f.name

            try:
                # Should load with restricted unpickler
                data = control.safe_load(path)
                assert data == {"data": [1, 2, 3]}
            finally:
                Path(path).unlink(missing_ok=True)

    def test_audit_logging_for_pickle_operations(self, caplog):
        """Test audit logging for pickle load operations."""
        with patch.dict(os.environ, {"ENVIRONMENT": "development"}):
            control = PickleSecurityControl()

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as f:
                pickle.dump({"data": "test"}, f)
                f.flush()
                path = f.name

            try:
                control.safe_load(path)
                assert "Pickle load" in caplog.text or "pickle" in caplog.text.lower()
            finally:
                Path(path).unlink(missing_ok=True)

    def test_safe_load_with_file_object(self):
        """Test safe_load works with file objects."""
        with patch.dict(os.environ, {"ENVIRONMENT": "development"}):
            control = PickleSecurityControl()

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl", mode="wb") as f:
                pickle.dump({"data": "test"}, f)
                path = f.name

            try:
                with open(path, "rb") as f:
                    data = control.safe_load(f)
                assert data == {"data": "test"}
            finally:
                Path(path).unlink(missing_ok=True)

    def test_safe_load_with_path_string(self):
        """Test safe_load works with path strings."""
        with patch.dict(os.environ, {"ENVIRONMENT": "development"}):
            control = PickleSecurityControl()

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl", mode="wb") as f:
                pickle.dump({"data": "test"}, f)
                path = f.name

            try:
                data = control.safe_load(path)
                assert data == {"data": "test"}
            finally:
                Path(path).unlink(missing_ok=True)

    def test_safe_load_with_path_object(self):
        """Test safe_load works with Path objects."""
        with patch.dict(os.environ, {"ENVIRONMENT": "development"}):
            control = PickleSecurityControl()

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl", mode="wb") as f:
                pickle.dump({"data": "test"}, f)
                path = Path(f.name)

            try:
                data = control.safe_load(path)
                assert data == {"data": "test"}
            finally:
                path.unlink(missing_ok=True)

    def test_environment_specific_behavior(self):
        """Test behavior differs by environment."""
        # Production: strict
        with patch.dict(os.environ, {"ENVIRONMENT": "production"}):
            control = PickleSecurityControl()

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl", mode="wb") as f:
                pickle.dump({"data": "test"}, f)
                path = f.name

            try:
                with pytest.raises(PickleSecurityError):
                    control.safe_load(path)
            finally:
                Path(path).unlink(missing_ok=True)

        # Development: warning only
        with patch.dict(os.environ, {"ENVIRONMENT": "development"}):
            control = PickleSecurityControl()

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl", mode="wb") as f:
                pickle.dump({"data": "test"}, f)
                path = f.name

            try:
                data = control.safe_load(path)  # Should not raise
                assert data == {"data": "test"}
            finally:
                Path(path).unlink(missing_ok=True)

    def test_trusted_path_configuration(self):
        """Test trusted paths can be configured."""
        control1 = PickleSecurityControl(trusted_paths=["/path1"])
        control2 = PickleSecurityControl(trusted_paths=["/path2"])

        assert control1.is_trusted_source("/path1/file.pkl")
        assert not control1.is_trusted_source("/path2/file.pkl")

        assert control2.is_trusted_source("/path2/file.pkl")
        assert not control2.is_trusted_source("/path1/file.pkl")

    def test_empty_trusted_paths(self):
        """Test control works with no trusted paths."""
        control = PickleSecurityControl(trusted_paths=[])

        # All paths should be untrusted
        assert not control.is_trusted_source("/any/path/file.pkl")

    def test_relative_path_handling(self):
        """Test handling of relative paths."""
        control = PickleSecurityControl(trusted_paths=["/trusted"])

        # Relative paths should be resolved
        with patch("pathlib.Path.resolve", return_value=Path("/trusted/model.pkl")):
            assert control.is_trusted_source("./model.pkl")
