"""
Unit tests for TempFileManager.

Tests secure temporary file creation with proper permissions and cleanup.
"""

import os
import stat
import tempfile
import pytest
from pathlib import Path

from src.security.temp_file_manager import TempFileManager


class TestTempFileManager:
    """Test TempFileManager functionality."""

    def test_create_temp_file_returns_fd_and_path(self):
        """Test create_temp_file returns file descriptor and path."""
        fd, path = TempFileManager.create_temp_file()

        try:
            assert isinstance(fd, int)
            assert isinstance(path, str)
            assert os.path.exists(path)
        finally:
            os.close(fd)
            Path(path).unlink(missing_ok=True)

    def test_temp_file_has_secure_permissions(self):
        """Test temp files have 0o600 permissions (owner read/write only)."""
        fd, path = TempFileManager.create_temp_file()

        try:
            # Get file permissions
            file_stat = os.stat(path)
            permissions = stat.S_IMODE(file_stat.st_mode)

            # Should be 0o600 (owner read/write only)
            assert permissions == 0o600
        finally:
            os.close(fd)
            Path(path).unlink(missing_ok=True)

    def test_create_temp_directory_returns_path(self):
        """Test create_temp_directory returns directory path."""
        temp_dir = TempFileManager.create_temp_directory()

        try:
            assert isinstance(temp_dir, tempfile.TemporaryDirectory)
            assert os.path.exists(temp_dir.name)
            assert os.path.isdir(temp_dir.name)
        finally:
            temp_dir.cleanup()

    def test_temp_directory_has_secure_permissions(self):
        """Test temp directories have 0o700 permissions (owner only)."""
        temp_dir = TempFileManager.create_temp_directory()

        try:
            # Get directory permissions
            dir_stat = os.stat(temp_dir.name)
            permissions = stat.S_IMODE(dir_stat.st_mode)

            # Should be 0o700 (owner read/write/execute only)
            assert permissions == 0o700
        finally:
            temp_dir.cleanup()

    def test_temp_file_with_suffix(self):
        """Test creating temp file with custom suffix."""
        fd, path = TempFileManager.create_temp_file(suffix=".txt")

        try:
            assert path.endswith(".txt")
        finally:
            os.close(fd)
            Path(path).unlink(missing_ok=True)

    def test_temp_file_with_prefix(self):
        """Test creating temp file with custom prefix."""
        fd, path = TempFileManager.create_temp_file(prefix="test_")

        try:
            filename = os.path.basename(path)
            assert filename.startswith("test_")
        finally:
            os.close(fd)
            Path(path).unlink(missing_ok=True)

    def test_temp_directory_with_suffix(self):
        """Test creating temp directory with custom suffix."""
        temp_dir = TempFileManager.create_temp_directory(suffix="_test")

        try:
            assert temp_dir.name.endswith("_test")
        finally:
            temp_dir.cleanup()

    def test_temp_directory_with_prefix(self):
        """Test creating temp directory with custom prefix."""
        temp_dir = TempFileManager.create_temp_directory(prefix="test_")

        try:
            dirname = os.path.basename(temp_dir.name)
            assert dirname.startswith("test_")
        finally:
            temp_dir.cleanup()

    def test_no_hardcoded_tmp_paths(self):
        """Test temp files don't use hardcoded /tmp paths."""
        fd, path = TempFileManager.create_temp_file()

        try:
            # Path should use system temp directory, not hardcoded /tmp
            assert path.startswith(tempfile.gettempdir())
        finally:
            os.close(fd)
            Path(path).unlink(missing_ok=True)

    def test_automatic_cleanup_with_context_manager(self):
        """Test temp directory automatically cleans up with context manager."""
        temp_dir = TempFileManager.create_temp_directory()
        dir_path = temp_dir.name

        # Directory exists
        assert os.path.exists(dir_path)

        # Cleanup
        temp_dir.cleanup()

        # Directory removed
        assert not os.path.exists(dir_path)

    def test_secure_delete_removes_file(self):
        """Test secure_delete removes file."""
        fd, path = TempFileManager.create_temp_file()
        os.close(fd)

        # File exists
        assert os.path.exists(path)

        # Secure delete
        TempFileManager.secure_delete(path)

        # File removed
        assert not os.path.exists(path)

    def test_secure_delete_handles_nonexistent_file(self):
        """Test secure_delete handles nonexistent files gracefully."""
        # Should not raise
        TempFileManager.secure_delete("/nonexistent/file.txt")

    def test_multiple_temp_files_independent(self):
        """Test multiple temp files are independent."""
        fd1, path1 = TempFileManager.create_temp_file()
        fd2, path2 = TempFileManager.create_temp_file()

        try:
            assert path1 != path2
            assert os.path.exists(path1)
            assert os.path.exists(path2)
        finally:
            os.close(fd1)
            os.close(fd2)
            Path(path1).unlink(missing_ok=True)
            Path(path2).unlink(missing_ok=True)

    def test_temp_file_writable(self):
        """Test temp files are writable."""
        fd, path = TempFileManager.create_temp_file()

        try:
            # Write to file
            os.write(fd, b"test data")
            os.close(fd)

            # Read back
            with open(path, "rb") as f:
                data = f.read()

            assert data == b"test data"
        finally:
            Path(path).unlink(missing_ok=True)

    def test_temp_directory_writable(self):
        """Test temp directories are writable."""
        temp_dir = TempFileManager.create_temp_directory()

        try:
            # Create file in directory
            test_file = Path(temp_dir.name) / "test.txt"
            test_file.write_text("test data")

            assert test_file.exists()
            assert test_file.read_text() == "test data"
        finally:
            temp_dir.cleanup()

    def test_temp_file_in_custom_directory(self):
        """Test creating temp file in custom directory."""
        custom_dir = TempFileManager.create_temp_directory()

        try:
            fd, path = TempFileManager.create_temp_file(dir=custom_dir.name)

            try:
                # File should be in custom directory
                assert Path(path).parent == Path(custom_dir.name)
            finally:
                os.close(fd)
                Path(path).unlink(missing_ok=True)
        finally:
            custom_dir.cleanup()

    def test_permissions_not_world_readable(self):
        """Test temp files are not world-readable."""
        fd, path = TempFileManager.create_temp_file()

        try:
            file_stat = os.stat(path)
            permissions = stat.S_IMODE(file_stat.st_mode)

            # Should not have world read permission
            assert not (permissions & stat.S_IROTH)
            # Should not have world write permission
            assert not (permissions & stat.S_IWOTH)
            # Should not have world execute permission
            assert not (permissions & stat.S_IXOTH)
        finally:
            os.close(fd)
            Path(path).unlink(missing_ok=True)

    def test_permissions_not_group_readable(self):
        """Test temp files are not group-readable."""
        fd, path = TempFileManager.create_temp_file()

        try:
            file_stat = os.stat(path)
            permissions = stat.S_IMODE(file_stat.st_mode)

            # Should not have group read permission
            assert not (permissions & stat.S_IRGRP)
            # Should not have group write permission
            assert not (permissions & stat.S_IWGRP)
            # Should not have group execute permission
            assert not (permissions & stat.S_IXGRP)
        finally:
            os.close(fd)
            Path(path).unlink(missing_ok=True)
