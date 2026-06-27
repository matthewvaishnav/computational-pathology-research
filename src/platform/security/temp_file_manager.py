"""Secure temporary file helpers."""

from __future__ import annotations

import os
import tempfile


class TempFileManager:
    """Create temporary files and directories with owner-only permissions."""

    @staticmethod
    def create_temp_file(
        suffix: str | None = None,
        prefix: str | None = None,
        dir: str | None = None,
    ) -> tuple[int, str]:
        fd, path = tempfile.mkstemp(
            suffix=suffix or "",
            prefix=prefix or "tmp",
            dir=dir,
        )
        os.chmod(path, 0o600)
        return fd, path

    @staticmethod
    def create_temp_directory(
        suffix: str | None = None,
        prefix: str | None = None,
        dir: str | None = None,
    ) -> tempfile.TemporaryDirectory:
        temp_dir = tempfile.TemporaryDirectory(
            suffix=suffix or "",
            prefix=prefix or "tmp",
            dir=dir,
        )
        os.chmod(temp_dir.name, 0o700)
        return temp_dir

    @staticmethod
    def secure_delete(path: str) -> None:
        try:
            os.remove(path)
        except FileNotFoundError:
            return
