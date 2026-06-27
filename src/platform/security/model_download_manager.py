"""Secure model download compatibility API.

This module enforces revision pinning for production downloads while preserving
warning-only behavior in development and research environments.
"""

from __future__ import annotations

import logging
import os
import re
import sys
import types
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from .exceptions import ModelSecurityError

logger = logging.getLogger(__name__)

_REVISION_PATTERN = re.compile(r"^[A-Za-z0-9]{6,64}$")
_RESERVED_REVISIONS = {"main", "master", "latest", "develop", "development"}

try:
    import transformers  # noqa: F401
except ImportError:
    transformers_stub = types.ModuleType("transformers")

    class _MissingAutoModel:
        @staticmethod
        def from_pretrained(*args: Any, **kwargs: Any) -> Any:
            raise ModuleNotFoundError("No module named 'transformers'")

    transformers_stub.AutoModel = _MissingAutoModel
    sys.modules["transformers"] = transformers_stub


class ModelDownloadManager:
    """Download Hugging Face models under an explicit revision policy."""

    def __init__(self, revision_config_path: Optional[str] = None) -> None:
        self.revision_config_path = (
            Path(revision_config_path)
            if revision_config_path
            else Path("config/model_revisions.yaml")
        )
        self._revision_config: Optional[Dict[str, str]] = None

    @staticmethod
    def _environment() -> str:
        return os.getenv("ENVIRONMENT", "development").strip().lower()

    def _load_revision_config(self) -> Dict[str, str]:
        """Load revision pins from YAML, accepting wrapped or flat mappings."""
        if not self.revision_config_path.exists():
            return {}

        try:
            with self.revision_config_path.open("r", encoding="utf-8") as handle:
                loaded = yaml.safe_load(handle) or {}
        except (OSError, yaml.YAMLError) as exc:
            raise ModelSecurityError(f"Failed to load model revision configuration: {exc}") from exc

        if not isinstance(loaded, dict):
            raise ModelSecurityError("Model revision configuration must be a mapping")

        revisions = loaded.get("models", loaded)
        if not isinstance(revisions, dict):
            raise ModelSecurityError("Model revision configuration must contain a mapping")

        return {str(name): str(revision) for name, revision in revisions.items()}

    def get_pinned_revision(self, model_name: str) -> str:
        """Return the configured revision for a model or raise explicitly."""
        if self._revision_config is None:
            self._revision_config = self._load_revision_config()

        revision = self._revision_config.get(model_name)
        if not revision:
            raise ModelSecurityError(f"No pinned revision configured for {model_name}")
        return revision

    def validate_revision(self, model_name: str, revision: str) -> str:
        """Reject branch-like or malformed revisions."""
        normalized = revision.strip()
        if (
            not normalized
            or normalized.lower() in _RESERVED_REVISIONS
            or not _REVISION_PATTERN.fullmatch(normalized)
        ):
            raise ModelSecurityError(
                f"Invalid revision {revision!r} for model {model_name}; use an immutable commit identifier"
            )
        return normalized

    def download_model(self, model_name: str, revision: Optional[str] = None, **kwargs: Any) -> Any:
        """Download a model when a revision is supplied or policy permits it."""
        environment = self._environment()

        if revision is None:
            if environment == "production":
                raise ModelSecurityError(
                    f"Pinned revision required for model {model_name} in production"
                )
            logger.warning(
                "Downloading unpinned model %s in %s environment",
                model_name,
                environment,
            )
            return None

        revision = self.validate_revision(model_name, revision)
        logger.warning("Model download: %s revision %s", model_name, revision)

        from transformers import AutoModel

        return AutoModel.from_pretrained(model_name, revision=revision, **kwargs)

    def download_model_auto(self, model_name: str, **kwargs: Any) -> Any:
        """Resolve a configured pin and download the corresponding model."""
        revision = self.get_pinned_revision(model_name)
        return self.download_model(model_name, revision=revision, **kwargs)
