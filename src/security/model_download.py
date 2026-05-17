"""
Model Download Manager for secure HuggingFace model downloads.

This module provides security controls for downloading models from HuggingFace Hub,
ensuring that models use pinned revisions to prevent supply chain attacks.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)


class ModelDownloadManager:
    """
    Manages secure HuggingFace model downloads with revision pinning.

    This class enforces security policies for model downloads based on the
    deployment environment:
    - Production: Requires pinned revisions (raises error if missing)
    - Development: Warns about unpinned revisions but allows download
    - Research: Recommends pinned revisions with warnings
    """

    _revision_config: Optional[Dict[str, str]] = None
    _config_path: Optional[Path] = None

    @classmethod
    def get_pinned_revision(cls, repo_id: str) -> Optional[str]:
        """
        Get the pinned revision for a model from configuration.

        Args:
            repo_id: HuggingFace repository ID (e.g., "owkin/phikon")

        Returns:
            Pinned revision hash/tag if configured, None otherwise
        """
        if cls._revision_config is None:
            cls._load_revision_config()

        revision = cls._revision_config.get(repo_id)

        if revision:
            logger.debug(f"Using pinned revision for {repo_id}: {revision}")
        else:
            environment = os.getenv("ENVIRONMENT", "development").lower()
            if environment == "production":
                logger.warning(
                    f"No pinned revision configured for {repo_id} in production environment. "
                    f"Add to config/model_revisions.yaml"
                )
            else:
                logger.debug(f"No pinned revision configured for {repo_id}")

        return revision

    @classmethod
    def _load_revision_config(cls) -> None:
        """Load model revision configuration from YAML file."""
        if cls._config_path is None:
            # Try multiple possible locations
            possible_paths = [
                Path("config/model_revisions.yaml"),
                Path("../config/model_revisions.yaml"),
                Path(__file__).parent.parent.parent / "config" / "model_revisions.yaml",
            ]

            for path in possible_paths:
                if path.exists():
                    cls._config_path = path
                    break

        if cls._config_path and cls._config_path.exists():
            try:
                with open(cls._config_path, "r") as f:
                    config = yaml.safe_load(f)
                    cls._revision_config = config.get("models", {})
                    logger.info(
                        f"Loaded {len(cls._revision_config)} pinned model revisions from {cls._config_path}"
                    )
            except Exception as e:
                logger.warning(f"Failed to load model revision config from {cls._config_path}: {e}")
                cls._revision_config = {}
        else:
            logger.warning("Model revision config file not found. Using empty configuration.")
            cls._revision_config = {}

    @classmethod
    def download_model(cls, repo_id: str, revision: Optional[str] = None, **kwargs) -> Any:
        """
        Download a model with security validation.

        This method is a placeholder for future implementation. Currently,
        the pattern is to use get_pinned_revision() and pass the revision
        to the appropriate download function (from_pretrained, hf_hub_download, etc.)

        Args:
            repo_id: HuggingFace repository ID
            revision: Optional revision hash/tag
            **kwargs: Additional arguments to pass to download function

        Returns:
            Downloaded model or path
        """
        if revision is None:
            revision = cls.get_pinned_revision(repo_id)

        environment = os.getenv("ENVIRONMENT", "development").lower()

        if revision is None and environment == "production":
            raise ValueError(
                f"Pinned revision required for model {repo_id} in production environment. "
                f"Add to config/model_revisions.yaml or provide revision parameter."
            )

        if revision is None:
            logger.warning(
                f"Downloading model {repo_id} without pinned revision in {environment} environment. "
                f"Consider adding to config/model_revisions.yaml for reproducibility and security."
            )

        # This is a placeholder - actual download is done by calling code
        # using the revision returned by get_pinned_revision()
        return None
