"""Detect the active application environment from process variables."""

from __future__ import annotations

import logging
import os
from typing import Optional

from .models import SecurityEnvironment

logger = logging.getLogger(__name__)


class SecurityEnvironmentDetector:
    """Resolve and cache the active application environment."""

    def __init__(self) -> None:
        self._detected: Optional[SecurityEnvironment] = None

    def detect(self) -> SecurityEnvironment:
        """Return the configured environment, caching the first result."""
        if self._detected is not None:
            return self._detected

        if "ENVIRONMENT" in os.environ:
            raw_value = os.environ.get("ENVIRONMENT", "")
        else:
            raw_value = os.environ.get("DEPLOYMENT_ENV", "")

        normalized = raw_value.strip().lower()

        if not normalized:
            logger.warning(
                "No environment specified; defaulting to DEVELOPMENT"
            )
            self._detected = SecurityEnvironment.DEVELOPMENT
            return self._detected

        try:
            self._detected = SecurityEnvironment(normalized)
        except ValueError as exc:
            valid_values = ", ".join(
                environment.value for environment in SecurityEnvironment
            )
            raise ValueError(
                f"Invalid environment {raw_value!r}; "
                f"expected one of: {valid_values}"
            ) from exc

        return self._detected
