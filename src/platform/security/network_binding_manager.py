"""Network binding policy enforcement."""

from __future__ import annotations

import ipaddress
import logging
import os

from src.platform.security.exceptions import NetworkBindingSecurityError
from src.platform.security.models import SecurityEnvironment

logger = logging.getLogger(__name__)
_UNSET = object()


class NetworkBindingManager:
    """Resolve and validate API bind addresses for the active environment."""

    def __init__(self, environment: str | SecurityEnvironment | None = None) -> None:
        self.environment = self._resolve_environment(environment)

    def get_safe_host(self, requested_host: str | None | object = _UNSET) -> str:
        """Return an allowed bind host or raise when the request is unsafe."""
        host = "127.0.0.1" if requested_host is _UNSET else requested_host
        self._validate_host_format(host)

        if host == "0.0.0.0" and self.environment == SecurityEnvironment.PRODUCTION:  # nosec B104
            if os.getenv("ALLOW_PUBLIC_BINDING", "").lower() != "true":
                logger.warning("Network binding blocked: 0.0.0.0 in production")
                raise NetworkBindingSecurityError(
                    "0.0.0.0 binding not allowed in production without ALLOW_PUBLIC_BINDING=true"
                )

        if host == "0.0.0.0" and self.environment in {  # nosec B104
            SecurityEnvironment.DEVELOPMENT,
            SecurityEnvironment.RESEARCH,
        }:
            logger.warning("0.0.0.0 binding in %s environment", self.environment.value)

        logger.warning("Network binding allowed: %s in %s", host, self.environment.value)
        return host

    def validate_binding(self, host: str, port: int) -> None:
        """Validate a host and port pair."""
        if not isinstance(port, int) or port < 1 or port > 65535:
            raise ValueError("Port must be between 1 and 65535")
        self.get_safe_host(requested_host=host)

    def get_safe_host_and_port(
        self, requested_host: str | None | object = _UNSET, requested_port: int = 8000
    ) -> tuple[str, int]:
        """Return a validated host and port tuple."""
        host = self.get_safe_host(requested_host=requested_host)
        self.validate_binding(host, requested_port)
        return host, requested_port

    @staticmethod
    def _resolve_environment(
        environment: str | SecurityEnvironment | None,
    ) -> SecurityEnvironment:
        if isinstance(environment, SecurityEnvironment):
            return environment
        if hasattr(environment, "current_environment"):
            return environment.current_environment

        raw_environment = environment or os.getenv("ENVIRONMENT", "development")
        try:
            return SecurityEnvironment(str(raw_environment).lower())
        except ValueError:
            return SecurityEnvironment.DEVELOPMENT

    @staticmethod
    def _validate_host_format(host: str | None) -> None:
        if not host or not isinstance(host, str):
            raise ValueError("Host must be a non-empty string")

        if host == "localhost":
            return

        try:
            ipaddress.ip_address(host)
        except ValueError as exc:
            raise NetworkBindingSecurityError(f"Invalid host binding: {host}") from exc
