"""Environment-specific security policy configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from src.platform.security.models import SecurityEnvironment


@dataclass(frozen=True)
class SecurityPolicyConfig:
    """Resolved security policy values for one environment."""

    enforce_strict_binding: bool
    require_pinned_models: bool
    allow_hardcoded_temp_paths: bool
    require_pickle_validation: bool
    require_url_scheme_validation: bool
    audit_all_operations: bool


class SecurityConfigManager:
    """Load and query security policies by environment."""

    REQUIRED_FIELDS = tuple(SecurityPolicyConfig.__dataclass_fields__)

    def __init__(
        self,
        config_path: str | os.PathLike[str] | None = None,
        environment: str | SecurityEnvironment | None = None,
    ) -> None:
        self.config_path = Path(config_path) if config_path else None
        self.current_environment = self._resolve_environment(environment)
        self.config = self._load_config()

    @classmethod
    def for_production(cls) -> "SecurityConfigManager":
        return cls(environment=SecurityEnvironment.PRODUCTION)

    @classmethod
    def for_development(cls) -> "SecurityConfigManager":
        return cls(environment=SecurityEnvironment.DEVELOPMENT)

    @classmethod
    def for_research(cls) -> "SecurityConfigManager":
        return cls(environment=SecurityEnvironment.RESEARCH)

    @classmethod
    def from_environment(
        cls, environment: str | SecurityEnvironment | None = None
    ) -> "SecurityConfigManager":
        return cls(environment=environment)

    def reload_config(self) -> None:
        """Reload configuration from disk or defaults."""
        self.config = self._load_config()

    def get_config_for_environment(self, environment: SecurityEnvironment) -> SecurityPolicyConfig:
        return self.config[environment]

    def should_enforce_strict_binding(self) -> bool:
        return self._current_config().enforce_strict_binding

    def should_require_pinned_models(self) -> bool:
        return self._current_config().require_pinned_models

    def should_allow_hardcoded_temp_paths(self) -> bool:
        return self._current_config().allow_hardcoded_temp_paths

    def should_require_pickle_validation(self) -> bool:
        return self._current_config().require_pickle_validation

    def should_require_url_scheme_validation(self) -> bool:
        return self._current_config().require_url_scheme_validation

    def should_audit_all_operations(self) -> bool:
        return self._current_config().audit_all_operations

    def _current_config(self) -> SecurityPolicyConfig:
        return self.get_config_for_environment(self.current_environment)

    def _load_config(self) -> dict[SecurityEnvironment, SecurityPolicyConfig]:
        if self.config_path and self.config_path.exists():
            raw_config = yaml.safe_load(self.config_path.read_text(encoding="utf-8")) or {}
        else:
            raw_config = self._default_config()

        self._validate_raw_config(raw_config)
        return {
            SecurityEnvironment(env_name): SecurityPolicyConfig(**values)
            for env_name, values in raw_config.items()
        }

    def _validate_raw_config(self, raw_config: dict[str, Any]) -> None:
        required_environments = {
            SecurityEnvironment.PRODUCTION.value,
            SecurityEnvironment.DEVELOPMENT.value,
            SecurityEnvironment.RESEARCH.value,
        }
        missing_environments = required_environments - set(raw_config)
        if missing_environments:
            raise ValueError(f"Missing configuration for: {sorted(missing_environments)}")

        for environment, values in raw_config.items():
            missing_fields = set(self.REQUIRED_FIELDS) - set(values or {})
            if missing_fields:
                raise ValueError(
                    f"Missing required fields for {environment}: {sorted(missing_fields)}"
                )

    @staticmethod
    def _resolve_environment(
        environment: str | SecurityEnvironment | None,
    ) -> SecurityEnvironment:
        if isinstance(environment, SecurityEnvironment):
            return environment

        raw_environment = environment or os.getenv("ENVIRONMENT", "production")
        try:
            return SecurityEnvironment(str(raw_environment).lower())
        except ValueError:
            return SecurityEnvironment.DEVELOPMENT

    @staticmethod
    def _default_config() -> dict[str, dict[str, bool]]:
        return {
            "production": {
                "enforce_strict_binding": True,
                "require_pinned_models": True,
                "allow_hardcoded_temp_paths": False,
                "require_pickle_validation": True,
                "require_url_scheme_validation": True,
                "audit_all_operations": True,
            },
            "development": {
                "enforce_strict_binding": False,
                "require_pinned_models": False,
                "allow_hardcoded_temp_paths": True,
                "require_pickle_validation": False,
                "require_url_scheme_validation": False,
                "audit_all_operations": False,
            },
            "research": {
                "enforce_strict_binding": False,
                "require_pinned_models": False,
                "allow_hardcoded_temp_paths": True,
                "require_pickle_validation": False,
                "require_url_scheme_validation": False,
                "audit_all_operations": True,
            },
        }
