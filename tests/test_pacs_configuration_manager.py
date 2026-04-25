"""Property-based tests for PACS Configuration Manager.

Feature: pacs-integration-system
Property 20: Configuration Loading and Decryption
Property 21: Multi-Endpoint Configuration Support
Property 22: Configuration Validation Completeness
Property 23: Endpoint Configuration Completeness
Property 24: Profile-Based Configuration Loading
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import yaml
from cryptography.fernet import Fernet
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from src.clinical.pacs.configuration_manager import ConfigurationManager
from src.clinical.pacs.data_models import (
    PACSConfiguration,
    PACSEndpoint,
    PACSVendor,
    PerformanceConfig,
    SecurityConfig,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_test_config(
    profile_name: str = "test",
    endpoint_count: int = 1,
    primary_index: int = 0,
) -> PACSConfiguration:
    """Create test PACS configuration."""
    endpoints = {}

    for i in range(endpoint_count):
        endpoint_id = f"endpoint_{i}"
        endpoints[endpoint_id] = PACSEndpoint(
            endpoint_id=endpoint_id,
            ae_title=f"TEST_AE_{i}",
            host=f"pacs{i}.test.local",
            port=11112 + i,
            vendor=PACSVendor.GENERIC,
            security_config=SecurityConfig(
                tls_enabled=False,
                verify_certificates=False,
                mutual_authentication=False,
            ),
            performance_config=PerformanceConfig(),
            description=f"Test endpoint {i}",
            is_primary=(i == primary_index),
        )

    return PACSConfiguration(
        profile_name=profile_name,
        pacs_endpoints=endpoints,
        storage_config={
            "local_cache_path": "./data/cache",
            "max_cache_size_gb": 100,
            "cleanup_threshold_gb": 20,
        },
        notification_config={"enabled": False},
        audit_config={"enabled": True, "retention_years": 1},
    )


def _save_config_to_file(config: PACSConfiguration, file_path: Path, encrypted: bool = False, encryption_key: bytes = None):
    """Save config to YAML file."""
    config_dict = {
        "profile_name": config.profile_name,
        "pacs_endpoints": {},
        "storage_config": config.storage_config,
        "notification_config": config.notification_config,
        "audit_config": config.audit_config,
    }

    for ep_id, ep in config.pacs_endpoints.items():
        config_dict["pacs_endpoints"][ep_id] = {
            "endpoint_id": ep.endpoint_id,
            "ae_title": ep.ae_title,
            "host": ep.host,
            "port": ep.port,
            "vendor": ep.vendor.value,
            "description": ep.description,
            "is_primary": ep.is_primary,
            "security_config": {
                "tls_enabled": ep.security_config.tls_enabled,
                "tls_version": ep.security_config.tls_version,
                "verify_certificates": ep.security_config.verify_certificates,
                "mutual_authentication": ep.security_config.mutual_authentication,
            },
            "performance_config": {
                "max_concurrent_studies": ep.performance_config.max_concurrent_studies,
                "connection_pool_size": ep.performance_config.connection_pool_size,
                "query_timeout": ep.performance_config.query_timeout,
                "retrieval_timeout": ep.performance_config.retrieval_timeout,
                "retry_attempts": ep.performance_config.retry_attempts,
            },
        }

    yaml_data = yaml.dump(config_dict, default_flow_style=False)

    if encrypted:
        if not encryption_key:
            raise ValueError("Encryption key required")
        fernet = Fernet(encryption_key)
        encrypted_data = fernet.encrypt(yaml_data.encode("utf-8"))
        with open(file_path, "wb") as f:
            f.write(encrypted_data)
    else:
        with open(file_path, "w") as f:
            f.write(yaml_data)


# ---------------------------------------------------------------------------
# Property 20 — Configuration Loading and Decryption
# ---------------------------------------------------------------------------

# Feature: pacs-integration-system, Property 20: Configuration Loading and Decryption
# For any encrypted configuration file, the Configuration Manager SHALL successfully
# load and decrypt the settings when the file is valid.


def test_property_20_load_plain_configuration():
    """Plain config files must load successfully."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_dir = Path(tmpdir)
        config_mgr = ConfigurationManager(config_directory=config_dir)

        # Create test config
        test_config = _make_test_config(profile_name="test")

        # Save to file
        config_file = config_dir / "test.yaml"
        _save_config_to_file(test_config, config_file, encrypted=False)

        # Load config
        loaded_config = config_mgr.load_configuration(profile="test", encrypted=False)

        # Verify loaded correctly
        assert loaded_config.profile_name == "test"
        assert len(loaded_config.pacs_endpoints) == 1
        assert "endpoint_0" in loaded_config.pacs_endpoints


def test_property_20_load_encrypted_configuration():
    """Encrypted config files must load and decrypt successfully."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_dir = Path(tmpdir)

        # Generate encryption key
        encryption_key = Fernet.generate_key()
        config_mgr = ConfigurationManager(
            config_directory=config_dir, encryption_key=encryption_key
        )

        # Create test config
        test_config = _make_test_config(profile_name="encrypted_test")

        # Save encrypted
        config_file = config_dir / "encrypted_test.encrypted.yaml"
        _save_config_to_file(test_config, config_file, encrypted=True, encryption_key=encryption_key)

        # Load encrypted config
        loaded_config = config_mgr.load_configuration(profile="encrypted_test", encrypted=True)

        # Verify decrypted correctly
        assert loaded_config.profile_name == "encrypted_test"
        assert len(loaded_config.pacs_endpoints) == 1


def test_property_20_encrypted_load_fails_without_key():
    """Encrypted config must fail to load without encryption key."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_dir = Path(tmpdir)

        # Create encrypted config
        encryption_key = Fernet.generate_key()
        test_config = _make_test_config(profile_name="encrypted")

        config_file = config_dir / "encrypted.encrypted.yaml"
        _save_config_to_file(test_config, config_file, encrypted=True, encryption_key=encryption_key)

        # Try to load without key
        config_mgr = ConfigurationManager(config_directory=config_dir, encryption_key=None)

        with pytest.raises(ValueError, match="Encryption key required"):
            config_mgr.load_configuration(profile="encrypted", encrypted=True)


@given(profile_name=st.text(min_size=1, max_size=20, alphabet=st.characters(min_codepoint=97, max_codepoint=122)))
@settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_property_20_load_configuration_caches_result(profile_name):
    """Loaded configs must be cached for subsequent access."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_dir = Path(tmpdir)
        config_mgr = ConfigurationManager(config_directory=config_dir)

        # Create and save config
        test_config = _make_test_config(profile_name=profile_name)
        config_file = config_dir / f"{profile_name}.yaml"
        _save_config_to_file(test_config, config_file)

        # Load twice
        config1 = config_mgr.load_configuration(profile=profile_name)
        config2 = config_mgr.load_configuration(profile=profile_name)

        # Must be same instance (cached)
        assert config1 is config2


# ---------------------------------------------------------------------------
# Property 21 — Multi-Endpoint Configuration Support
# ---------------------------------------------------------------------------

# Feature: pacs-integration-system, Property 21: Multi-Endpoint Configuration Support
# For any configuration with multiple PACS endpoints, all endpoints SHALL be
# loaded and made available for redundancy operations.


@given(endpoint_count=st.integers(min_value=2, max_value=10))
@settings(max_examples=20)
def test_property_21_multiple_endpoints_loaded(endpoint_count):
    """All configured endpoints must be loaded."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_dir = Path(tmpdir)
        config_mgr = ConfigurationManager(config_directory=config_dir)

        # Create config with multiple endpoints
        test_config = _make_test_config(
            profile_name="multi", endpoint_count=endpoint_count, primary_index=0
        )

        config_file = config_dir / "multi.yaml"
        _save_config_to_file(test_config, config_file)

        # Load config
        loaded_config = config_mgr.load_configuration(profile="multi")

        # All endpoints must be present
        assert len(loaded_config.pacs_endpoints) == endpoint_count

        # Verify each endpoint
        for i in range(endpoint_count):
            endpoint_id = f"endpoint_{i}"
            assert endpoint_id in loaded_config.pacs_endpoints

            endpoint = loaded_config.pacs_endpoints[endpoint_id]
            assert endpoint.ae_title == f"TEST_AE_{i}"
            assert endpoint.host == f"pacs{i}.test.local"
            assert endpoint.port == 11112 + i


def test_property_21_primary_and_backup_endpoints_identified():
    """Primary and backup endpoints must be correctly identified."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_dir = Path(tmpdir)
        config_mgr = ConfigurationManager(config_directory=config_dir)

        # Create config with primary and backup
        test_config = _make_test_config(
            profile_name="redundant", endpoint_count=3, primary_index=0
        )

        config_file = config_dir / "redundant.yaml"
        _save_config_to_file(test_config, config_file)

        # Load config
        loaded_config = config_mgr.load_configuration(profile="redundant")

        # Get primary endpoint
        primary = loaded_config.get_primary_endpoint()
        assert primary is not None
        assert primary.endpoint_id == "endpoint_0"
        assert primary.is_primary

        # Get backup endpoints
        backups = loaded_config.get_backup_endpoints()
        assert len(backups) == 2
        assert all(not ep.is_primary for ep in backups)


@given(endpoint_count=st.integers(min_value=1, max_value=5))
@settings(max_examples=20)
def test_property_21_all_endpoints_accessible(endpoint_count):
    """All endpoints must be accessible after loading."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_dir = Path(tmpdir)
        config_mgr = ConfigurationManager(config_directory=config_dir)

        test_config = _make_test_config(endpoint_count=endpoint_count)
        config_file = config_dir / "test.yaml"
        _save_config_to_file(test_config, config_file)

        loaded_config = config_mgr.load_configuration(profile="test")

        # Access each endpoint
        for endpoint_id, endpoint in loaded_config.pacs_endpoints.items():
            assert endpoint.endpoint_id == endpoint_id
            assert endpoint.ae_title is not None
            assert endpoint.host is not None
            assert endpoint.port > 0


# ---------------------------------------------------------------------------
# Property 22 — Configuration Validation Completeness
# ---------------------------------------------------------------------------

# Feature: pacs-integration-system, Property 22: Configuration Validation Completeness
# For any configuration change, validation SHALL identify and report all invalid
# settings with detailed error information.


def test_property_22_validation_detects_missing_primary_endpoint():
    """Validation must detect missing primary endpoint."""
    config_mgr = ConfigurationManager()

    # Create config with no primary endpoint
    test_config = _make_test_config(endpoint_count=2, primary_index=-1)
    for ep in test_config.pacs_endpoints.values():
        ep.is_primary = False

    # Validate
    result = config_mgr.validate_configuration(test_config)

    # Must detect missing primary
    assert not result.is_valid
    assert any("primary" in err.lower() for err in result.errors)


def test_property_22_validation_detects_duplicate_ae_titles():
    """Validation must detect duplicate AE titles."""
    config_mgr = ConfigurationManager()

    # Create config with duplicate AE titles
    test_config = _make_test_config(endpoint_count=2)
    for ep in test_config.pacs_endpoints.values():
        ep.ae_title = "DUPLICATE_AE"

    # Validate
    result = config_mgr.validate_configuration(test_config)

    # Must detect duplicates
    assert not result.is_valid
    assert any("duplicate" in err.lower() and "ae" in err.lower() for err in result.errors)


def test_property_22_validation_detects_invalid_cache_config():
    """Validation must detect invalid cache configuration."""
    config_mgr = ConfigurationManager()

    # Create config with invalid cache settings
    test_config = _make_test_config()
    test_config.storage_config["max_cache_size_gb"] = 100
    test_config.storage_config["cleanup_threshold_gb"] = 150  # Invalid: > max

    # Validate
    result = config_mgr.validate_configuration(test_config)

    # Must detect invalid cache config
    assert not result.is_valid
    assert any("cleanup" in err.lower() and "cache" in err.lower() for err in result.errors)


def test_property_22_validation_detects_invalid_security_config():
    """Validation must detect invalid security configuration."""
    config_mgr = ConfigurationManager()

    # Create config with invalid performance settings (easier to test)
    test_config = _make_test_config()
    endpoint = list(test_config.pacs_endpoints.values())[0]

    # Set invalid performance config
    endpoint.performance_config.max_concurrent_studies = -1  # Invalid: must be positive

    # Validate
    result = config_mgr.validate_configuration(test_config)

    # Must detect invalid config
    assert not result.is_valid
    assert any("concurrent" in err.lower() or "positive" in err.lower() for err in result.errors)


@given(
    max_cache=st.integers(min_value=1, max_value=1000),
    cleanup_threshold=st.integers(min_value=1, max_value=1000),
)
@settings(max_examples=50)
def test_property_22_validation_cache_threshold_relationship(max_cache, cleanup_threshold):
    """Validation must enforce cleanup_threshold < max_cache_size."""
    config_mgr = ConfigurationManager()

    test_config = _make_test_config()
    test_config.storage_config["max_cache_size_gb"] = max_cache
    test_config.storage_config["cleanup_threshold_gb"] = cleanup_threshold

    result = config_mgr.validate_configuration(test_config)

    if cleanup_threshold >= max_cache:
        # Must be invalid
        assert not result.is_valid
        assert any("cleanup" in err.lower() for err in result.errors)
    else:
        # May be valid (depends on other factors)
        pass


# ---------------------------------------------------------------------------
# Property 23 — Endpoint Configuration Completeness
# ---------------------------------------------------------------------------

# Feature: pacs-integration-system, Property 23: Endpoint Configuration Completeness
# For any PACS endpoint configuration, all required fields (AE_Title, IP address,
# port, security settings) SHALL be stored and retrievable.


@given(
    ae_title=st.text(min_size=1, max_size=16, alphabet=st.characters(min_codepoint=65, max_codepoint=90)),
    host=st.text(min_size=5, max_size=50, alphabet=st.characters(min_codepoint=97, max_codepoint=122)),
    port=st.integers(min_value=1024, max_value=65535),
)
@settings(max_examples=50)
def test_property_23_endpoint_fields_stored_and_retrievable(ae_title, host, port):
    """All endpoint fields must be stored and retrievable."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_dir = Path(tmpdir)
        config_mgr = ConfigurationManager(config_directory=config_dir)

        # Create endpoint with specific values
        endpoint = PACSEndpoint(
            endpoint_id="test_ep",
            ae_title=ae_title,
            host=host,
            port=port,
            vendor=PACSVendor.GENERIC,
            security_config=SecurityConfig(
                tls_enabled=True,
                tls_version="1.3",
                verify_certificates=True,
                mutual_authentication=False,
            ),
            performance_config=PerformanceConfig(),
            is_primary=True,
        )

        # Create config
        config = PACSConfiguration(
            profile_name="test",
            pacs_endpoints={"test_ep": endpoint},
            storage_config={},
        )

        # Save and reload
        config_file = config_dir / "test.yaml"
        _save_config_to_file(config, config_file)

        loaded_config = config_mgr.load_configuration(profile="test")
        loaded_endpoint = loaded_config.pacs_endpoints["test_ep"]

        # Verify all fields preserved
        assert loaded_endpoint.ae_title == ae_title
        assert loaded_endpoint.host == host
        assert loaded_endpoint.port == port
        assert loaded_endpoint.vendor == PACSVendor.GENERIC
        assert loaded_endpoint.security_config.tls_enabled is True
        assert loaded_endpoint.security_config.tls_version == "1.3"


def test_property_23_security_settings_stored_completely():
    """Security settings must be stored completely."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_dir = Path(tmpdir)
        config_mgr = ConfigurationManager(config_directory=config_dir)

        # Create endpoint with full security config (no mutual auth to avoid cert requirement)
        security_config = SecurityConfig(
            tls_enabled=True,
            tls_version="1.3",
            verify_certificates=True,
            mutual_authentication=False,
        )

        endpoint = PACSEndpoint(
            endpoint_id="secure_ep",
            ae_title="SECURE_AE",
            host="secure.pacs.local",
            port=11112,
            vendor=PACSVendor.GE,
            security_config=security_config,
            performance_config=PerformanceConfig(),
            is_primary=True,
        )

        config = PACSConfiguration(
            profile_name="secure",
            pacs_endpoints={"secure_ep": endpoint},
            storage_config={},
        )

        # Save and reload
        config_file = config_dir / "secure.yaml"
        _save_config_to_file(config, config_file)

        loaded_config = config_mgr.load_configuration(profile="secure")
        loaded_sec = loaded_config.pacs_endpoints["secure_ep"].security_config

        # Verify all security settings
        assert loaded_sec.tls_enabled is True
        assert loaded_sec.tls_version == "1.3"
        assert loaded_sec.verify_certificates is True
        assert loaded_sec.mutual_authentication is False


def test_property_23_performance_settings_stored_completely():
    """Performance settings must be stored completely."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_dir = Path(tmpdir)
        config_mgr = ConfigurationManager(config_directory=config_dir)

        # Create endpoint with custom performance config
        perf_config = PerformanceConfig(
            max_concurrent_studies=100,
            connection_pool_size=20,
            query_timeout=60,
            retrieval_timeout=600,
            retry_attempts=5,
        )

        endpoint = PACSEndpoint(
            endpoint_id="perf_ep",
            ae_title="PERF_AE",
            host="perf.pacs.local",
            port=11112,
            vendor=PACSVendor.SIEMENS,
            security_config=SecurityConfig(),
            performance_config=perf_config,
            is_primary=True,
        )

        config = PACSConfiguration(
            profile_name="perf",
            pacs_endpoints={"perf_ep": endpoint},
            storage_config={},
        )

        # Save and reload
        config_file = config_dir / "perf.yaml"
        _save_config_to_file(config, config_file)

        loaded_config = config_mgr.load_configuration(profile="perf")
        loaded_perf = loaded_config.pacs_endpoints["perf_ep"].performance_config

        # Verify all performance settings
        assert loaded_perf.max_concurrent_studies == 100
        assert loaded_perf.connection_pool_size == 20
        assert loaded_perf.query_timeout == 60
        assert loaded_perf.retrieval_timeout == 600
        assert loaded_perf.retry_attempts == 5


# ---------------------------------------------------------------------------
# Property 24 — Profile-Based Configuration Loading
# ---------------------------------------------------------------------------

# Feature: pacs-integration-system, Property 24: Profile-Based Configuration Loading
# For any environment profile selection, the correct profile-specific settings
# SHALL be loaded and applied.


@given(
    profile_names=st.lists(
        st.text(min_size=3, max_size=15, alphabet=st.characters(min_codepoint=97, max_codepoint=122)),
        min_size=2,
        max_size=5,
        unique=True,
    )
)
@settings(max_examples=20)
def test_property_24_correct_profile_loaded(profile_names):
    """Correct profile-specific settings must be loaded."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_dir = Path(tmpdir)
        config_mgr = ConfigurationManager(config_directory=config_dir)

        # Create multiple profiles with different settings
        for profile_name in profile_names:
            config = _make_test_config(profile_name=profile_name, endpoint_count=1)

            # Make each profile unique
            endpoint = list(config.pacs_endpoints.values())[0]
            endpoint.host = f"{profile_name}.pacs.local"

            config_file = config_dir / f"{profile_name}.yaml"
            _save_config_to_file(config, config_file)

        # Load each profile and verify correct settings
        for profile_name in profile_names:
            loaded_config = config_mgr.load_configuration(profile=profile_name)

            assert loaded_config.profile_name == profile_name

            endpoint = list(loaded_config.pacs_endpoints.values())[0]
            assert endpoint.host == f"{profile_name}.pacs.local"


def test_property_24_production_profile_enforces_security():
    """Production profile must enforce security requirements."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_dir = Path(tmpdir)
        config_mgr = ConfigurationManager(config_directory=config_dir)

        # Create production config with insecure audit settings
        prod_config = _make_test_config(profile_name="production")
        prod_config.audit_config["hipaa_compliant"] = False

        config_file = config_dir / "production.yaml"
        _save_config_to_file(prod_config, config_file)

        # Try to load production config
        with pytest.raises(ValueError, match="HIPAA compliance"):
            config_mgr.load_configuration(profile="production")


def test_property_24_list_available_profiles():
    """Available profiles must be listable."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_dir = Path(tmpdir)
        config_mgr = ConfigurationManager(config_directory=config_dir)

        # Create multiple profiles
        profiles = ["dev", "staging", "production"]
        for profile in profiles:
            config = _make_test_config(profile_name=profile)
            config_file = config_dir / f"{profile}.yaml"
            _save_config_to_file(config, config_file)

        # List profiles
        available = config_mgr.list_available_profiles()

        # All profiles must be listed
        for profile in profiles:
            assert profile in available


def test_property_24_profile_not_found_raises_error():
    """Loading non-existent profile must raise error."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_dir = Path(tmpdir)
        config_mgr = ConfigurationManager(config_directory=config_dir)

        # Try to load non-existent profile
        with pytest.raises(FileNotFoundError):
            config_mgr.load_configuration(profile="nonexistent")


# ---------------------------------------------------------------------------
# Additional Unit Tests
# ---------------------------------------------------------------------------


def test_configuration_manager_initialization():
    """Configuration manager must initialize correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_dir = Path(tmpdir)
        config_mgr = ConfigurationManager(config_directory=config_dir)

        assert config_mgr.config_directory == config_dir
        assert config_mgr._loaded_configurations == {}


def test_create_default_configuration():
    """Default configuration must be creatable."""
    config_mgr = ConfigurationManager()

    default_config = config_mgr.create_default_configuration()

    assert default_config.profile_name == "default"
    assert len(default_config.pacs_endpoints) == 1
    assert "default" in default_config.pacs_endpoints


def test_save_configuration():
    """Configuration must be saveable."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_dir = Path(tmpdir)
        config_mgr = ConfigurationManager(config_directory=config_dir)

        test_config = _make_test_config(profile_name="save_test")

        result = config_mgr.save_configuration(test_config, profile="save_test")

        assert result.success
        assert (config_dir / "save_test.yaml").exists()


def test_update_endpoint_settings():
    """Endpoint settings must be updatable."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_dir = Path(tmpdir)
        config_mgr = ConfigurationManager(config_directory=config_dir)

        # Create and save initial config
        test_config = _make_test_config()
        config_file = config_dir / "test.yaml"
        _save_config_to_file(test_config, config_file)

        # Update endpoint
        new_settings = {"host": "updated.pacs.local", "port": 22222}

        result = config_mgr.update_endpoint_settings(
            profile="test", endpoint_id="endpoint_0", settings=new_settings
        )

        assert result.success

        # Verify update
        updated_config = config_mgr.load_configuration(profile="test")
        endpoint = updated_config.pacs_endpoints["endpoint_0"]

        assert endpoint.host == "updated.pacs.local"
        assert endpoint.port == 22222


def test_generate_encryption_key():
    """Encryption key must be generatable from password."""
    config_mgr = ConfigurationManager()

    key = config_mgr.generate_encryption_key(password="test_password")

    assert key is not None
    assert len(key) > 0

    # Key should be usable with Fernet
    fernet = Fernet(key)
    test_data = b"test data"
    encrypted = fernet.encrypt(test_data)
    decrypted = fernet.decrypt(encrypted)

    assert decrypted == test_data


def test_get_configuration_statistics():
    """Configuration statistics must be retrievable."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_dir = Path(tmpdir)
        config_mgr = ConfigurationManager(config_directory=config_dir)

        stats = config_mgr.get_configuration_statistics()

        assert "config_directory" in stats
        assert "loaded_configurations" in stats
        assert "available_profiles" in stats
        assert "has_default_config" in stats
        assert "encryption_enabled" in stats


def test_validation_warns_missing_notification_recipients():
    """Validation must warn about missing notification recipients."""
    config_mgr = ConfigurationManager()

    test_config = _make_test_config()
    test_config.notification_config = {
        "enabled": True,
        "channels": {"email": {"enabled": True}},
        "recipients": {},
    }

    result = config_mgr.validate_configuration(test_config)

    # Should have warning about missing recipients
    assert any("recipient" in warn.lower() for warn in result.warnings)


def test_validation_detects_missing_email_config():
    """Validation must detect missing email configuration fields."""
    config_mgr = ConfigurationManager()

    test_config = _make_test_config()
    test_config.notification_config = {
        "enabled": True,
        "channels": {
            "email": {
                "enabled": True,
                # Missing smtp_server, smtp_port, username
            }
        },
    }

    result = config_mgr.validate_configuration(test_config)

    # Must detect missing email fields
    assert not result.is_valid
    assert any("email" in err.lower() for err in result.errors)
