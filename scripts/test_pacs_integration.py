#!/usr/bin/env python3
"""
Test script for PACS Integration System.

This script performs basic validation of the PACS integration components
to ensure they are working correctly.
"""

import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.clinical.pacs import (
    PACSAdapter, ConfigurationManager, SecurityManager,
    PACSEndpoint, PACSConfiguration, SecurityConfig, PerformanceConfig,
    PACSVendor, StudyInfo, AnalysisResults, DetectedRegion, DiagnosticRecommendation
)
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Run PACS integration tests."""
    logger.info("Starting PACS Integration System Tests")
    
    tests_passed = 0
    tests_failed = 0
    
    # Test 1: Configuration Management
    logger.info("=== Test 1: Configuration Management ===")
    try:
        test_configuration_management()
        logger.info("✓ Configuration management test passed")
        tests_passed += 1
    except Exception as e:
        logger.error(f"✗ Configuration management test failed: {e}")
        tests_failed += 1
    
    # Test 2: Security Manager
    logger.info("=== Test 2: Security Manager ===")
    try:
        test_security_manager()
        logger.info("✓ Security manager test passed")
        tests_passed += 1
    except Exception as e:
        logger.error(f"✗ Security manager test failed: {e}")
        tests_failed += 1
    
    # Test 3: Data Models
    logger.info("=== Test 3: Data Models ===")
    try:
        test_data_models()
        logger.info("✓ Data models test passed")
        tests_passed += 1
    except Exception as e:
        logger.error(f"✗ Data models test failed: {e}")
        tests_failed += 1
    
    # Test 4: PACS Adapter Initialization
    logger.info("=== Test 4: PACS Adapter Initialization ===")
    try:
        test_pacs_adapter_init()
        logger.info("✓ PACS adapter initialization test passed")
        tests_passed += 1
    except Exception as e:
        logger.error(f"✗ PACS adapter initialization test failed: {e}")
        tests_failed += 1
    
    # Test 5: Analysis Results Creation
    logger.info("=== Test 5: Analysis Results Creation ===")
    try:
        test_analysis_results()
        logger.info("✓ Analysis results test passed")
        tests_passed += 1
    except Exception as e:
        logger.error(f"✗ Analysis results test failed: {e}")
        tests_failed += 1
    
    # Summary
    total_tests = tests_passed + tests_failed
    logger.info(f"=== Test Summary ===")
    logger.info(f"Total tests: {total_tests}")
    logger.info(f"Passed: {tests_passed}")
    logger.info(f"Failed: {tests_failed}")
    
    if tests_failed == 0:
        logger.info("🎉 All tests passed!")
        return 0
    else:
        logger.error(f"❌ {tests_failed} tests failed")
        return 1


def test_configuration_management():
    """Test configuration management functionality."""
    config_manager = ConfigurationManager("configs/pacs")
    
    # Test default configuration creation
    default_config = config_manager.create_default_configuration()
    assert default_config.profile_name == "default"
    assert len(default_config.pacs_endpoints) > 0
    
    # Test configuration validation
    validation = config_manager.validate_configuration(default_config)
    assert validation.is_valid, f"Validation errors: {validation.errors}"
    
    # Test configuration serialization
    config_dict = config_manager._configuration_to_dict(default_config)
    assert "profile_name" in config_dict
    assert "pacs_endpoints" in config_dict
    
    logger.info("  ✓ Default configuration created and validated")


def test_security_manager():
    """Test security manager functionality."""
    security_manager = SecurityManager()
    
    # Test certificate generation
    cert_path = Path("./test_cert.pem")
    key_path = Path("./test_key.pem")
    
    try:
        result = security_manager.generate_self_signed_certificate(
            common_name="test.example.com",
            output_cert_path=cert_path,
            output_key_path=key_path,
            validity_days=30
        )
        
        assert result.success, f"Certificate generation failed: {result.message}"
        assert cert_path.exists(), "Certificate file not created"
        assert key_path.exists(), "Key file not created"
        
        logger.info("  ✓ Self-signed certificate generated successfully")
        
    finally:
        # Cleanup
        if cert_path.exists():
            cert_path.unlink()
        if key_path.exists():
            key_path.unlink()


def test_data_models():
    """Test data model functionality."""
    # Test SecurityConfig
    security_config = SecurityConfig(
        tls_enabled=True,
        verify_certificates=True
    )
    
    errors = security_config.validate()
    assert isinstance(errors, list)
    
    # Test PerformanceConfig
    perf_config = PerformanceConfig(
        max_concurrent_studies=10,
        connection_pool_size=5
    )
    
    errors = perf_config.validate()
    assert isinstance(errors, list)
    
    # Test PACSEndpoint
    endpoint = PACSEndpoint(
        endpoint_id="test_endpoint",
        ae_title="TEST_AE",
        host="localhost",
        port=11112,
        vendor=PACSVendor.GENERIC,
        security_config=security_config,
        performance_config=perf_config
    )
    
    assert endpoint.endpoint_id == "test_endpoint"
    assert endpoint.vendor == PACSVendor.GENERIC
    
    # Test association parameters
    assoc_params = endpoint.create_association_parameters()
    assert "ae_title" in assoc_params
    assert "address" in assoc_params
    
    logger.info("  ✓ Data models created and validated")


def test_pacs_adapter_init():
    """Test PACS adapter initialization."""
    # This should work with default configuration
    adapter = PACSAdapter(config_profile="development")
    
    assert adapter.config_profile == "development"
    assert adapter.configuration is not None
    assert adapter.query_engine is not None
    assert adapter.retrieval_engine is not None
    assert adapter.storage_engine is not None
    assert adapter.security_manager is not None
    assert adapter.config_manager is not None
    
    # Test statistics
    stats = adapter.get_adapter_statistics()
    assert "config_profile" in stats
    assert "endpoints_configured" in stats
    
    # Test endpoint status (will fail connection but should not crash)
    status = adapter.get_endpoint_status()
    assert isinstance(status, dict)
    
    logger.info("  ✓ PACS adapter initialized successfully")


def test_analysis_results():
    """Test analysis results creation and validation."""
    # Create detected regions
    regions = [
        DetectedRegion(
            region_id="test_region",
            coordinates=(10, 20, 30, 40),
            confidence=0.85,
            region_type="test_type"
        )
    ]
    
    # Create recommendations
    recommendations = [
        DiagnosticRecommendation(
            recommendation_id="test_rec",
            recommendation_text="Test recommendation",
            confidence=0.90,
            urgency_level="MEDIUM"
        )
    ]
    
    # Create analysis results
    analysis = AnalysisResults(
        study_instance_uid="1.2.3.4.5",
        series_instance_uid="1.2.3.4.6",
        algorithm_name="Test Algorithm",
        algorithm_version="1.0.0",
        confidence_score=0.88,
        detected_regions=regions,
        diagnostic_recommendations=recommendations,
        processing_timestamp=datetime.now()
    )
    
    assert analysis.study_instance_uid == "1.2.3.4.5"
    assert len(analysis.detected_regions) == 1
    assert len(analysis.diagnostic_recommendations) == 1
    
    # Test validation
    assert analysis.validate_clinical_thresholds()
    
    logger.info("  ✓ Analysis results created and validated")


if __name__ == "__main__":
    sys.exit(main())