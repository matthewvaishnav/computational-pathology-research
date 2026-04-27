"""
Integration tests for secure aggregator with federated learning framework.

Tests that secure aggregator integrates correctly with the existing
federated learning infrastructure without requiring TenSEAL.
"""

import pytest
import torch

from src.federated.aggregator.factory import AggregatorFactory, create_aggregator
from src.federated.aggregator.secure import SecureAggregator
from src.federated.common.data_models import ClientUpdate


def test_secure_aggregator_in_factory():
    """Test that secure aggregator is registered in factory."""
    available = AggregatorFactory.get_available_algorithms()
    assert "secure" in available


def test_secure_aggregator_factory_creation():
    """Test creating secure aggregator through factory."""
    aggregator = AggregatorFactory.create_aggregator("secure")
    
    assert isinstance(aggregator, SecureAggregator)
    assert aggregator.algorithm_name == "SecureAggregation"


def test_secure_aggregator_factory_with_config():
    """Test creating secure aggregator with custom config."""
    config = {
        "poly_modulus_degree": 4096,
        "max_workers": 2,
        "dropout_threshold": 0.6,
    }
    
    aggregator = AggregatorFactory.create_aggregator("secure", config)
    
    assert isinstance(aggregator, SecureAggregator)
    assert aggregator.poly_modulus_degree == 4096
    assert aggregator.max_workers == 2
    assert aggregator.dropout_threshold == 0.6


def test_secure_aggregator_convenience_function():
    """Test creating secure aggregator with convenience function."""
    aggregator = create_aggregator(
        "secure",
        poly_modulus_degree=4096,
        dropout_threshold=0.7,
    )
    
    assert isinstance(aggregator, SecureAggregator)
    assert aggregator.poly_modulus_degree == 4096
    assert aggregator.dropout_threshold == 0.7


def test_secure_aggregator_algorithm_info():
    """Test getting algorithm information."""
    info = AggregatorFactory.get_algorithm_info("secure")
    
    assert info["name"] == "secure"
    assert info["class"] == "SecureAggregator"
    assert "poly_modulus_degree" in info["default_config"]
    assert "max_workers" in info["default_config"]
    assert "dropout_threshold" in info["default_config"]


def test_secure_aggregator_interface_compliance():
    """Test that secure aggregator implements BaseAggregator interface."""
    aggregator = SecureAggregator(poly_modulus_degree=4096)
    
    # Check required methods exist
    assert hasattr(aggregator, "aggregate")
    assert hasattr(aggregator, "algorithm_name")
    assert callable(aggregator.aggregate)
    
    # Check algorithm name
    assert aggregator.algorithm_name == "SecureAggregation"


def test_secure_aggregator_setup_round():
    """Test setup_round method."""
    aggregator = SecureAggregator(poly_modulus_degree=4096)
    
    expected_clients = ["hospital_a", "hospital_b", "hospital_c"]
    
    # This will fail without TenSEAL, but we can test the interface
    try:
        public_context = aggregator.setup_round(expected_clients)
        # If TenSEAL is available, should return bytes
        assert isinstance(public_context, bytes)
    except RuntimeError as e:
        # Expected if TenSEAL not available
        assert "TenSEAL not available" in str(e)


def test_secure_aggregator_dropout_handling():
    """Test dropout handling logic."""
    aggregator = SecureAggregator(
        poly_modulus_degree=4096,
        dropout_threshold=0.5,
    )
    
    # Setup round with 4 clients
    expected_clients = ["hospital_a", "hospital_b", "hospital_c", "hospital_d"]
    
    try:
        aggregator.setup_round(expected_clients)
        
        # Check minimum clients required
        assert aggregator.min_clients_required == 2  # 50% of 4
        
        # Handle dropout
        aggregator.handle_dropout(["hospital_d"])
        
        # Minimum should still be based on remaining clients
        assert aggregator.min_clients_required >= 2
        
    except RuntimeError:
        # TenSEAL not available - skip
        pytest.skip("TenSEAL not available")


def test_secure_aggregator_empty_updates_validation():
    """Test that empty updates are rejected."""
    aggregator = SecureAggregator(poly_modulus_degree=4096)
    
    with pytest.raises(ValueError, match="Cannot aggregate empty list"):
        aggregator.aggregate([])


def test_secure_aggregator_zero_weight_validation():
    """Test that zero total weight is rejected."""
    aggregator = SecureAggregator(poly_modulus_degree=4096)
    
    updates = [
        ClientUpdate(
            client_id="hospital_a",
            round_id=1,
            model_version=0,
            gradients={"param_0": torch.ones(2, 2)},
            dataset_size=0,  # Zero weight
            training_time_seconds=1.0,
        )
    ]
    
    with pytest.raises(ValueError, match="Total dataset size is zero"):
        aggregator.aggregate(updates)


def test_secure_aggregator_insufficient_clients_validation():
    """Test that insufficient clients are rejected when threshold is set."""
    aggregator = SecureAggregator(
        poly_modulus_degree=4096,
        dropout_threshold=0.5,
    )
    
    # Setup expecting 4 clients
    expected_clients = ["hospital_a", "hospital_b", "hospital_c", "hospital_d"]
    
    try:
        aggregator.setup_round(expected_clients)
        
        # Only 1 client responds (below 50% threshold)
        updates = [
            ClientUpdate(
                client_id="hospital_a",
                round_id=1,
                model_version=0,
                gradients={"param_0": torch.ones(2, 2)},
                dataset_size=100,
                training_time_seconds=1.0,
            )
        ]
        
        # Should fail due to insufficient clients
        with pytest.raises(ValueError, match="Insufficient clients"):
            aggregator.aggregate(updates)
            
    except RuntimeError:
        # TenSEAL not available - skip
        pytest.skip("TenSEAL not available")


def test_secure_aggregator_string_representations():
    """Test string representations."""
    aggregator = SecureAggregator(
        poly_modulus_degree=4096,
        max_workers=2,
        dropout_threshold=0.6,
    )
    
    # Test __str__
    str_repr = str(aggregator)
    assert "SecureAggregator" in str_repr
    assert "4096" in str_repr
    
    # Test __repr__
    repr_str = repr(aggregator)
    assert "SecureAggregator" in repr_str
    assert "poly_modulus_degree=4096" in repr_str
    assert "max_workers=2" in repr_str
    assert "dropout_threshold=0.6" in repr_str


def test_secure_aggregator_default_config():
    """Test default configuration values."""
    aggregator = SecureAggregator()
    
    # Check defaults
    assert aggregator.poly_modulus_degree == 8192
    assert aggregator.max_workers == 4
    assert aggregator.dropout_threshold == 0.5
    assert aggregator.algorithm_name == "SecureAggregation"


def test_secure_aggregator_get_public_context():
    """Test getting public context."""
    aggregator = SecureAggregator(poly_modulus_degree=4096)
    
    try:
        public_context = aggregator.get_public_context()
        assert isinstance(public_context, bytes)
    except RuntimeError as e:
        # Expected if TenSEAL not available
        assert "TenSEAL not available" in str(e)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
