"""
Tests for secure aggregation using homomorphic encryption.

Tests that the central server cannot access individual hospital updates
while still computing correct aggregated results.
"""

import pytest
import torch

from src.federated.aggregator.secure import SecureAggregator
from src.federated.common.data_models import ClientUpdate

# Check if TenSEAL is available
try:
    import tenseal as ts

    TENSEAL_AVAILABLE = True
except ImportError:
    TENSEAL_AVAILABLE = False


@pytest.mark.skipif(not TENSEAL_AVAILABLE, reason="TenSEAL not available")
class TestSecureAggregation:
    """Test suite for secure aggregation."""

    def test_secure_aggregator_initialization(self):
        """Test secure aggregator initializes correctly."""
        aggregator = SecureAggregator(
            poly_modulus_degree=4096,
            max_workers=2,
            dropout_threshold=0.5,
        )

        assert aggregator.algorithm_name == "SecureAggregation"
        assert aggregator.poly_modulus_degree == 4096
        assert aggregator.max_workers == 2
        assert aggregator.dropout_threshold == 0.5

    def test_secure_aggregation_basic(self):
        """Test basic secure aggregation with 2 clients."""
        aggregator = SecureAggregator(poly_modulus_degree=4096)

        # Create client updates
        updates = [
            ClientUpdate(
                client_id="hospital_a",
                round_id=1,
                model_version=0,
                gradients={"param_0": torch.ones(3, 3)},
                dataset_size=100,
                training_time_seconds=1.0,
            ),
            ClientUpdate(
                client_id="hospital_b",
                round_id=1,
                model_version=0,
                gradients={"param_0": torch.ones(3, 3) * 2},
                dataset_size=100,
                training_time_seconds=1.0,
            ),
        ]

        # Aggregate securely
        aggregated = aggregator.aggregate(updates)

        # Expected: (1*100 + 2*100) / 200 = 1.5
        expected = torch.ones(3, 3) * 1.5

        # Check result (allow small error due to HE approximation)
        assert torch.allclose(aggregated["param_0"], expected, atol=1e-3)

    def test_secure_aggregation_weighted(self):
        """Test secure aggregation with different dataset sizes."""
        aggregator = SecureAggregator(poly_modulus_degree=4096)

        # Create client updates with different weights
        updates = [
            ClientUpdate(
                client_id="hospital_a",
                round_id=1,
                model_version=0,
                gradients={"param_0": torch.ones(2, 2) * 1.0},
                dataset_size=300,  # 75% weight
                training_time_seconds=1.0,
            ),
            ClientUpdate(
                client_id="hospital_b",
                round_id=1,
                model_version=0,
                gradients={"param_0": torch.ones(2, 2) * 4.0},
                dataset_size=100,  # 25% weight
                training_time_seconds=1.0,
            ),
        ]

        # Aggregate securely
        aggregated = aggregator.aggregate(updates)

        # Expected: (1.0 * 0.75) + (4.0 * 0.25) = 0.75 + 1.0 = 1.75
        expected = torch.ones(2, 2) * 1.75

        assert torch.allclose(aggregated["param_0"], expected, atol=1e-3)

    def test_secure_aggregation_multiple_params(self):
        """Test secure aggregation with multiple parameters."""
        aggregator = SecureAggregator(poly_modulus_degree=4096)

        # Create client updates with multiple parameters
        updates = [
            ClientUpdate(
                client_id="hospital_a",
                round_id=1,
                model_version=0,
                gradients={
                    "layer1.weight": torch.ones(2, 2),
                    "layer1.bias": torch.ones(2),
                    "layer2.weight": torch.ones(3, 2),
                },
                dataset_size=100,
                training_time_seconds=1.0,
            ),
            ClientUpdate(
                client_id="hospital_b",
                round_id=1,
                model_version=0,
                gradients={
                    "layer1.weight": torch.ones(2, 2) * 2,
                    "layer1.bias": torch.ones(2) * 2,
                    "layer2.weight": torch.ones(3, 2) * 2,
                },
                dataset_size=100,
                training_time_seconds=1.0,
            ),
        ]

        # Aggregate securely
        aggregated = aggregator.aggregate(updates)

        # Check all parameters
        assert len(aggregated) == 3
        assert torch.allclose(aggregated["layer1.weight"], torch.ones(2, 2) * 1.5, atol=1e-3)
        assert torch.allclose(aggregated["layer1.bias"], torch.ones(2) * 1.5, atol=1e-3)
        assert torch.allclose(aggregated["layer2.weight"], torch.ones(3, 2) * 1.5, atol=1e-3)

    def test_secure_aggregation_empty_fails(self):
        """Test that aggregating empty list raises error."""
        aggregator = SecureAggregator(poly_modulus_degree=4096)

        with pytest.raises(ValueError, match="Cannot aggregate empty list"):
            aggregator.aggregate([])

    def test_secure_aggregation_zero_weight_fails(self):
        """Test that zero total dataset size raises error."""
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

    def test_dropout_handling(self):
        """Test dropout handling with minimum client threshold."""
        aggregator = SecureAggregator(
            poly_modulus_degree=4096,
            dropout_threshold=0.5,  # Need at least 50% of clients
        )

        # Setup round with 4 expected clients
        expected_clients = ["hospital_a", "hospital_b", "hospital_c", "hospital_d"]
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

    def test_dropout_handling_success(self):
        """Test successful aggregation with sufficient clients after dropout."""
        aggregator = SecureAggregator(
            poly_modulus_degree=4096,
            dropout_threshold=0.5,  # Need at least 50% of clients
        )

        # Setup round with 4 expected clients
        expected_clients = ["hospital_a", "hospital_b", "hospital_c", "hospital_d"]
        aggregator.setup_round(expected_clients)

        # 3 clients respond (75% > 50% threshold)
        updates = [
            ClientUpdate(
                client_id="hospital_a",
                round_id=1,
                model_version=0,
                gradients={"param_0": torch.ones(2, 2)},
                dataset_size=100,
                training_time_seconds=1.0,
            ),
            ClientUpdate(
                client_id="hospital_b",
                round_id=1,
                model_version=0,
                gradients={"param_0": torch.ones(2, 2) * 2},
                dataset_size=100,
                training_time_seconds=1.0,
            ),
            ClientUpdate(
                client_id="hospital_c",
                round_id=1,
                model_version=0,
                gradients={"param_0": torch.ones(2, 2) * 3},
                dataset_size=100,
                training_time_seconds=1.0,
            ),
        ]

        # Should succeed
        aggregated = aggregator.aggregate(updates)

        # Expected: (1 + 2 + 3) / 3 = 2
        expected = torch.ones(2, 2) * 2
        assert torch.allclose(aggregated["param_0"], expected, atol=1e-3)

    def test_public_context_generation(self):
        """Test that public context can be generated for clients."""
        aggregator = SecureAggregator(poly_modulus_degree=4096)

        # Setup round
        public_context = aggregator.setup_round(["hospital_a", "hospital_b"])

        # Should return serialized context
        assert isinstance(public_context, bytes)
        assert len(public_context) > 0

    def test_secure_aggregation_accuracy(self):
        """Test that secure aggregation maintains accuracy vs plaintext."""
        aggregator = SecureAggregator(poly_modulus_degree=8192)  # Higher precision

        # Create realistic gradients
        updates = []
        for i in range(3):
            gradients = {
                "conv1.weight": torch.randn(16, 3, 3, 3) * 0.01,
                "conv1.bias": torch.randn(16) * 0.01,
                "fc.weight": torch.randn(10, 256) * 0.01,
                "fc.bias": torch.randn(10) * 0.01,
            }
            updates.append(
                ClientUpdate(
                    client_id=f"hospital_{i}",
                    round_id=1,
                    model_version=0,
                    gradients=gradients,
                    dataset_size=100,
                    training_time_seconds=1.0,
                )
            )

        # Secure aggregation
        secure_result = aggregator.aggregate(updates)

        # Plaintext aggregation (for comparison)
        plaintext_result = {}
        for param_name in secure_result.keys():
            plaintext_result[param_name] = torch.zeros_like(secure_result[param_name])
            for update in updates:
                plaintext_result[param_name] += update.gradients[param_name]
            plaintext_result[param_name] /= len(updates)

        # Check accuracy (should be very close)
        for param_name in secure_result.keys():
            max_error = torch.max(
                torch.abs(secure_result[param_name] - plaintext_result[param_name])
            ).item()
            # Allow small error due to HE approximation
            assert max_error < 1e-3, f"Error too large for {param_name}: {max_error}"

    def test_string_representation(self):
        """Test string representations."""
        aggregator = SecureAggregator(
            poly_modulus_degree=4096,
            max_workers=2,
            dropout_threshold=0.6,
        )

        str_repr = str(aggregator)
        assert "SecureAggregator" in str_repr
        assert "4096" in str_repr

        repr_str = repr(aggregator)
        assert "SecureAggregator" in repr_str
        assert "poly_modulus_degree=4096" in repr_str
        assert "max_workers=2" in repr_str
        assert "dropout_threshold=0.6" in repr_str


@pytest.mark.skipif(not TENSEAL_AVAILABLE, reason="TenSEAL not available")
def test_secure_aggregator_integration():
    """Integration test: secure aggregation in realistic scenario."""
    # Simulate 5 hospitals with different dataset sizes
    aggregator = SecureAggregator(
        poly_modulus_degree=8192,
        dropout_threshold=0.6,  # Need at least 60% of hospitals
    )

    # Setup round
    expected_hospitals = [f"hospital_{i}" for i in range(5)]
    public_context = aggregator.setup_round(expected_hospitals)

    assert len(public_context) > 0

    # 4 hospitals respond (80% > 60% threshold)
    updates = []
    dataset_sizes = [500, 300, 200, 100]  # Different sizes

    for i, size in enumerate(dataset_sizes):
        gradients = {
            "encoder.weight": torch.randn(128, 64) * 0.01,
            "encoder.bias": torch.randn(128) * 0.01,
            "classifier.weight": torch.randn(2, 128) * 0.01,
            "classifier.bias": torch.randn(2) * 0.01,
        }
        updates.append(
            ClientUpdate(
                client_id=f"hospital_{i}",
                round_id=1,
                model_version=0,
                gradients=gradients,
                dataset_size=size,
                training_time_seconds=float(i + 1),
            )
        )

    # Aggregate securely
    aggregated = aggregator.aggregate(updates)

    # Verify all parameters aggregated
    assert len(aggregated) == 4
    assert "encoder.weight" in aggregated
    assert "encoder.bias" in aggregated
    assert "classifier.weight" in aggregated
    assert "classifier.bias" in aggregated

    # Verify shapes preserved
    assert aggregated["encoder.weight"].shape == (128, 64)
    assert aggregated["encoder.bias"].shape == (128,)
    assert aggregated["classifier.weight"].shape == (2, 128)
    assert aggregated["classifier.bias"].shape == (2,)

    # Verify weighted averaging (manually compute expected)
    total_size = sum(dataset_sizes)
    weights = [size / total_size for size in dataset_sizes]

    # Check one parameter in detail
    expected_encoder_weight = sum(
        updates[i].gradients["encoder.weight"] * weights[i] for i in range(len(updates))
    )

    assert torch.allclose(
        aggregated["encoder.weight"],
        expected_encoder_weight,
        atol=1e-3,
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
