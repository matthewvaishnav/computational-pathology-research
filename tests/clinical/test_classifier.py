"""
Unit tests for multi-class disease state classifier.

Tests probability distribution properties, primary diagnosis identification,
multiple taxonomy configurations, and integration with existing MIL models.
"""

import pytest
import torch
import torch.nn as nn

from src.clinical.classifier import MultiClassDiseaseClassifier
from src.clinical.taxonomy import DiseaseTaxonomy


class TestMultiClassDiseaseClassifier:
    """Test cases for MultiClassDiseaseClassifier class."""

    @pytest.fixture
    def simple_taxonomy(self):
        """Simple 3-class taxonomy for testing."""
        return DiseaseTaxonomy(
            config_dict={
                "name": "Simple Cancer Grading",
                "version": "1.0",
                "diseases": [
                    {"id": "benign", "name": "Benign", "parent": None, "children": []},
                    {"id": "grade_1", "name": "Grade 1", "parent": None, "children": []},
                    {"id": "grade_2", "name": "Grade 2", "parent": None, "children": []},
                ],
            }
        )

    @pytest.fixture
    def complex_taxonomy(self):
        """Complex hierarchical taxonomy for testing."""
        return DiseaseTaxonomy(
            config_dict={
                "name": "Complex Cancer Classification",
                "version": "1.0",
                "diseases": [
                    {"id": "benign", "name": "Benign", "parent": None, "children": []},
                    {
                        "id": "malignant",
                        "name": "Malignant",
                        "parent": None,
                        "children": ["grade_1", "grade_2", "grade_3"],
                    },
                    {"id": "grade_1", "name": "Grade 1", "parent": "malignant", "children": []},
                    {"id": "grade_2", "name": "Grade 2", "parent": "malignant", "children": []},
                    {"id": "grade_3", "name": "Grade 3", "parent": "malignant", "children": []},
                ],
            }
        )

    @pytest.fixture
    def classifier(self, simple_taxonomy):
        """Basic classifier instance."""
        return MultiClassDiseaseClassifier(
            taxonomy=simple_taxonomy,
            input_dim=256,
            hidden_dim=128,
            dropout=0.3,
            use_hidden_layer=True,
        )

    def test_initialization(self, simple_taxonomy):
        """Test classifier initialization."""
        classifier = MultiClassDiseaseClassifier(
            taxonomy=simple_taxonomy, input_dim=256, hidden_dim=128
        )

        assert classifier.num_classes == 3
        assert classifier.input_dim == 256
        assert classifier.hidden_dim == 128
        assert len(classifier.disease_ids) == 3
        assert "benign" in classifier.disease_ids
        assert "grade_1" in classifier.disease_ids
        assert "grade_2" in classifier.disease_ids

    def test_initialization_invalid_taxonomy(self):
        """Test initialization with invalid taxonomy type."""
        with pytest.raises(TypeError):
            MultiClassDiseaseClassifier(taxonomy="not_a_taxonomy", input_dim=256)

    def test_forward_output_structure(self, classifier):
        """Test forward pass output structure."""
        batch_size = 16
        embeddings = torch.randn(batch_size, 256)

        output = classifier(embeddings)

        # Check output keys
        assert "probabilities" in output
        assert "primary_diagnosis" in output
        assert "confidence" in output
        assert "logits" not in output  # Not returned by default

        # Check shapes
        assert output["probabilities"].shape == (batch_size, 3)
        assert output["primary_diagnosis"].shape == (batch_size,)
        assert output["confidence"].shape == (batch_size,)

    def test_forward_with_logits(self, classifier):
        """Test forward pass with logits return."""
        batch_size = 16
        embeddings = torch.randn(batch_size, 256)

        output = classifier(embeddings, return_logits=True)

        assert "logits" in output
        assert output["logits"].shape == (batch_size, 3)

    def test_probability_distribution_properties(self, classifier):
        """
        Test probability distribution properties (sum to 1.0, values in [0,1]).

        **Validates: Requirements 1.1, 1.2, 1.7, 1.8**
        """
        batch_size = 32
        embeddings = torch.randn(batch_size, 256)

        output = classifier(embeddings)
        probabilities = output["probabilities"]

        # Test 1: All probabilities in [0, 1]
        assert torch.all(probabilities >= 0.0), "Some probabilities are negative"
        assert torch.all(probabilities <= 1.0), "Some probabilities exceed 1.0"

        # Test 2: Probabilities sum to 1.0 (within numerical tolerance)
        prob_sums = probabilities.sum(dim=1)
        assert torch.allclose(
            prob_sums, torch.ones(batch_size), atol=1e-6
        ), f"Probability sums not equal to 1.0: {prob_sums}"

        # Test 3: No NaN or Inf values
        assert not torch.any(torch.isnan(probabilities)), "NaN values in probabilities"
        assert not torch.any(torch.isinf(probabilities)), "Inf values in probabilities"

    def test_primary_diagnosis_identification(self, classifier):
        """
        Test primary diagnosis identification (highest probability).

        **Validates: Requirements 1.3**
        """
        batch_size = 16
        embeddings = torch.randn(batch_size, 256)

        output = classifier(embeddings)
        probabilities = output["probabilities"]
        primary_diagnosis = output["primary_diagnosis"]
        confidence = output["confidence"]

        # Verify primary diagnosis is the argmax of probabilities
        expected_primary = torch.argmax(probabilities, dim=1)
        assert torch.all(
            primary_diagnosis == expected_primary
        ), "Primary diagnosis does not match highest probability"

        # Verify confidence matches the probability of primary diagnosis
        for i in range(batch_size):
            expected_confidence = probabilities[i, primary_diagnosis[i]]
            assert torch.isclose(
                confidence[i], expected_confidence, atol=1e-6
            ), f"Confidence mismatch at sample {i}"

    def test_multiple_taxonomy_configurations(self, simple_taxonomy, complex_taxonomy):
        """
        Test multiple taxonomy configurations.

        **Validates: Requirements 1.4**
        """
        # Test simple taxonomy
        classifier_simple = MultiClassDiseaseClassifier(taxonomy=simple_taxonomy, input_dim=256)
        embeddings = torch.randn(8, 256)
        output_simple = classifier_simple(embeddings)

        assert output_simple["probabilities"].shape == (8, 3)
        assert classifier_simple.num_classes == 3

        # Test complex taxonomy
        classifier_complex = MultiClassDiseaseClassifier(taxonomy=complex_taxonomy, input_dim=256)
        output_complex = classifier_complex(embeddings)

        assert output_complex["probabilities"].shape == (8, 5)
        assert classifier_complex.num_classes == 5

        # Verify both produce valid probability distributions
        assert torch.allclose(output_simple["probabilities"].sum(dim=1), torch.ones(8), atol=1e-6)
        assert torch.allclose(output_complex["probabilities"].sum(dim=1), torch.ones(8), atol=1e-6)

    def test_integration_with_different_input_dimensions(self, simple_taxonomy):
        """
        Test integration with different input dimensions (MIL model outputs).

        **Validates: Requirements 1.8**
        """
        # Test with different input dimensions (simulating different MIL models)
        input_dims = [128, 256, 512, 1024]
        batch_size = 8

        for input_dim in input_dims:
            classifier = MultiClassDiseaseClassifier(taxonomy=simple_taxonomy, input_dim=input_dim)
            embeddings = torch.randn(batch_size, input_dim)
            output = classifier(embeddings)

            # Verify valid probability distribution
            assert output["probabilities"].shape == (batch_size, 3)
            assert torch.allclose(
                output["probabilities"].sum(dim=1), torch.ones(batch_size), atol=1e-6
            )

    def test_predict_disease_ids(self, classifier):
        """Test disease ID prediction."""
        batch_size = 8
        embeddings = torch.randn(batch_size, 256)

        disease_ids, probabilities, confidence = classifier.predict_disease_ids(embeddings)

        # Check types and shapes
        assert isinstance(disease_ids, list)
        assert len(disease_ids) == batch_size
        assert all(isinstance(did, str) for did in disease_ids)
        assert probabilities.shape == (batch_size, 3)
        assert confidence.shape == (batch_size,)

        # Verify disease IDs are valid
        valid_ids = {"benign", "grade_1", "grade_2"}
        assert all(did in valid_ids for did in disease_ids)

    def test_get_disease_probabilities(self, classifier):
        """Test getting probabilities by disease ID."""
        batch_size = 8
        embeddings = torch.randn(batch_size, 256)

        disease_probs = classifier.get_disease_probabilities(embeddings)

        # Check structure
        assert isinstance(disease_probs, dict)
        assert "benign" in disease_probs
        assert "grade_1" in disease_probs
        assert "grade_2" in disease_probs

        # Check shapes
        for disease_id, probs in disease_probs.items():
            assert probs.shape == (batch_size,)

        # Verify probabilities sum to 1.0
        total_probs = sum(disease_probs.values())
        assert torch.allclose(total_probs, torch.ones(batch_size), atol=1e-6)

    def test_get_top_k_diagnoses(self, classifier):
        """Test top-k diagnosis retrieval."""
        batch_size = 8
        embeddings = torch.randn(batch_size, 256)

        # Test k=2
        top_k_ids, top_k_probs = classifier.get_top_k_diagnoses(embeddings, k=2)

        assert len(top_k_ids) == batch_size
        assert top_k_probs.shape == (batch_size, 2)

        # Verify each sample has k disease IDs
        for sample_ids in top_k_ids:
            assert len(sample_ids) == 2
            assert all(isinstance(did, str) for did in sample_ids)

        # Verify probabilities are sorted in descending order
        for i in range(batch_size):
            assert top_k_probs[i, 0] >= top_k_probs[i, 1]

    def test_update_taxonomy(self, classifier, complex_taxonomy):
        """Test dynamic taxonomy update."""
        # Initial state
        assert classifier.num_classes == 3

        # Update to complex taxonomy
        classifier.update_taxonomy(complex_taxonomy)

        # Verify update
        assert classifier.num_classes == 5
        assert len(classifier.disease_ids) == 5
        assert "malignant" in classifier.disease_ids

        # Test forward pass with new taxonomy
        embeddings = torch.randn(8, 256)
        output = classifier(embeddings)

        assert output["probabilities"].shape == (8, 5)
        assert torch.allclose(output["probabilities"].sum(dim=1), torch.ones(8), atol=1e-6)

    def test_update_taxonomy_invalid_type(self, classifier):
        """Test taxonomy update with invalid type."""
        with pytest.raises(TypeError):
            classifier.update_taxonomy("not_a_taxonomy")

    def test_get_taxonomy_info(self, classifier):
        """Test taxonomy information retrieval."""
        info = classifier.get_taxonomy_info()

        assert isinstance(info, dict)
        assert info["name"] == "Simple Cancer Grading"
        assert info["version"] == "1.0"
        assert info["num_classes"] == 3
        assert len(info["disease_ids"]) == 3
        assert "root_diseases" in info
        assert "leaf_diseases" in info

    def test_invalid_embedding_shape(self, classifier):
        """Test error handling for invalid embedding shapes."""
        # Test 1D input
        with pytest.raises(ValueError, match="Expected 2D embeddings"):
            classifier(torch.randn(256))

        # Test 3D input
        with pytest.raises(ValueError, match="Expected 2D embeddings"):
            classifier(torch.randn(8, 10, 256))

        # Test wrong input dimension
        with pytest.raises(ValueError, match="Expected input_dim"):
            classifier(torch.randn(8, 128))  # Wrong dimension

    def test_classifier_without_hidden_layer(self, simple_taxonomy):
        """Test classifier with simple linear head."""
        classifier = MultiClassDiseaseClassifier(
            taxonomy=simple_taxonomy, input_dim=256, use_hidden_layer=False
        )

        embeddings = torch.randn(8, 256)
        output = classifier(embeddings)

        # Verify valid probability distribution
        assert output["probabilities"].shape == (8, 3)
        assert torch.allclose(output["probabilities"].sum(dim=1), torch.ones(8), atol=1e-6)

    def test_gradient_flow(self, classifier):
        """Test gradient flow through classifier."""
        embeddings = torch.randn(8, 256, requires_grad=True)
        output = classifier(embeddings, return_logits=True)

        # Compute loss using logits (more stable gradients)
        # Use a proper loss function instead of sum
        target = torch.randint(0, 3, (8,))
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(output["logits"], target)
        loss.backward()

        # Verify gradients exist and are non-zero
        assert embeddings.grad is not None
        # Check that at least some gradients are non-zero (more robust than checking all)
        assert torch.any(embeddings.grad != 0), "All gradients are zero - no gradient flow"

    def test_batch_size_one(self, classifier):
        """Test with batch size of 1."""
        embeddings = torch.randn(1, 256)
        output = classifier(embeddings)

        assert output["probabilities"].shape == (1, 3)
        assert torch.allclose(output["probabilities"].sum(dim=1), torch.ones(1), atol=1e-6)

    def test_large_batch_size(self, classifier):
        """Test with large batch size."""
        batch_size = 256
        embeddings = torch.randn(batch_size, 256)
        output = classifier(embeddings)

        assert output["probabilities"].shape == (batch_size, 3)
        assert torch.allclose(output["probabilities"].sum(dim=1), torch.ones(batch_size), atol=1e-6)

    def test_deterministic_output(self, classifier):
        """Test deterministic output for same input."""
        classifier.eval()  # Set to eval mode to disable dropout

        embeddings = torch.randn(8, 256)

        # Run twice with same input
        output1 = classifier(embeddings)
        output2 = classifier(embeddings)

        # Verify outputs are identical
        assert torch.allclose(output1["probabilities"], output2["probabilities"])
        assert torch.all(output1["primary_diagnosis"] == output2["primary_diagnosis"])

    def test_repr(self, classifier):
        """Test string representation."""
        repr_str = repr(classifier)

        assert "MultiClassDiseaseClassifier" in repr_str
        assert "Simple Cancer Grading" in repr_str
        assert "num_classes=3" in repr_str
        assert "input_dim=256" in repr_str


class TestMultiClassDiseaseClassifierIntegration:
    """Integration tests with MIL models and multimodal fusion."""

    @pytest.fixture
    def taxonomy(self):
        """Taxonomy for integration tests."""
        return DiseaseTaxonomy(
            config_dict={
                "name": "Integration Test Taxonomy",
                "version": "1.0",
                "diseases": [
                    {"id": "normal", "name": "Normal", "parent": None, "children": []},
                    {"id": "abnormal", "name": "Abnormal", "parent": None, "children": []},
                ],
            }
        )

    def test_integration_with_attention_mil_output(self, taxonomy):
        """
        Test integration with AttentionMIL-like output.

        **Validates: Requirements 1.8**
        """
        # Simulate AttentionMIL output (aggregated features)
        batch_size = 16
        feature_dim = 256
        mil_output = torch.randn(batch_size, feature_dim)

        # Create classifier
        classifier = MultiClassDiseaseClassifier(taxonomy=taxonomy, input_dim=feature_dim)

        # Forward pass
        output = classifier(mil_output)

        # Verify valid probability distribution
        assert output["probabilities"].shape == (batch_size, 2)
        assert torch.allclose(output["probabilities"].sum(dim=1), torch.ones(batch_size), atol=1e-6)

    def test_integration_with_multimodal_fusion(self, taxonomy):
        """
        Test integration with multimodal fusion output.

        **Validates: Requirements 1.8**
        """
        # Simulate multimodal fusion output
        batch_size = 8
        fusion_dim = 256
        fusion_output = torch.randn(batch_size, fusion_dim)

        # Create classifier
        classifier = MultiClassDiseaseClassifier(taxonomy=taxonomy, input_dim=fusion_dim)

        # Forward pass
        output = classifier(fusion_output)

        # Verify valid probability distribution
        assert output["probabilities"].shape == (batch_size, 2)
        assert torch.allclose(output["probabilities"].sum(dim=1), torch.ones(batch_size), atol=1e-6)

    def test_end_to_end_pipeline(self, taxonomy):
        """
        Test end-to-end pipeline: encoder -> classifier.

        **Validates: Requirements 1.1, 1.2, 1.3, 1.7, 1.8**
        """
        # Simulate simple encoder
        encoder = nn.Sequential(nn.Linear(1024, 512), nn.ReLU(), nn.Linear(512, 256))

        # Create classifier
        classifier = MultiClassDiseaseClassifier(taxonomy=taxonomy, input_dim=256)

        # End-to-end forward pass
        batch_size = 16
        raw_features = torch.randn(batch_size, 1024)
        embeddings = encoder(raw_features)
        output = classifier(embeddings)

        # Verify valid probability distribution
        assert output["probabilities"].shape == (batch_size, 2)
        probabilities = output["probabilities"]

        # Test all invariant properties
        assert torch.all(probabilities >= 0.0)
        assert torch.all(probabilities <= 1.0)
        assert torch.allclose(probabilities.sum(dim=1), torch.ones(batch_size), atol=1e-6)

        # Test gradient flow through entire pipeline
        loss = probabilities.sum()
        loss.backward()

        # Verify gradients exist in encoder
        for param in encoder.parameters():
            assert param.grad is not None
