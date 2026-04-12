"""
Unit tests for risk factor analysis module.

Tests risk score calculation, range validation, time horizon predictions,
and threshold flagging logic.
"""

import pytest
import torch

from src.clinical.patient_context import ClinicalMetadata, Sex, SmokingStatus
from src.clinical.risk_analysis import ClinicalRiskFactorEncoder, RiskAnalyzer
from src.clinical.taxonomy import DiseaseTaxonomy


@pytest.fixture
def sample_taxonomy():
    """Create a sample disease taxonomy for testing."""
    return DiseaseTaxonomy(
        config_dict={
            "name": "Cancer Risk Taxonomy",
            "version": "1.0",
            "diseases": [
                {"id": "benign", "name": "Benign", "parent": None, "children": []},
                {"id": "grade_1", "name": "Grade 1", "parent": None, "children": []},
                {"id": "grade_2", "name": "Grade 2", "parent": None, "children": []},
            ],
        }
    )


@pytest.fixture
def sample_clinical_metadata():
    """Create sample clinical metadata for testing."""
    return [
        ClinicalMetadata(
            age=65,
            sex=Sex.MALE,
            smoking_status=SmokingStatus.FORMER,
            family_history=["cancer", "diabetes"],
        ),
        ClinicalMetadata(
            age=45,
            sex=Sex.FEMALE,
            smoking_status=SmokingStatus.NEVER,
            family_history=[],
        ),
        ClinicalMetadata(
            age=70,
            sex=Sex.MALE,
            smoking_status=SmokingStatus.CURRENT,
            family_history=["cancer", "heart_disease", "hypertension"],
        ),
    ]


class TestRiskAnalyzer:
    """Test suite for RiskAnalyzer class."""

    def test_initialization(self, sample_taxonomy):
        """Test RiskAnalyzer initialization."""
        analyzer = RiskAnalyzer(sample_taxonomy, input_dim=256, hidden_dim=128)

        assert analyzer.num_diseases == 3
        assert analyzer.time_horizons == [1, 5, 10]
        assert analyzer.num_time_horizons == 3
        assert len(analyzer.disease_ids) == 3
        assert "benign" in analyzer.disease_ids

    def test_initialization_custom_time_horizons(self, sample_taxonomy):
        """Test RiskAnalyzer with custom time horizons."""
        analyzer = RiskAnalyzer(sample_taxonomy, time_horizons=[1, 3, 5, 10])

        assert analyzer.time_horizons == [1, 3, 5, 10]
        assert analyzer.num_time_horizons == 4

    def test_forward_without_clinical_metadata(self, sample_taxonomy):
        """Test forward pass without clinical metadata."""
        analyzer = RiskAnalyzer(sample_taxonomy, input_dim=256)
        batch_size = 4

        imaging_features = torch.randn(batch_size, 256)
        output = analyzer(imaging_features)

        # Check output structure
        assert "risk_scores" in output
        assert "anomaly_scores" in output
        assert "primary_risk_disease" in output
        assert "max_risk_scores" in output

        # Check shapes
        assert output["risk_scores"].shape == (batch_size, 3, 3)  # [batch, diseases, horizons]
        assert output["anomaly_scores"].shape == (batch_size, 3)  # [batch, diseases]
        assert output["primary_risk_disease"].shape == (batch_size, 3)  # [batch, horizons]
        assert output["max_risk_scores"].shape == (batch_size, 3)  # [batch, horizons]

    def test_forward_with_clinical_metadata(self, sample_taxonomy, sample_clinical_metadata):
        """Test forward pass with clinical metadata."""
        analyzer = RiskAnalyzer(sample_taxonomy, input_dim=256)
        batch_size = len(sample_clinical_metadata)

        imaging_features = torch.randn(batch_size, 256)
        output = analyzer(imaging_features, clinical_metadata=sample_clinical_metadata)

        # Check output structure and shapes
        assert output["risk_scores"].shape == (batch_size, 3, 3)
        assert output["anomaly_scores"].shape == (batch_size, 3)

    def test_risk_score_range_validation(self, sample_taxonomy):
        """Test that risk scores are in valid range [0, 1]."""
        analyzer = RiskAnalyzer(sample_taxonomy, input_dim=256)
        batch_size = 10

        imaging_features = torch.randn(batch_size, 256)
        output = analyzer(imaging_features)

        risk_scores = output["risk_scores"]
        anomaly_scores = output["anomaly_scores"]

        # All risk scores should be in [0, 1]
        assert torch.all(risk_scores >= 0.0)
        assert torch.all(risk_scores <= 1.0)

        # All anomaly scores should be in [0, 1]
        assert torch.all(anomaly_scores >= 0.0)
        assert torch.all(anomaly_scores <= 1.0)

    def test_time_horizon_predictions(self, sample_taxonomy):
        """Test that different time horizons produce different predictions."""
        analyzer = RiskAnalyzer(sample_taxonomy, input_dim=256, time_horizons=[1, 5, 10])
        batch_size = 5

        imaging_features = torch.randn(batch_size, 256)
        output = analyzer(imaging_features)

        risk_scores = output["risk_scores"]

        # Risk scores should have 3 time horizons
        assert risk_scores.shape[2] == 3

        # Different time horizons should produce different scores (with high probability)
        # Check that not all time horizons are identical
        horizon_1 = risk_scores[:, :, 0]
        horizon_2 = risk_scores[:, :, 1]
        horizon_3 = risk_scores[:, :, 2]

        # At least some differences should exist (not all identical)
        assert not torch.allclose(horizon_1, horizon_2, atol=1e-6)
        assert not torch.allclose(horizon_2, horizon_3, atol=1e-6)

    def test_get_risk_by_disease_id(self, sample_taxonomy):
        """Test getting risk scores organized by disease ID."""
        analyzer = RiskAnalyzer(sample_taxonomy, input_dim=256)
        batch_size = 4

        imaging_features = torch.randn(batch_size, 256)
        risk_by_disease = analyzer.get_risk_by_disease_id(imaging_features)

        # Check that all disease IDs are present
        assert "benign" in risk_by_disease
        assert "grade_1" in risk_by_disease
        assert "grade_2" in risk_by_disease

        # Check shapes
        for disease_id, risk_scores in risk_by_disease.items():
            assert risk_scores.shape == (batch_size, 3)  # [batch, time_horizons]

    def test_get_high_risk_cases(self, sample_taxonomy):
        """Test identification of high-risk cases."""
        analyzer = RiskAnalyzer(sample_taxonomy, input_dim=256)
        batch_size = 10

        imaging_features = torch.randn(batch_size, 256)

        # Test with different thresholds
        for threshold in [0.3, 0.5, 0.7]:
            high_risk_mask, flagged_details = analyzer.get_high_risk_cases(
                imaging_features, threshold=threshold
            )

            # Check mask shape
            assert high_risk_mask.shape == (batch_size,)
            assert high_risk_mask.dtype == torch.bool

            # Check flagged details
            num_flagged = high_risk_mask.sum().item()
            if num_flagged > 0:
                assert flagged_details["risk_scores"].shape[0] == num_flagged
                assert flagged_details["anomaly_scores"].shape[0] == num_flagged

    def test_get_time_horizon_names(self, sample_taxonomy):
        """Test getting human-readable time horizon names."""
        analyzer = RiskAnalyzer(sample_taxonomy, time_horizons=[1, 5, 10])

        horizon_names = analyzer.get_time_horizon_names()

        assert horizon_names == ["1-year", "5-year", "10-year"]

    def test_invalid_input_shape(self, sample_taxonomy):
        """Test error handling for invalid input shapes."""
        analyzer = RiskAnalyzer(sample_taxonomy, input_dim=256)

        # 1D input (should fail)
        with pytest.raises(ValueError, match="Expected 2D"):
            analyzer(torch.randn(256))

        # 3D input (should fail)
        with pytest.raises(ValueError, match="Expected 2D"):
            analyzer(torch.randn(4, 256, 10))

        # Wrong input dimension
        with pytest.raises(ValueError, match="Expected input_dim"):
            analyzer(torch.randn(4, 128))

    def test_clinical_metadata_batch_size_mismatch(self, sample_taxonomy, sample_clinical_metadata):
        """Test error handling for batch size mismatch."""
        analyzer = RiskAnalyzer(sample_taxonomy, input_dim=256)

        imaging_features = torch.randn(5, 256)  # batch_size = 5
        # sample_clinical_metadata has 3 samples

        with pytest.raises(ValueError, match="batch size"):
            analyzer(imaging_features, clinical_metadata=sample_clinical_metadata)

    def test_repr(self, sample_taxonomy):
        """Test string representation."""
        analyzer = RiskAnalyzer(sample_taxonomy, input_dim=256, hidden_dim=128)

        repr_str = repr(analyzer)

        assert "RiskAnalyzer" in repr_str
        assert "Cancer Risk Taxonomy" in repr_str
        assert "num_diseases=3" in repr_str


class TestClinicalRiskFactorEncoder:
    """Test suite for ClinicalRiskFactorEncoder class."""

    def test_initialization(self):
        """Test encoder initialization."""
        encoder = ClinicalRiskFactorEncoder(embed_dim=128)

        assert encoder.embed_dim == 128

    def test_forward(self, sample_clinical_metadata):
        """Test encoding clinical risk factors."""
        encoder = ClinicalRiskFactorEncoder(embed_dim=128)
        device = torch.device("cpu")

        embeddings = encoder(sample_clinical_metadata, device=device)

        # Check shape
        assert embeddings.shape == (len(sample_clinical_metadata), 128)

        # Check that embeddings are different for different metadata
        # (with high probability)
        assert not torch.allclose(embeddings[0], embeddings[1], atol=1e-6)
        assert not torch.allclose(embeddings[1], embeddings[2], atol=1e-6)

    def test_smoking_status_encoding(self):
        """Test that different smoking statuses produce different embeddings."""
        encoder = ClinicalRiskFactorEncoder(embed_dim=128)
        device = torch.device("cpu")

        metadata_never = [ClinicalMetadata(age=50, smoking_status=SmokingStatus.NEVER)]
        metadata_current = [ClinicalMetadata(age=50, smoking_status=SmokingStatus.CURRENT)]

        emb_never = encoder(metadata_never, device=device)
        emb_current = encoder(metadata_current, device=device)

        # Different smoking statuses should produce different embeddings
        assert not torch.allclose(emb_never, emb_current, atol=1e-6)

    def test_age_encoding(self):
        """Test that different ages produce different embeddings."""
        encoder = ClinicalRiskFactorEncoder(embed_dim=128)
        device = torch.device("cpu")

        metadata_young = [ClinicalMetadata(age=30, smoking_status=SmokingStatus.NEVER)]
        metadata_old = [ClinicalMetadata(age=70, smoking_status=SmokingStatus.NEVER)]

        emb_young = encoder(metadata_young, device=device)
        emb_old = encoder(metadata_old, device=device)

        # Different ages should produce different embeddings
        assert not torch.allclose(emb_young, emb_old, atol=1e-6)

    def test_family_history_encoding(self):
        """Test that family history affects embeddings."""
        encoder = ClinicalRiskFactorEncoder(embed_dim=128)
        device = torch.device("cpu")

        metadata_no_history = [
            ClinicalMetadata(age=50, smoking_status=SmokingStatus.NEVER, family_history=[])
        ]
        metadata_with_history = [
            ClinicalMetadata(
                age=50,
                smoking_status=SmokingStatus.NEVER,
                family_history=["cancer", "diabetes", "heart_disease"],
            )
        ]

        emb_no_history = encoder(metadata_no_history, device=device)
        emb_with_history = encoder(metadata_with_history, device=device)

        # Different family histories should produce different embeddings
        assert not torch.allclose(emb_no_history, emb_with_history, atol=1e-6)


class TestRiskAnalysisIntegration:
    """Integration tests for risk analysis components."""

    def test_end_to_end_risk_analysis(self, sample_taxonomy, sample_clinical_metadata):
        """Test complete risk analysis workflow."""
        # Create analyzer
        analyzer = RiskAnalyzer(sample_taxonomy, input_dim=256, hidden_dim=128)

        # Generate imaging features
        batch_size = len(sample_clinical_metadata)
        imaging_features = torch.randn(batch_size, 256)

        # Run risk analysis
        output = analyzer(imaging_features, clinical_metadata=sample_clinical_metadata)

        # Verify all outputs are valid
        assert output["risk_scores"].shape == (batch_size, 3, 3)
        assert torch.all(output["risk_scores"] >= 0.0)
        assert torch.all(output["risk_scores"] <= 1.0)

        assert output["anomaly_scores"].shape == (batch_size, 3)
        assert torch.all(output["anomaly_scores"] >= 0.0)
        assert torch.all(output["anomaly_scores"] <= 1.0)

        # Get risk by disease
        risk_by_disease = analyzer.get_risk_by_disease_id(
            imaging_features, clinical_metadata=sample_clinical_metadata
        )

        assert len(risk_by_disease) == 3
        for disease_id in ["benign", "grade_1", "grade_2"]:
            assert disease_id in risk_by_disease
            assert risk_by_disease[disease_id].shape == (batch_size, 3)

    def test_gradient_flow(self, sample_taxonomy):
        """Test that gradients flow through the model."""
        analyzer = RiskAnalyzer(sample_taxonomy, input_dim=256)
        analyzer.train()

        imaging_features = torch.randn(4, 256, requires_grad=True)
        output = analyzer(imaging_features)

        # Compute loss and backpropagate
        loss = output["risk_scores"].mean() + output["anomaly_scores"].mean()
        loss.backward()

        # Check that gradients exist
        assert imaging_features.grad is not None
        assert not torch.all(imaging_features.grad == 0)
