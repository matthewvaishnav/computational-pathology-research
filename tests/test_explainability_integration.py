"""
Integration tests for Phase 2 explainability components
Tests uncertainty quantification, case-based reasoning, and counterfactual explanations
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
from pathlib import Path
import tempfile
import shutil

from src.explainability.vision_language_explainer import VisionLanguageExplainer
from src.explainability.uncertainty_quantification import (
    UncertaintyQuantificationSystem, MonteCarloDropout, EnsembleUncertainty
)
from src.explainability.case_based_reasoning import (
    CaseDatabase, CaseMetadata, RetrievalQuery
)
from src.explainability.counterfactual_explanations import (
    CounterfactualExplanationSystem, BiologicalPlausibilityValidator
)
from src.foundation.multi_disease_model import create_foundation_model


class TestUncertaintyQuantification:
    """Test uncertainty quantification system"""
    
    @pytest.fixture
    def models(self):
        """Create test models"""
        model1 = create_foundation_model(encoder_type="resnet50")
        model2 = create_foundation_model(encoder_type="resnet50")
        return [model1, model2]
    
    @pytest.fixture
    def uncertainty_system(self, models):
        """Create uncertainty quantification system"""
        return UncertaintyQuantificationSystem(
            models=models,
            mc_samples=5,  # Reduced for testing
            calibration_method="platt"
        )
    
    def test_monte_carlo_dropout(self, models):
        """Test Monte Carlo dropout uncertainty estimation"""
        mc_dropout = MonteCarloDropout(models[0], num_samples=5)
        patches = torch.randn(1, 10, 3, 224, 224)
        
        uncertainty_metrics = mc_dropout.estimate_uncertainty(
            patches, disease_type="breast"
        )
        
        assert uncertainty_metrics.total_uncertainty >= 0.0
        assert uncertainty_metrics.epistemic_uncertainty >= 0.0
        assert uncertainty_metrics.aleatoric_uncertainty >= 0.0
        assert len(uncertainty_metrics.confidence_interval) == 2
        assert uncertainty_metrics.confidence_interval[0] <= uncertainty_metrics.confidence_interval[1]
    
    def test_ensemble_uncertainty(self, models):
        """Test ensemble-based uncertainty estimation"""
        ensemble = EnsembleUncertainty(models)
        patches = torch.randn(1, 10, 3, 224, 224)
        
        uncertainty_metrics = ensemble.estimate_uncertainty(
            patches, disease_type="breast"
        )
        
        assert uncertainty_metrics.total_uncertainty >= 0.0
        assert uncertainty_metrics.ensemble_disagreement >= 0.0
        assert uncertainty_metrics.prediction_variance >= 0.0
    
    def test_uncertainty_system_integration(self, uncertainty_system):
        """Test full uncertainty quantification system"""
        patches = torch.randn(1, 10, 3, 224, 224)
        
        # Test ensemble method
        uncertainty_metrics = uncertainty_system.estimate_uncertainty(
            patches, disease_type="breast", method="ensemble"
        )
        
        assert uncertainty_metrics.total_uncertainty >= 0.0
        
        # Test second opinion recommendation
        needs_review, reason = uncertainty_system.should_request_second_opinion(
            uncertainty_metrics
        )
        
        assert isinstance(needs_review, bool)
        assert isinstance(reason, str)
    
    def test_calibration_fitting(self, uncertainty_system):
        """Test confidence calibrator fitting"""
        # Create mock validation data
        validation_patches = [torch.randn(1, 10, 3, 224, 224) for _ in range(10)]
        validation_labels = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        
        calibration_metrics = uncertainty_system.fit_calibrator(
            validation_patches, validation_labels, disease_type="breast"
        )
        
        assert calibration_metrics.expected_calibration_error >= 0.0
        assert calibration_metrics.brier_score >= 0.0


class TestCaseBasedReasoning:
    """Test case-based reasoning system"""
    
    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database path"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def case_database(self, temp_db_path):
        """Create case database"""
        return CaseDatabase(
            database_path=temp_db_path,
            feature_dim=2048,
            index_type="Flat"  # Use flat index for testing
        )
    
    @pytest.fixture
    def sample_metadata(self):
        """Create sample case metadata"""
        return CaseMetadata(
            case_id="test_case_001",
            slide_id="slide_001",
            patient_id="patient_001_anon",
            institution="Test Hospital",
            scanner_type="Test Scanner",
            magnification=40.0,
            stain_type="H&E",
            tissue_type="breast",
            diagnosis="invasive ductal carcinoma",
            grade="Grade 2",
            stage="T2N1M0",
            molecular_markers={"ER": "positive", "PR": "positive"},
            pathologist_id="pathologist_001",
            confidence_score=0.92,
            annotation_time=datetime.now(),
            image_quality_score=0.88,
            artifact_flags=[],
            demographics={"age_group": "50-60", "sex": "F"},
            treatment_response="complete_response",
            follow_up_months=24,
            tags=["training", "high_quality"]
        )
    
    def test_case_addition(self, case_database, sample_metadata):
        """Test adding cases to database"""
        features = torch.randn(2048)
        
        success = case_database.add_case(
            "test_case_001", features, sample_metadata
        )
        
        assert success
        
        # Test duplicate addition
        success = case_database.add_case(
            "test_case_001", features, sample_metadata
        )
        
        assert not success  # Should fail for duplicate
    
    def test_batch_case_addition(self, case_database):
        """Test batch case addition"""
        cases = []
        for i in range(5):
            metadata = CaseMetadata(
                case_id=f"batch_case_{i}",
                slide_id=f"slide_{i}",
                patient_id=f"patient_{i}_anon",
                institution="Test Hospital",
                scanner_type="Test Scanner",
                magnification=40.0,
                stain_type="H&E",
                tissue_type="breast",
                diagnosis="invasive ductal carcinoma",
                grade="Grade 2",
                stage="T2N1M0",
                molecular_markers={},
                pathologist_id="pathologist_001",
                confidence_score=0.9,
                annotation_time=datetime.now(),
                image_quality_score=0.8,
                artifact_flags=[],
                demographics={},
                treatment_response=None,
                follow_up_months=None,
                tags=[]
            )
            features = torch.randn(2048)
            cases.append((f"batch_case_{i}", features, metadata))
        
        added_count = case_database.add_cases_batch(cases)
        assert added_count == 5
    
    def test_case_retrieval(self, case_database, sample_metadata):
        """Test case retrieval"""
        # Add some test cases
        for i in range(3):
            features = torch.randn(2048)
            metadata = CaseMetadata(
                case_id=f"retrieval_case_{i}",
                slide_id=f"slide_{i}",
                patient_id=f"patient_{i}_anon",
                institution="Test Hospital",
                scanner_type="Test Scanner",
                magnification=40.0,
                stain_type="H&E",
                tissue_type="breast",
                diagnosis="invasive ductal carcinoma",
                grade="Grade 2",
                stage="T2N1M0",
                molecular_markers={},
                pathologist_id="pathologist_001",
                confidence_score=0.9,
                annotation_time=datetime.now(),
                image_quality_score=0.8,
                artifact_flags=[],
                demographics={},
                treatment_response=None,
                follow_up_months=None,
                tags=[]
            )
            case_database.add_case(f"retrieval_case_{i}", features, metadata)
        
        # Test retrieval
        query = RetrievalQuery(
            features=torch.randn(2048),
            disease_filter="invasive ductal carcinoma",
            k=2
        )
        
        similar_cases = case_database.retrieve_similar(query)
        
        assert len(similar_cases) <= 2
        for case in similar_cases:
            assert case.similarity_score >= 0.0
            assert case.diagnosis == "invasive ductal carcinoma"
    
    def test_database_statistics(self, case_database, sample_metadata):
        """Test database statistics"""
        # Add test case
        features = torch.randn(2048)
        case_database.add_case("stats_test", features, sample_metadata)
        
        stats = case_database.get_statistics()
        
        assert stats.total_cases >= 1
        assert "invasive ductal carcinoma" in stats.cases_by_disease
        assert "Test Hospital" in stats.cases_by_institution
        assert stats.feature_dimensionality == 2048


class TestCounterfactualExplanations:
    """Test counterfactual explanation system"""
    
    @pytest.fixture
    def counterfactual_system(self):
        """Create counterfactual explanation system"""
        return CounterfactualExplanationSystem(
            generator_type="gradient",
            max_iterations=10,  # Reduced for testing
            learning_rate=0.1
        )
    
    @pytest.fixture
    def model(self):
        """Create test model"""
        return create_foundation_model()
    
    def test_biological_plausibility_validator(self):
        """Test biological plausibility validation"""
        validator = BiologicalPlausibilityValidator("breast")
        
        original_features = torch.randn(2048)
        modified_features = original_features + 0.1 * torch.randn(2048)
        
        validation_results = validator.validate_changes(
            original_features, modified_features
        )
        
        assert 0.0 <= validation_results['overall_plausibility'] <= 1.0
        assert 0.0 <= validation_results['morphology_score'] <= 1.0
        assert 0.0 <= validation_results['magnitude_score'] <= 1.0
        assert isinstance(validation_results['constraint_violations'], list)
        assert isinstance(validation_results['warnings'], list)
    
    def test_counterfactual_generation(self, counterfactual_system, model):
        """Test counterfactual explanation generation"""
        patches = torch.randn(1, 10, 3, 224, 224)
        
        # Generate prediction
        with torch.no_grad():
            prediction = model(patches, disease_type="breast")
        
        # Generate counterfactual
        counterfactual = counterfactual_system.generate_explanation(
            model, patches, prediction, disease_type="breast"
        )
        
        if counterfactual:  # May be None if generation fails
            assert counterfactual.original_prediction != ""
            assert counterfactual.target_prediction != ""
            assert 0.0 <= counterfactual.plausibility_score <= 1.0
            assert 0.0 <= counterfactual.success_probability <= 1.0
            assert len(counterfactual.required_changes) > 0
            assert counterfactual.natural_language != ""
    
    def test_multiple_counterfactuals(self, counterfactual_system, model):
        """Test generation of multiple counterfactuals"""
        patches = torch.randn(1, 10, 3, 224, 224)
        
        # Generate prediction
        with torch.no_grad():
            prediction = model(patches, disease_type="breast")
        
        # Generate multiple counterfactuals
        counterfactuals = counterfactual_system.generate_multiple_counterfactuals(
            model, patches, prediction, disease_type="breast", num_targets=2
        )
        
        assert len(counterfactuals) <= 2
        for cf in counterfactuals:
            assert cf.original_prediction != ""
            assert cf.target_prediction != ""


class TestExplainabilityIntegration:
    """Test full explainability system integration"""
    
    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database path"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def explainer(self, temp_db_path):
        """Create vision-language explainer"""
        return VisionLanguageExplainer(
            case_database_path=temp_db_path,
            uncertainty_method="ensemble",
            num_mc_samples=5  # Reduced for testing
        )
    
    @pytest.fixture
    def models(self):
        """Create ensemble of models"""
        model1 = create_foundation_model()
        model2 = create_foundation_model()
        return [model1, model2]
    
    def test_full_explanation_generation(self, explainer, models):
        """Test complete explanation generation pipeline"""
        patches = torch.randn(1, 20, 3, 224, 224)
        
        # Generate prediction
        with torch.no_grad():
            prediction = models[0](patches, disease_type="breast")
        
        # Generate complete explanation
        explanation = explainer.generate_explanation(
            foundation_model=models[0],
            patches=patches,
            prediction=prediction,
            disease_type="breast",
            return_counterfactual=True,
            ensemble_models=models
        )
        
        # Verify explanation components
        assert explanation.natural_language_explanation != ""
        assert explanation.uncertainty_metrics.total_uncertainty >= 0.0
        assert isinstance(explanation.similar_cases, list)
        assert isinstance(explanation.requires_second_opinion, bool)
        assert explanation.explanation_generation_time > 0.0
        
        # Feature attribution should be present
        assert "patch_attribution" in explanation.feature_attribution
        
        # Confidence intervals should be computed
        assert len(explanation.confidence_intervals) > 0
    
    def test_case_database_integration(self, explainer):
        """Test case database integration with explainer"""
        # Add a training case
        features = torch.randn(2048)
        metadata = {
            'slide_id': 'test_slide',
            'institution': 'Test Hospital',
            'scanner_type': 'Test Scanner',
            'magnification': 40.0,
            'stain_type': 'H&E',
            'tissue_type': 'breast',
            'grade': 'Grade 2',
            'stage': 'T2N1M0',
            'molecular_markers': {'ER': 'positive'},
            'pathologist_id': 'pathologist_001',
            'annotation_time': datetime.now(),
            'image_quality_score': 0.9,
            'artifact_flags': [],
            'demographics': {'age_group': '50-60'},
            'treatment_response': 'complete_response',
            'follow_up_months': 24,
            'tags': ['training']
        }
        
        success = explainer.add_training_case(
            case_id="integration_test_case",
            features=features,
            diagnosis="invasive ductal carcinoma",
            confidence=0.92,
            metadata=metadata
        )
        
        assert success
        
        # Verify case was added
        stats = explainer.case_database.get_statistics()
        assert stats.total_cases >= 1
    
    def test_uncertainty_calibration_integration(self, explainer, models):
        """Test uncertainty calibration integration"""
        # Create mock validation data
        validation_patches = [torch.randn(1, 10, 3, 224, 224) for _ in range(5)]
        validation_labels = [0, 1, 0, 1, 0]
        
        # Initialize uncertainty system
        explainer.initialize_uncertainty_system(models)
        
        # Fit calibrator
        calibration_metrics = explainer.fit_uncertainty_calibrator(
            validation_patches, validation_labels, disease_type="breast"
        )
        
        if calibration_metrics:  # May be None if not enough data
            assert calibration_metrics.expected_calibration_error >= 0.0
    
    def test_performance_requirements(self, explainer, models):
        """Test that performance requirements are met"""
        patches = torch.randn(1, 50, 3, 224, 224)  # Larger input
        
        # Generate prediction
        with torch.no_grad():
            prediction = models[0](patches, disease_type="breast")
        
        # Time explanation generation
        start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        
        if start_time:
            start_time.record()
        
        explanation = explainer.generate_explanation(
            foundation_model=models[0],
            patches=patches,
            prediction=prediction,
            disease_type="breast",
            return_counterfactual=False,  # Skip counterfactual for speed
            ensemble_models=models
        )
        
        if end_time:
            end_time.record()
            torch.cuda.synchronize()
            elapsed_time = start_time.elapsed_time(end_time) / 1000.0  # Convert to seconds
        else:
            elapsed_time = explanation.explanation_generation_time
        
        # Verify performance requirements
        assert elapsed_time < 10.0  # Should be under 10 seconds
        assert explanation.natural_language_explanation != ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])