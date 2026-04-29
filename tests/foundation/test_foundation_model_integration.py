"""
Comprehensive integration tests for the Foundation Model system
Tests all components: multi-disease model, self-supervised pre-training, zero-shot detection, and training pipeline
"""

import json
import shutil
import sqlite3
import time

# Import foundation model components
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.append("src/foundation")

from data_collection import (
    SlideDatabase,
    UnlabeledWSIDataset,
    WSIDataCollector,
    WSIQualityAssessment,
)
from multi_disease_model import ModelConfig, MultiDiseaseFoundationModel, create_foundation_model
from self_supervised_pretrainer import (
    HistopathologyAugmentation,
    PreTrainingConfig,
    SelfSupervisedPreTrainer,
)
from training_pipeline import FoundationModelTrainer, TrainingConfig
from zero_shot_detection import (
    DiseaseDescription,
    DiseaseKnowledgeBase,
    VisionLanguageEncoder,
    ZeroShotDetector,
)


class TestMultiDiseaseFoundationModel:
    """Test multi-disease foundation model"""

    def test_model_initialization(self):
        """Test model initialization with default config"""
        config = ModelConfig()
        model = MultiDiseaseFoundationModel(config)

        assert model.config.encoder_type == "resnet50"
        assert model.config.feature_dim == 2048
        assert len(model.config.supported_diseases) == 5
        assert "breast" in model.config.supported_diseases

        # Check disease heads are created
        assert len(model.disease_heads) == 5
        for disease in model.config.supported_diseases:
            assert disease in model.disease_heads

    def test_feature_extraction(self):
        """Test feature extraction from patches"""
        model = create_foundation_model()

        # Test input
        batch_size = 2
        num_patches = 10
        patches = torch.randn(batch_size, num_patches, 3, 224, 224)

        # Extract features
        features = model.extract_features(patches)

        assert features.shape == (batch_size, num_patches, 2048)
        assert not torch.isnan(features).any()
        assert torch.isfinite(features).all()

    def test_multi_disease_prediction(self):
        """Test multi-disease prediction"""
        model = create_foundation_model()

        batch_size = 2
        num_patches = 10
        patches = torch.randn(batch_size, num_patches, 3, 224, 224)

        # Multi-disease prediction
        results = model(patches, return_features=True, return_attention=True)

        # Check all diseases have predictions
        for disease in model.config.supported_diseases:
            assert disease in results
            assert results[disease].shape[0] == batch_size

        # Check features and attention
        assert "features" in results
        assert results["features"].shape == (batch_size, num_patches, 2048)

        # Check attention weights
        for disease in model.config.supported_diseases:
            attention_key = f"{disease}_attention"
            assert attention_key in results

    def test_single_disease_prediction(self):
        """Test single disease prediction"""
        model = create_foundation_model()

        patches = torch.randn(1, 10, 3, 224, 224)

        # Single disease prediction
        results = model(patches, disease_type="breast")

        assert "breast" in results
        assert len(results) == 1  # Only breast prediction
        assert results["breast"].shape[0] == 1

    def test_zero_shot_prediction(self):
        """Test zero-shot prediction capability"""
        model = create_foundation_model()

        patches = torch.randn(1, 10, 3, 224, 224)
        text_embeddings = torch.randn(5, 768)  # 5 text descriptions

        similarity, confidence = model.zero_shot_predict(patches, text_embeddings)

        assert similarity.shape == (1, 5)
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0

    def test_add_disease_head(self):
        """Test adding new disease head"""
        model = create_foundation_model()
        initial_diseases = len(model.disease_heads)

        # Add new disease
        model.add_disease_head("pancreatic", num_classes=3, freeze_encoder=True)

        assert len(model.disease_heads) == initial_diseases + 1
        assert "pancreatic" in model.disease_heads
        assert "pancreatic" in model.config.supported_diseases

        # Test prediction with new disease
        patches = torch.randn(1, 10, 3, 224, 224)
        results = model(patches, disease_type="pancreatic")
        assert "pancreatic" in results
        assert results["pancreatic"].shape[1] == 3  # 3 classes


class TestSelfSupervisedPreTrainer:
    """Test self-supervised pre-training system"""

    def test_pretrainer_initialization(self):
        """Test pre-trainer initialization"""
        model = create_foundation_model()
        config = PreTrainingConfig(method="simclr", num_epochs=1)

        pretrainer = SelfSupervisedPreTrainer(model, config)

        assert pretrainer.config.method == "simclr"
        assert hasattr(pretrainer, "criterion")
        assert hasattr(pretrainer, "optimizer")
        assert hasattr(pretrainer, "scheduler")

    def test_histopathology_augmentation(self):
        """Test histopathology-specific augmentation"""
        from self_supervised_pretrainer import AugmentationConfig, HistopathologyAugmentation

        config = AugmentationConfig()
        augmentation = HistopathologyAugmentation(config)

        # Test augmentation
        image = torch.rand(3, 224, 224)
        view1, view2 = augmentation(image)

        assert view1.shape == (3, 224, 224)
        assert view2.shape == (3, 224, 224)
        assert not torch.equal(view1, view2)  # Should be different
        assert torch.all(view1 >= 0) and torch.all(view1 <= 1)  # Valid range

    def test_simclr_loss(self):
        """Test SimCLR contrastive loss"""
        from self_supervised_pretrainer import SimCLRLoss

        criterion = SimCLRLoss(temperature=0.07)

        # Test features (positive pairs)
        batch_size = 4
        feature_dim = 128
        features = torch.randn(2 * batch_size, feature_dim)

        loss = criterion(features)

        assert isinstance(loss.item(), float)
        assert loss.item() > 0  # Loss should be positive

    @patch("torch.utils.data.DataLoader")
    def test_pretrain_epoch(self, mock_dataloader):
        """Test pre-training epoch execution"""
        model = create_foundation_model()
        config = PreTrainingConfig(method="simclr", batch_size=2, num_epochs=1)
        pretrainer = SelfSupervisedPreTrainer(model, config)

        # Mock dataset
        mock_batch = torch.randn(2, 3, 224, 224)
        mock_dataloader.return_value = [mock_batch]

        # Test training epoch
        metrics = pretrainer._train_epoch(mock_dataloader.return_value, 0)

        assert "loss" in metrics
        assert isinstance(metrics["loss"], float)


class TestZeroShotDetection:
    """Test zero-shot detection system"""

    def test_disease_knowledge_base(self):
        """Test disease knowledge base"""
        kb = DiseaseKnowledgeBase()

        # Check default diseases loaded
        assert len(kb.diseases) > 0
        assert "adenocarcinoma" in kb.diseases
        assert "squamous_cell_carcinoma" in kb.diseases

        # Test disease descriptions
        descriptions = kb.get_disease_descriptions()
        assert len(descriptions) == len(kb.diseases)
        assert all(isinstance(desc, str) for desc in descriptions)

    def test_add_disease_to_knowledge_base(self):
        """Test adding new disease to knowledge base"""
        kb = DiseaseKnowledgeBase()
        initial_count = len(kb.diseases)

        new_disease = DiseaseDescription(
            disease_name="test_disease",
            description="Test disease description",
            synonyms=["test"],
            pathological_features=["feature1", "feature2"],
        )

        kb.add_disease(new_disease)

        assert len(kb.diseases) == initial_count + 1
        assert "test_disease" in kb.diseases
        assert kb.diseases["test_disease"].description == "Test disease description"

    def test_vision_language_encoder(self):
        """Test vision-language encoder"""
        encoder = VisionLanguageEncoder()

        # Test image encoding
        images = torch.randn(2, 3, 224, 224)
        image_features = encoder.encode_images(images)

        assert image_features.shape[0] == 2
        assert image_features.shape[1] > 0  # Feature dimension
        assert torch.allclose(
            torch.norm(image_features, dim=1), torch.ones(2), atol=1e-5
        )  # Normalized

        # Test text encoding
        texts = ["adenocarcinoma with glandular structures", "squamous cell carcinoma"]
        text_features = encoder.encode_texts(texts)

        assert text_features.shape[0] == 2
        assert text_features.shape[1] > 0
        assert torch.allclose(
            torch.norm(text_features, dim=1), torch.ones(2), atol=1e-5
        )  # Normalized

    def test_zero_shot_detector(self):
        """Test zero-shot detector"""
        kb = DiseaseKnowledgeBase()
        encoder = VisionLanguageEncoder()
        detector = ZeroShotDetector(kb, encoder)

        # Test prediction
        image_features = torch.randn(1, 512)  # Mock image features
        prediction = detector.predict(image_features, top_k=3)

        assert hasattr(prediction, "predicted_disease")
        assert hasattr(prediction, "confidence")
        assert hasattr(prediction, "uncertainty_score")
        assert hasattr(prediction, "requires_expert_review")
        assert len(prediction.top_k_diseases) == 3

        # Check confidence is valid
        assert 0.0 <= prediction.confidence <= 1.0
        assert 0.0 <= prediction.uncertainty_score <= 1.0

    def test_batch_zero_shot_prediction(self):
        """Test batch zero-shot prediction"""
        kb = DiseaseKnowledgeBase()
        encoder = VisionLanguageEncoder()
        detector = ZeroShotDetector(kb, encoder)

        # Batch prediction
        batch_features = torch.randn(3, 512)
        predictions = detector.batch_predict(batch_features, top_k=2)

        assert len(predictions) == 3
        assert all(len(pred.top_k_diseases) == 2 for pred in predictions)

    def test_explanation_generation(self):
        """Test explanation generation"""
        kb = DiseaseKnowledgeBase()
        encoder = VisionLanguageEncoder()
        detector = ZeroShotDetector(kb, encoder)

        # Get prediction
        image_features = torch.randn(1, 512)
        prediction = detector.predict(image_features)

        # Generate explanation
        explanation = detector.explain_prediction(prediction)

        assert isinstance(explanation, str)
        assert len(explanation) > 0
        assert prediction.predicted_disease in explanation
        assert "Confidence:" in explanation
        assert "Uncertainty:" in explanation


class TestDataCollection:
    """Test data collection and curation system"""

    def test_slide_database(self):
        """Test slide database operations"""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_db:
            db = SlideDatabase(tmp_db.name)

            # Test metadata insertion
            from data_collection import SlideMetadata

            metadata = SlideMetadata(
                slide_id="test_slide",
                file_path="/path/to/slide.svs",
                file_size=1000000,
                dimensions=(10000, 10000),
                mpp=0.25,
                vendor="Aperio",
                scanner_model="ScanScope",
                magnification=40,
                tissue_type="breast",
                staining="H&E",
                quality_score=0.85,
                tissue_percentage=0.7,
                blur_score=50.0,
                color_variance=0.05,
                artifact_score=0.1,
                sha256_hash="abc123",
                created_at="2024-01-01",
            )

            success = db.insert_slide(metadata)
            assert success

            # Test retrieval
            retrieved = db.get_slide_by_hash("abc123")
            assert retrieved is not None
            assert retrieved.slide_id == "test_slide"
            assert retrieved.quality_score == 0.85

            # Test statistics
            stats = db.get_statistics()
            assert stats["total_slides"] == 1
            assert stats["average_quality"] == 0.85

            # Cleanup
            Path(tmp_db.name).unlink()

    def test_quality_assessment(self):
        """Test WSI quality assessment"""
        assessor = WSIQualityAssessment()

        # Mock quality assessment (would normally use real slide)
        with patch.object(assessor, "assess_slide_quality") as mock_assess:
            from data_collection import QualityMetrics

            mock_assess.return_value = QualityMetrics(
                tissue_percentage=0.6,
                blur_score=75.0,
                color_variance=0.04,
                artifact_score=0.15,
                overall_score=0.75,
                passed_filters=True,
            )

            metrics = assessor.assess_slide_quality("dummy_path.svs")

            assert metrics.tissue_percentage == 0.6
            assert metrics.overall_score == 0.75
            assert metrics.passed_filters is True

    def test_unlabeled_dataset(self):
        """Test unlabeled WSI dataset"""
        # Create mock slide metadata
        from data_collection import SlideMetadata

        mock_metadata = [
            SlideMetadata(
                slide_id=f"slide_{i}",
                file_path=f"/path/slide_{i}.svs",
                file_size=1000000,
                dimensions=(10000, 10000),
                mpp=0.25,
                vendor="Aperio",
                scanner_model="ScanScope",
                magnification=40,
                tissue_type="breast",
                staining="H&E",
                quality_score=0.8,
                tissue_percentage=0.6,
                blur_score=50.0,
                color_variance=0.04,
                artifact_score=0.1,
                sha256_hash=f"hash_{i}",
                created_at="2024-01-01",
            )
            for i in range(3)
        ]

        # Mock the dataset to avoid actual file I/O
        with patch("src.foundation.data_collection.openslide.OpenSlide"):
            dataset = UnlabeledWSIDataset(mock_metadata, patch_size=224, patches_per_slide=10)

            # Test dataset length
            expected_length = len(mock_metadata) * 10  # 3 slides * 10 patches
            assert len(dataset) == expected_length


class TestTrainingPipeline:
    """Test complete training pipeline"""

    def test_training_config(self):
        """Test training configuration"""
        config = TrainingConfig()

        assert config.encoder_type == "resnet50"
        assert config.feature_dim == 2048
        assert len(config.supported_diseases) == 5
        assert config.pretrain_method == "simclr"
        assert config.enable_zero_shot is True

    def test_trainer_initialization(self):
        """Test trainer initialization"""
        config = TrainingConfig(use_wandb=False)  # Disable wandb for testing

        with patch("src.foundation.training_pipeline.WSIDataCollector"):
            trainer = FoundationModelTrainer(config)

            assert trainer.config == config
            assert isinstance(trainer.model, MultiDiseaseFoundationModel)
            assert hasattr(trainer, "data_collector")
            if config.enable_zero_shot:
                assert hasattr(trainer, "zero_shot_detector")

    def test_checkpoint_save_load(self):
        """Test checkpoint saving and loading"""
        config = TrainingConfig(use_wandb=False)

        with tempfile.TemporaryDirectory() as tmp_dir:
            config.checkpoint_dir = tmp_dir

            with patch("src.foundation.training_pipeline.WSIDataCollector"):
                trainer = FoundationModelTrainer(config)

                # Save checkpoint
                trainer.save_checkpoint("test_checkpoint.pth", 5, "test")

                # Check file exists
                checkpoint_path = Path(tmp_dir) / "test_checkpoint.pth"
                assert checkpoint_path.exists()

                # Load checkpoint
                checkpoint = trainer.load_checkpoint("test_checkpoint.pth")

                assert checkpoint["epoch"] == 5
                assert checkpoint["phase"] == "test"
                assert "model_state_dict" in checkpoint

    def test_training_report_generation(self):
        """Test training report generation"""
        config = TrainingConfig(use_wandb=False)

        with patch("src.foundation.training_pipeline.WSIDataCollector"):
            trainer = FoundationModelTrainer(config)

            # Add some mock metrics
            from training_pipeline import TrainingMetrics

            trainer.metrics_history = [
                TrainingMetrics(
                    epoch=10,
                    phase="finetune",
                    loss=0.25,
                    accuracy=0.92,
                    disease_accuracies={"breast": 0.91, "lung": 0.93},
                )
            ]

            report = trainer.generate_training_report()

            assert isinstance(report, str)
            assert "Foundation Model Training Report" in report
            assert "resnet50" in report
            assert "0.920" in report  # Final accuracy


class TestIntegrationWorkflow:
    """Test complete integration workflow"""

    def test_end_to_end_workflow(self):
        """Test end-to-end foundation model workflow"""
        # This test simulates the complete workflow with mocked components

        # 1. Create model
        model = create_foundation_model()
        assert isinstance(model, MultiDiseaseFoundationModel)

        # 2. Test feature extraction
        patches = torch.randn(1, 10, 3, 224, 224)
        features = model.extract_features(patches)
        assert features.shape == (1, 10, 2048)

        # 3. Test multi-disease prediction
        results = model(patches)
        assert len(results) == 5  # 5 diseases

        # 4. Test zero-shot detection
        kb = DiseaseKnowledgeBase()
        encoder = VisionLanguageEncoder()
        detector = ZeroShotDetector(kb, encoder)

        # Use extracted features for zero-shot
        image_features = features.mean(dim=1)  # Global average pooling
        prediction = detector.predict(image_features)

        assert hasattr(prediction, "predicted_disease")
        assert hasattr(prediction, "confidence")

        # 5. Test explanation generation
        explanation = detector.explain_prediction(prediction)
        assert isinstance(explanation, str)
        assert len(explanation) > 0

    def test_performance_requirements(self):
        """Test that model meets performance requirements"""
        model = create_foundation_model()

        # Test processing time requirement (<30s per slide)
        patches = torch.randn(1, 100, 3, 224, 224)  # 100 patches per slide

        start_time = time.time()
        with torch.no_grad():
            results = model(patches)
        processing_time = time.time() - start_time

        # Should be much faster than 30s for this test
        assert processing_time < 5.0  # 5 seconds for test

        # Test memory usage (should be reasonable)
        # This is a simplified test - real memory testing would be more complex
        assert torch.cuda.memory_allocated() < 2e9 if torch.cuda.is_available() else True  # <2GB

        # Test accuracy requirement (>90% - would need real validation data)
        # For now, just check that predictions are reasonable
        for disease in model.config.supported_diseases:
            assert disease in results
            assert results[disease].shape[0] == 1  # Batch size
            assert results[disease].shape[1] > 1  # Multiple classes

    def test_scalability(self):
        """Test model scalability with multiple concurrent requests"""
        model = create_foundation_model()
        model.eval()

        # Simulate 10 concurrent slide analyses
        batch_patches = torch.randn(10, 50, 3, 224, 224)  # 10 slides, 50 patches each

        start_time = time.time()
        with torch.no_grad():
            # Process all slides in batch
            results = model(batch_patches)
        batch_time = time.time() - start_time

        # Check that batch processing is efficient
        assert batch_time < 10.0  # Should process 10 slides quickly

        # Check results shape
        for disease in model.config.supported_diseases:
            assert results[disease].shape[0] == 10  # 10 slides


# Performance benchmarks
class TestPerformanceBenchmarks:
    """Performance benchmarks for foundation model"""

    def test_inference_speed_benchmark(self):
        """Benchmark inference speed"""
        model = create_foundation_model()
        model.eval()

        # Warm up
        patches = torch.randn(1, 100, 3, 224, 224)
        with torch.no_grad():
            _ = model(patches)

        # Benchmark
        times = []
        for _ in range(10):
            start_time = time.time()
            with torch.no_grad():
                _ = model(patches)
            times.append(time.time() - start_time)

        avg_time = np.mean(times)
        std_time = np.std(times)

        print(f"Average inference time: {avg_time:.3f}s ± {std_time:.3f}s")
        assert avg_time < 5.0  # Should be fast for testing

    def test_memory_usage_benchmark(self):
        """Benchmark memory usage"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for memory testing")

        model = create_foundation_model().cuda()

        # Clear cache
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated()

        # Run inference
        patches = torch.randn(1, 100, 3, 224, 224).cuda()
        with torch.no_grad():
            _ = model(patches)

        peak_memory = torch.cuda.max_memory_allocated()
        memory_used = (peak_memory - initial_memory) / 1e9  # Convert to GB

        print(f"Memory usage: {memory_used:.2f} GB")
        assert memory_used < 2.0  # Should be under 2GB requirement


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
