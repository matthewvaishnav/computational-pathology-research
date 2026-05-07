"""
End-to-end integration tests for the complete system.

Tests the entire pipeline from data loading through training to inference.
"""

import json
import shutil
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from src.data import MultimodalDataset
from src.models import ClassificationHead, MultimodalFusionModel


@pytest.fixture
def temp_workspace():
    """Create temporary workspace for integration tests."""
    tmpdir = tempfile.mkdtemp()
    workspace = Path(tmpdir)

    # Create directory structure
    (workspace / "data").mkdir()
    (workspace / "data" / "wsi_features").mkdir()
    (workspace / "data" / "genomic").mkdir()
    (workspace / "data" / "clinical_text").mkdir()
    (workspace / "checkpoints").mkdir()
    (workspace / "logs").mkdir()

    # Create mock data files
    for split in ["train", "val", "test"]:
        # Create WSI features
        for i in range(5):
            wsi_file = workspace / "data" / "wsi_features" / f"{split}_patient_{i}_wsi.h5"
            with h5py.File(wsi_file, "w") as f:
                f.create_dataset("features", data=np.random.randn(100, 1024).astype(np.float32))

        # Create genomic features
        for i in range(5):
            genomic_file = workspace / "data" / "genomic" / f"{split}_patient_{i}_genomic.npy"
            np.save(genomic_file, np.random.randn(2000).astype(np.float32))

        # Create clinical text
        for i in range(5):
            clinical_file = (
                workspace / "data" / "clinical_text" / f"{split}_patient_{i}_clinical.npy"
            )
            np.save(clinical_file, np.random.randint(0, 1000, size=50).astype(np.int64))

        # Create metadata
        metadata = {
            "samples": [
                {
                    "patient_id": f"{split}_patient_{i}",
                    "wsi_file": f"{split}_patient_{i}_wsi.h5",
                    "genomic_file": f"{split}_patient_{i}_genomic.npy",
                    "clinical_file": f"{split}_patient_{i}_clinical.npy",
                    "label": i % 4,
                    "timestamp": 1234567890.0 + i,
                }
                for i in range(5)
            ]
        }

        with open(workspace / "data" / f"{split}_metadata.json", "w") as f:
            json.dump(metadata, f)

    yield workspace

    # Cleanup
    shutil.rmtree(tmpdir)


def test_complete_training_workflow(temp_workspace):
    """Test complete training workflow from data loading to model saving."""
    # Configuration
    config = {
        "wsi_enabled": True,
        "genomic_enabled": True,
        "clinical_text_enabled": True,
        "max_text_length": 100,
        "batch_size": 2,
        "num_epochs": 2,
        "learning_rate": 1e-3,
    }

    # Load data
    train_dataset = MultimodalDataset(
        data_dir=temp_workspace / "data", split="train", config=config
    )

    val_dataset = MultimodalDataset(data_dir=temp_workspace / "data", split="val", config=config)

    assert len(train_dataset) == 5
    assert len(val_dataset) == 5

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, drop_last=True)

    # Initialize model
    model = MultimodalFusionModel(embed_dim=128)
    classifier = ClassificationHead(input_dim=128, num_classes=4)

    # Setup training
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(classifier.parameters()), lr=config["learning_rate"]
    )
    criterion = torch.nn.CrossEntropyLoss()

    # Training loop
    model.train()
    classifier.train()

    for epoch in range(config["num_epochs"]):
        total_loss = 0
        for batch in train_loader:
            # Skip label
            labels = batch.pop("label")

            # Forward pass
            embeddings = model(batch)
            logits = classifier(embeddings)
            loss = criterion(logits, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        assert avg_loss > 0, "Loss should be positive"

    # Validation
    model.eval()
    classifier.eval()

    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in val_loader:
            labels = batch.pop("label")

            embeddings = model(batch)
            logits = classifier(embeddings)
            loss = criterion(logits, labels)

            val_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    assert 0 <= accuracy <= 1, "Accuracy should be between 0 and 1"

    # Save checkpoint
    checkpoint_path = temp_workspace / "checkpoints" / "test_model.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "task_head_state_dict": classifier.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": config["num_epochs"],
            "config": config,
        },
        checkpoint_path,
    )

    assert checkpoint_path.exists(), "Checkpoint should be saved"

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path)
    assert "model_state_dict" in checkpoint
    assert "task_head_state_dict" in checkpoint


def test_missing_modality_handling(temp_workspace):
    """Test system handles missing modalities gracefully."""
    config = {
        "wsi_enabled": True,
        "genomic_enabled": True,
        "clinical_text_enabled": True,
        "max_text_length": 100,
    }

    # Load dataset
    dataset = MultimodalDataset(data_dir=temp_workspace / "data", split="train", config=config)

    # Initialize model
    model = MultimodalFusionModel(embed_dim=128)
    classifier = ClassificationHead(input_dim=128, num_classes=4)

    model.eval()
    classifier.eval()

    def add_batch_dim(sample):
        """Add batch dimension to sample tensors."""
        batched = {}
        for key, value in sample.items():
            if value is not None and isinstance(value, torch.Tensor):
                batched[key] = value.unsqueeze(0)
            else:
                batched[key] = value
        return batched

    # Test with all modalities
    sample = dataset[0]
    sample.pop("label")
    sample_batched = add_batch_dim(sample)

    with torch.no_grad():
        embeddings = model(sample_batched)
        logits = classifier(embeddings)

    assert embeddings.shape == (1, 128), "Should produce batched embedding"
    assert logits.shape == (1, 4), "Should produce batched logits"

    # Test with missing WSI
    sample_no_wsi = dataset[0]
    sample_no_wsi["wsi_features"] = None
    sample_no_wsi.pop("label")
    sample_no_wsi_batched = add_batch_dim(sample_no_wsi)

    with torch.no_grad():
        embeddings = model(sample_no_wsi_batched)
        logits = classifier(embeddings)

    assert embeddings.shape == (1, 128), "Should handle missing WSI"

    # Test with missing genomic
    sample_no_genomic = dataset[0]
    sample_no_genomic["genomic"] = None
    sample_no_genomic.pop("label")
    sample_no_genomic_batched = add_batch_dim(sample_no_genomic)

    with torch.no_grad():
        embeddings = model(sample_no_genomic_batched)
        logits = classifier(embeddings)

    assert embeddings.shape == (1, 128), "Should handle missing genomic"

    # Test with only one modality
    sample_wsi_only = dataset[0]
    sample_wsi_only["genomic"] = None
    sample_wsi_only["clinical_text"] = None
    sample_wsi_only.pop("label")
    sample_wsi_only_batched = add_batch_dim(sample_wsi_only)

    with torch.no_grad():
        embeddings = model(sample_wsi_only_batched)
        logits = classifier(embeddings)

    assert embeddings.shape == (1, 128), "Should handle single modality"


def test_all_modalities_missing(temp_workspace):
    """Test model raises error when all modalities are missing."""
    # Initialize model
    model = MultimodalFusionModel(embed_dim=128)
    model.eval()

    # Create batch with all modalities as None
    batch = {
        "wsi_features": None,
        "genomic": None,
        "clinical_text": None,
    }

    # Should raise ValueError because no modalities available to determine batch size
    with pytest.raises(ValueError, match="At least one modality must be provided"):
        model(batch)


def test_inference_pipeline(temp_workspace):
    """Test complete inference pipeline."""
    config = {
        "wsi_enabled": True,
        "genomic_enabled": True,
        "clinical_text_enabled": True,
        "max_text_length": 100,
    }

    # Load test data
    test_dataset = MultimodalDataset(data_dir=temp_workspace / "data", split="test", config=config)

    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

    # Initialize model
    model = MultimodalFusionModel(embed_dim=128)
    classifier = ClassificationHead(input_dim=128, num_classes=4)

    model.eval()
    classifier.eval()

    # Run inference
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            labels = batch.pop("label")

            embeddings = model(batch)
            logits = classifier(embeddings)
            predictions = torch.argmax(logits, dim=1)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    assert len(all_predictions) == len(test_dataset)
    assert len(all_labels) == len(test_dataset)
    assert all(0 <= p < 4 for p in all_predictions), "Predictions should be valid class indices"


def test_data_validation(temp_workspace):
    """Test data validation catches issues."""
    config = {
        "wsi_enabled": True,
        "genomic_enabled": True,
        "clinical_text_enabled": True,
        "max_text_length": 100,
    }

    # Load dataset
    dataset = MultimodalDataset(data_dir=temp_workspace / "data", split="train", config=config)

    # Check all samples load correctly
    for i in range(len(dataset)):
        sample = dataset[i]

        # Validate structure
        assert "patient_id" in sample
        assert "label" in sample
        assert "wsi_features" in sample
        assert "genomic" in sample
        assert "clinical_text" in sample

        # Validate types
        if sample["wsi_features"] is not None:
            assert isinstance(sample["wsi_features"], torch.Tensor)

        if sample["genomic"] is not None:
            assert isinstance(sample["genomic"], torch.Tensor)

        if sample["clinical_text"] is not None:
            assert isinstance(sample["clinical_text"], torch.Tensor)

        assert isinstance(sample["label"], torch.Tensor)


def test_model_serialization(temp_workspace):
    """Test model can be saved and loaded correctly."""
    # Initialize model
    model = MultimodalFusionModel(embed_dim=128)
    classifier = ClassificationHead(input_dim=128, num_classes=4)

    # Save model
    checkpoint_path = temp_workspace / "checkpoints" / "serialization_test.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "task_head_state_dict": classifier.state_dict(),
            "config": {"embed_dim": 128, "num_classes": 4},
        },
        checkpoint_path,
    )

    # Load model
    checkpoint = torch.load(checkpoint_path)

    new_model = MultimodalFusionModel(embed_dim=128)
    new_classifier = ClassificationHead(input_dim=128, num_classes=4)

    new_model.load_state_dict(checkpoint["model_state_dict"])
    new_classifier.load_state_dict(checkpoint["task_head_state_dict"])

    # Verify models produce same output
    dummy_input = {
        "wsi_features": torch.randn(1, 100, 1024),
        "genomic": torch.randn(1, 2000),
        "clinical_text": torch.randint(0, 1000, (1, 50)),
    }

    model.eval()
    new_model.eval()
    classifier.eval()
    new_classifier.eval()

    with torch.no_grad():
        emb1 = model(dummy_input)
        emb2 = new_model(dummy_input)

        logits1 = classifier(emb1)
        logits2 = new_classifier(emb2)

    assert torch.allclose(emb1, emb2, atol=1e-6), "Embeddings should match"
    assert torch.allclose(logits1, logits2, atol=1e-6), "Logits should match"


def test_gradient_flow(temp_workspace):
    """Test gradients flow through entire model."""
    # Initialize model
    model = MultimodalFusionModel(embed_dim=128)
    classifier = ClassificationHead(input_dim=128, num_classes=4)

    model.train()
    classifier.train()

    # Create dummy batch
    batch = {
        "wsi_features": torch.randn(2, 100, 1024, requires_grad=True),
        "genomic": torch.randn(2, 2000, requires_grad=True),
        "clinical_text": torch.randint(0, 1000, (2, 50)),
    }
    labels = torch.tensor([0, 1])

    # Forward pass
    embeddings = model(batch)
    logits = classifier(embeddings)

    # Compute loss
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(logits, labels)

    # Backward pass
    loss.backward()

    # Check gradients exist
    assert batch["wsi_features"].grad is not None, "WSI gradients should exist"
    assert batch["genomic"].grad is not None, "Genomic gradients should exist"

    # Check model gradients
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradient for {name} should exist"

    for name, param in classifier.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradient for {name} should exist"


def test_batch_processing(temp_workspace):
    """Test model handles different batch sizes correctly."""
    model = MultimodalFusionModel(embed_dim=128)
    classifier = ClassificationHead(input_dim=128, num_classes=4)

    model.eval()
    classifier.eval()

    batch_sizes = [1, 2, 4, 8]

    for batch_size in batch_sizes:
        batch = {
            "wsi_features": torch.randn(batch_size, 100, 1024),
            "genomic": torch.randn(batch_size, 2000),
            "clinical_text": torch.randint(0, 1000, (batch_size, 50)),
        }

        with torch.no_grad():
            embeddings = model(batch)
            logits = classifier(embeddings)

        assert embeddings.shape == (
            batch_size,
            128,
        ), f"Embedding shape incorrect for batch size {batch_size}"
        assert logits.shape == (
            batch_size,
            4,
        ), f"Logits shape incorrect for batch size {batch_size}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
