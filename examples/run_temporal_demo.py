"""
Demo testing temporal reasoning across multiple slides.
Tests the CrossSlideTemporalReasoner for disease progression modeling.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

from src.models import MultimodalFusionModel, CrossSlideTemporalReasoner, ClassificationHead
from experiments.tracking import log_experiment

torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

os.makedirs("results/temporal_demo", exist_ok=True)


class TemporalSequenceDataset(Dataset):
    """Dataset with temporal sequences of slides showing progression."""

    def __init__(self, num_patients=100, num_classes=3):
        self.num_patients = num_patients
        self.num_classes = num_classes
        self.labels = torch.randint(0, num_classes, (num_patients,))

    def __len__(self):
        return self.num_patients

    def __getitem__(self, idx):
        label = self.labels[idx].item()

        # Generate sequence of 3-5 slides over time
        num_slides = np.random.randint(3, 6)
        slides = []
        timestamps = sorted(np.random.uniform(0, 365, num_slides))  # Days

        for t_idx, timestamp in enumerate(timestamps):
            # Features evolve over time based on label (disease progression)
            progression_factor = t_idx / num_slides  # 0 to 1

            # WSI features change over time
            num_patches = np.random.randint(30, 80)
            wsi_features = torch.randn(num_patches, 1024) + label * 1.5 + progression_factor * 0.5

            # Genomic features (relatively stable)
            genomic = torch.randn(2000) + label * 1.2

            # Clinical text
            seq_len = np.random.randint(30, 100)
            clinical_text = torch.randint(1, 30000, (seq_len,))
            clinical_text[:10] = clinical_text[:10] + label * 1500
            clinical_text = torch.clamp(clinical_text, 1, 29999)

            slides.append(
                {
                    "wsi_features": wsi_features,
                    "genomic": genomic,
                    "clinical_text": clinical_text,
                    "timestamp": timestamp,
                }
            )

        return {"slides": slides, "label": self.labels[idx]}


def collate_temporal_fn(batch):
    """Collate function for temporal sequences."""
    batch_size = len(batch)
    max_slides = max(len(item["slides"]) for item in batch)
    labels = torch.stack([item["label"] for item in batch])

    # Prepare batch structure
    batch_slides = []
    timestamps = []

    for slide_idx in range(max_slides):
        # Collect all samples' slide at this position
        slide_batch = []
        slide_timestamps = []

        for patient_idx in range(batch_size):
            if slide_idx < len(batch[patient_idx]["slides"]):
                slide_batch.append(batch[patient_idx]["slides"][slide_idx])
                slide_timestamps.append(batch[patient_idx]["slides"][slide_idx]["timestamp"])
            else:
                # Padding for shorter sequences
                slide_batch.append(None)
                slide_timestamps.append(0.0)

        # Collate this slide position across batch
        # WSI
        wsi_list = [s["wsi_features"] if s is not None else None for s in slide_batch]
        max_patches = max((wsi.shape[0] for wsi in wsi_list if wsi is not None), default=0)

        if max_patches > 0:
            wsi_padded = torch.zeros(batch_size, max_patches, 1024)
            wsi_mask = torch.zeros(batch_size, max_patches, dtype=torch.bool)
            for i, wsi in enumerate(wsi_list):
                if wsi is not None:
                    length = wsi.shape[0]
                    wsi_padded[i, :length] = wsi
                    wsi_mask[i, :length] = True
        else:
            wsi_padded = None
            wsi_mask = None

        # Genomic
        genomic = torch.zeros(batch_size, 2000)
        for i, s in enumerate(slide_batch):
            if s is not None:
                genomic[i] = s["genomic"]

        # Clinical
        clinical_list = [s["clinical_text"] if s is not None else None for s in slide_batch]
        clinical_list_valid = [c for c in clinical_list if c is not None]
        if len(clinical_list_valid) > 0:
            max_len = max(text.shape[0] for text in clinical_list_valid)
            clinical_text = torch.zeros(batch_size, max_len, dtype=torch.long)
            clinical_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
            for i, c in enumerate(clinical_list):
                if c is not None:
                    length = c.shape[0]
                    clinical_text[i, :length] = c
                    clinical_mask[i, :length] = True
        else:
            clinical_text = None
            clinical_mask = None

        batch_slides.append(
            {
                "wsi_features": wsi_padded,
                "wsi_mask": wsi_mask,
                "genomic": genomic,
                "clinical_text": clinical_text,
                "clinical_mask": clinical_mask,
            }
        )

        timestamps.append(torch.tensor(slide_timestamps))

    return {
        "slides": batch_slides,
        "timestamps": torch.stack(timestamps).T,  # [batch, num_slides]
        "label": labels,
    }


print("Creating temporal dataset...")
train_dataset = TemporalSequenceDataset(num_patients=150, num_classes=3)
test_dataset = TemporalSequenceDataset(num_patients=50, num_classes=3)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_temporal_fn)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=collate_temporal_fn)

print(f"Train: {len(train_dataset)} patients, Test: {len(test_dataset)} patients")

# Initialize models
print("\nInitializing temporal model...")
fusion_model = MultimodalFusionModel(embed_dim=128).to(device)
temporal_model = CrossSlideTemporalReasoner(embed_dim=128, num_heads=4, num_layers=2).to(device)
classifier = ClassificationHead(input_dim=128, num_classes=3).to(device)

total_params = (
    sum(p.numel() for p in fusion_model.parameters())
    + sum(p.numel() for p in temporal_model.parameters())
    + sum(p.numel() for p in classifier.parameters())
)
print(f"Total parameters: {total_params:,}")

optimizer = optim.AdamW(
    list(fusion_model.parameters())
    + list(temporal_model.parameters())
    + list(classifier.parameters()),
    lr=5e-4,
    weight_decay=0.01,
)
criterion = nn.CrossEntropyLoss()

# Training
print("\nTraining temporal model...")
history = {"train_loss": [], "train_acc": []}

for epoch in range(5):
    fusion_model.train()
    temporal_model.train()
    classifier.train()

    total_loss = 0
    correct = 0
    total = 0

    for batch in train_loader:
        labels = batch["label"].to(device)
        timestamps = batch["timestamps"].to(device)

        # Process each slide through fusion model
        slide_embeddings = []
        for slide_data in batch["slides"]:
            slide_batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in slide_data.items()
            }
            embedding = fusion_model(slide_batch)
            slide_embeddings.append(embedding)

        # Stack: [batch, num_slides, embed_dim]
        slide_embeddings = torch.stack(slide_embeddings, dim=1)

        # Temporal reasoning
        temporal_output, progression_features = temporal_model(slide_embeddings, timestamps)

        # Classification
        logits = classifier(temporal_output)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = total_loss / len(train_loader)
    train_acc = correct / total
    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)

    print(f"Epoch {epoch+1}/5 - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")

# Evaluation
print("\nEvaluating...")
fusion_model.eval()
temporal_model.eval()
classifier.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        labels = batch["label"].to(device)
        timestamps = batch["timestamps"].to(device)

        slide_embeddings = []
        for slide_data in batch["slides"]:
            slide_batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in slide_data.items()
            }
            embedding = fusion_model(slide_batch)
            slide_embeddings.append(embedding)

        slide_embeddings = torch.stack(slide_embeddings, dim=1)
        temporal_output, progression_features = temporal_model(slide_embeddings, timestamps)
        logits = classifier(temporal_output)

        _, predicted = torch.max(logits, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

from sklearn.metrics import accuracy_score

test_acc = accuracy_score(all_labels, all_preds)
print(f"Test Accuracy: {test_acc:.4f}")

# Visualize training
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history["train_loss"], marker="o", linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history["train_acc"], marker="o", linewidth=2, color="green")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training Accuracy")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("results/temporal_demo/training_curves.png", dpi=300, bbox_inches="tight")
print("✓ Saved training_curves.png")

# Create report
report = f"""
Temporal Reasoning Demo Report
{'='*60}

Architecture: Multimodal Fusion + Cross-Slide Temporal Reasoner
Model Parameters: {total_params:,}

Dataset:
- Training patients: {len(train_dataset)}
- Test patients: {len(test_dataset)}
- Slides per patient: 3-5 (variable)
- Temporal span: 0-365 days

Results:
{'-'*60}
Final Training Accuracy: {history['train_acc'][-1]:.2%}
Test Accuracy: {test_acc:.2%}
{'-'*60}

Key Features Demonstrated:
1. Temporal attention across multiple slides
2. Positional encoding for temporal distances
3. Progression feature extraction
4. Variable-length sequence handling

This demonstrates the architecture's ability to model disease
progression and temporal patterns across longitudinal patient data.
"""

with open("results/temporal_demo/report.txt", "w") as f:
    f.write(report)

print("\n" + "=" * 60)
print("TEMPORAL REASONING DEMO COMPLETE!")
print("=" * 60)
print(f"Test Accuracy: {test_acc:.2%}")
print("Results saved to: results/temporal_demo/")

# Log experiment for tracking
config = {
    "num_train_patients": len(train_dataset),
    "num_test_patients": len(test_dataset),
    "num_classes": 3,
    "embed_dim": 128,
    "num_heads": 4,
    "num_layers": 2,
    "epochs": 5,
    "slides_per_patient": "3-5 (variable)",
    "temporal_span_days": 365,
}
metrics = {
    "test_accuracy": float(test_acc),
    "final_train_accuracy": float(history["train_acc"][-1]),
    "final_train_loss": float(history["train_loss"][-1]),
}
exp_path = log_experiment("temporal_demo", config, metrics)
print(f"\nExperiment logged to: {exp_path}")
print("=" * 60)
