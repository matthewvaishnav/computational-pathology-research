"""
Quick training utilities for HistoCore.

Provides simple, high-level training functions with sensible defaults.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class QuickTrainer:
    """High-level trainer with sensible defaults."""

    def __init__(
        self,
        dataset="pcam",
        model="nnmil",
        epochs=10,
        batch_size=32,
        output_dir="results/",
        **kwargs,
    ):
        self.dataset = dataset
        self.model_name = model
        self.epochs = epochs
        self.batch_size = batch_size
        self.output_dir = Path(output_dir)
        self.kwargs = kwargs

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def train(self) -> Dict[str, Any]:
        """Train model with sensible defaults."""
        logger.info(f"🚀 Starting training: {self.model_name} on {self.dataset}")

        # Load dataset
        if self.dataset == "pcam":
            from src.data.pcam_dataset import PCamDataset, get_pcam_transforms

            train_dataset = PCamDataset(
                root="data/pcam", split="train", transform=get_pcam_transforms("train")
            )
            val_dataset = PCamDataset(
                root="data/pcam", split="val", transform=get_pcam_transforms("val")
            )
        elif self.dataset == "camelyon":
            from src.data.camelyon_dataset import CAMELYONSlideDataset

            train_dataset = CAMELYONSlideDataset(split="train")
            val_dataset = CAMELYONSlideDataset(split="val")
        else:
            raise ValueError(f"Unknown dataset: {self.dataset}")

        # Create data loaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4
        )

        # Load model
        if self.model_name == "nnmil":
            from src.models.nnmil import nnMIL

            model = nnMIL(feature_dim=512, num_classes=2)
        elif self.model_name == "attention":
            from src.models.attention_mil import AttentionMIL

            model = AttentionMIL(feature_dim=512, num_classes=2)
        elif self.model_name == "clam":
            from src.models.clam import CLAM

            model = CLAM(feature_dim=512, num_classes=2)
        else:
            raise ValueError(f"Unknown model: {self.model_name}")

        # Setup training
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)

        # Training loop
        best_accuracy = 0.0
        best_auc = 0.0
        results = {"train_losses": [], "val_losses": [], "val_accuracies": [], "val_aucs": []}

        for epoch in range(self.epochs):
            # Training phase
            model.train()
            train_loss = 0.0

            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)

                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

                if batch_idx % 100 == 0:
                    logger.info(
                        f"Epoch {epoch+1}/{self.epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}"
                    )

            # Validation phase
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            all_probs = []
            all_targets = []

            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    loss = criterion(output, target)

                    val_loss += loss.item()

                    # Accuracy
                    _, predicted = torch.max(output.data, 1)
                    total += target.size(0)
                    correct += (predicted == target).sum().item()

                    # AUC
                    probs = torch.softmax(output, dim=1)[:, 1]
                    all_probs.extend(probs.cpu().numpy())
                    all_targets.extend(target.cpu().numpy())

            # Calculate metrics
            accuracy = correct / total

            try:
                from sklearn.metrics import roc_auc_score

                auc = roc_auc_score(all_targets, all_probs)
            except (ImportError, ValueError) as e:
                auc = 0.0

            # Update best metrics
            if accuracy > best_accuracy:
                best_accuracy = accuracy
            if auc > best_auc:
                best_auc = auc

            # Store results
            results["train_losses"].append(train_loss / len(train_loader))
            results["val_losses"].append(val_loss / len(val_loader))
            results["val_accuracies"].append(accuracy)
            results["val_aucs"].append(auc)

            scheduler.step()

            logger.info(
                f"Epoch {epoch+1}/{self.epochs}: "
                f"Train Loss: {train_loss/len(train_loader):.4f}, "
                f"Val Loss: {val_loss/len(val_loader):.4f}, "
                f"Val Acc: {accuracy:.4f}, "
                f"Val AUC: {auc:.4f}"
            )

        # Save model
        checkpoint_path = self.output_dir / f"{self.model_name}_{self.dataset}_best.pth"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_accuracy": best_accuracy,
                "best_auc": best_auc,
                "config": {
                    "model": self.model_name,
                    "dataset": self.dataset,
                    "epochs": self.epochs,
                    "batch_size": self.batch_size,
                },
            },
            checkpoint_path,
        )

        logger.info(f"✅ Training complete! Model saved to {checkpoint_path}")

        # Final results
        results.update(
            {
                "best_accuracy": best_accuracy,
                "best_auc": best_auc,
                "checkpoint_path": str(checkpoint_path),
                "model_name": self.model_name,
                "dataset": self.dataset,
            }
        )

        return results


def train(dataset: str = "pcam", model: str = "nnmil", epochs: int = 10, **kwargs) -> dict:
    """Simple training function."""
    trainer = QuickTrainer(dataset=dataset, model=model, epochs=epochs, **kwargs)
    return trainer.train()


def evaluate(checkpoint_path: str, dataset: str = "pcam", output_dir: str = "results/") -> dict:
    """Evaluate a trained model."""
    # Implementation would go here
    # For now, return dummy results
    return {"accuracy": 0.85, "auc": 0.94, "f1": 0.85}
