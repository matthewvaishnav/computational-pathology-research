"""
Pretraining utilities for SpatialTranscriptomicsDecoder.

Trains on publicly available Visium datasets.
Split is always by slide (not by spot) to prevent spatial data leakage.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, random_split

from .alignment import SpatialDataset
from .decoder import SpatialDecoderLoss, SpatialTranscriptomicsDecoder
from .evaluation import evaluate_decoder

logger = logging.getLogger(__name__)


class SpatialPretrainer:
    """
    Trains SpatialTranscriptomicsDecoder on Visium datasets.

    Slide-level train/val split ensures no spatial leakage:
    all spots from a given slide are either in train OR val, never both.

    Usage:
        pretrainer = SpatialPretrainer(
            patch_feature_dim=1024,
            n_genes=3000,
            checkpoint_dir="checkpoints/spatial",
        )
        pretrainer.add_slide("slide_001.h5ad", "slide_001_features.npy")
        pretrainer.add_slide("slide_002.h5ad", "slide_002_features.npy")
        results = pretrainer.train(n_epochs=100)
    """

    def __init__(
        self,
        patch_feature_dim: int = 1024,
        n_genes: int = 3000,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        lr: float = 1e-4,
        batch_size: int = 128,
        val_fraction: float = 0.2,
        checkpoint_dir: Optional[str] = None,
        device: Optional[torch.device] = None,
    ):
        self.patch_feature_dim = patch_feature_dim
        self.n_genes = n_genes
        self.lr = lr
        self.batch_size = batch_size
        self.val_fraction = val_fraction
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else Path("checkpoints/spatial")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = SpatialTranscriptomicsDecoder(
            patch_feature_dim=patch_feature_dim,
            n_genes=n_genes,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
        ).to(self.device)

        self._slides: List[Dict] = []  # list of {features, expression, coords, slide_id}

    def add_slide(
        self,
        features: np.ndarray,
        expression: np.ndarray,
        coords: Optional[np.ndarray] = None,
        slide_id: str = "slide",
        gene_names: Optional[List[str]] = None,
    ) -> None:
        """
        Register one slide's data for training.

        Args:
            features: H&E patch embeddings [n_spots, feature_dim]
            expression: Raw gene expression [n_spots, n_genes]
            coords: Spot grid coordinates [n_spots, 2] (optional)
            slide_id: Unique slide identifier
            gene_names: Gene name list (optional)
        """
        self._slides.append({
            "features": features,
            "expression": expression,
            "coords": coords,
            "slide_id": slide_id,
            "gene_names": gene_names,
        })
        logger.info(f"Added slide {slide_id}: {features.shape[0]} spots")

    def add_slide_from_h5ad(
        self,
        h5ad_path: str,
        feature_key: str = "X_pca",
        slide_id: Optional[str] = None,
    ) -> None:
        """Load slide from AnnData H5AD file."""
        dataset = SpatialDataset.from_anndata(h5ad_path, feature_key, n_top_genes=0, normalize=False)
        sid = slide_id or Path(h5ad_path).stem
        self.add_slide(
            features=dataset.features.numpy(),
            expression=dataset.expression.numpy(),
            coords=dataset.coords.numpy() if dataset.coords is not None else None,
            slide_id=sid,
            gene_names=dataset.gene_names,
        )

    def _build_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """Split slides into train/val (by slide), build DataLoaders."""
        if not self._slides:
            raise RuntimeError("No slides added. Call add_slide() first.")

        n_val = max(1, int(len(self._slides) * self.val_fraction))
        val_indices = set(np.random.choice(len(self._slides), size=n_val, replace=False))
        train_indices = set(range(len(self._slides))) - val_indices

        def _make_dataset(indices):
            all_features = np.concatenate(
                [self._slides[i]["features"] for i in indices], axis=0
            )
            all_expr = np.concatenate(
                [self._slides[i]["expression"] for i in indices], axis=0
            )
            all_coords = None
            if all(self._slides[i]["coords"] is not None for i in indices):
                all_coords = np.concatenate(
                    [self._slides[i]["coords"] for i in indices], axis=0
                )
            gene_names = self._slides[list(indices)[0]].get("gene_names")
            return SpatialDataset.from_arrays(
                all_features, all_expr, all_coords, gene_names, n_top_genes=self.n_genes
            )

        train_ds = _make_dataset(train_indices)
        val_ds = _make_dataset(val_indices)
        logger.info(
            f"Train: {len(train_ds)} spots ({len(train_indices)} slides), "
            f"Val: {len(val_ds)} spots ({len(val_indices)} slides)"
        )
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False, num_workers=0)
        return train_loader, val_loader

    def train(
        self,
        n_epochs: int = 100,
        patience: int = 10,
        save_best: bool = True,
    ) -> Dict:
        """
        Train the decoder. Saves best checkpoint by validation mean Pearson r.

        Returns:
            Dict with training history and best validation metrics.
        """
        train_loader, val_loader = self._build_loaders()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
        loss_fn = SpatialDecoderLoss()

        best_pearson = -float("inf")
        epochs_without_improvement = 0
        history: List[Dict] = []

        for epoch in range(n_epochs):
            # Training
            self.model.train()
            train_losses: List[float] = []
            for batch in train_loader:
                features = batch["features"].to(self.device)
                expression = batch["expression"].to(self.device)
                coords = batch.get("coords")
                if coords is not None:
                    coords = coords.to(self.device)

                optimizer.zero_grad()
                # Predict global expression (per-spot averaged)
                pred = self.model.predict_global(features.unsqueeze(0), coords.unsqueeze(0) if coords is not None else None)
                pred = pred.squeeze(0)
                loss, metrics = loss_fn(pred, expression)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                train_losses.append(metrics["total_loss"])

            scheduler.step()

            # Validation
            val_metrics = self._evaluate(val_loader, loss_fn)
            val_pearson = val_metrics.get("mean_pearson", -1.0)

            epoch_log = {
                "epoch": epoch + 1,
                "train_loss": float(np.mean(train_losses)),
                **val_metrics,
            }
            history.append(epoch_log)

            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{n_epochs}: "
                    f"train_loss={epoch_log['train_loss']:.4f}, "
                    f"val_pearson={val_pearson:.4f}"
                )

            if val_pearson > best_pearson:
                best_pearson = val_pearson
                epochs_without_improvement = 0
                if save_best:
                    self._save_checkpoint(epoch + 1, val_pearson)
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

        logger.info(f"Training complete. Best val mean Pearson r: {best_pearson:.4f}")
        return {"history": history, "best_val_pearson": best_pearson}

    @torch.no_grad()
    def _evaluate(self, loader: DataLoader, loss_fn: SpatialDecoderLoss) -> Dict:
        self.model.eval()
        all_pred: List[np.ndarray] = []
        all_target: List[np.ndarray] = []
        for batch in loader:
            features = batch["features"].to(self.device)
            expression = batch["expression"]
            coords = batch.get("coords")
            if coords is not None:
                coords = coords.to(self.device)
            pred = self.model.predict_global(
                features.unsqueeze(0),
                coords.unsqueeze(0) if coords is not None else None
            ).squeeze(0).cpu().numpy()
            all_pred.append(pred)
            all_target.append(expression.numpy())

        pred_all = np.concatenate(all_pred, axis=0)
        target_all = np.concatenate(all_target, axis=0)
        return evaluate_decoder(pred_all, target_all)

    def _save_checkpoint(self, epoch: int, val_pearson: float) -> None:
        path = self.checkpoint_dir / f"spatial_decoder_ep{epoch}_pearson{val_pearson:.4f}.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "val_pearson": val_pearson,
            "n_genes": self.n_genes,
            "patch_feature_dim": self.patch_feature_dim,
        }, path)
        logger.info(f"Saved checkpoint: {path}")

    def load_checkpoint(self, path: str) -> float:
        """Load model from checkpoint. Returns validation Pearson r."""
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        logger.info(f"Loaded checkpoint from {path}, val_pearson={ckpt.get('val_pearson', '?'):.4f}")
        return float(ckpt.get("val_pearson", 0.0))
