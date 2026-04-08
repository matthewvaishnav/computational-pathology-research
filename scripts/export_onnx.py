#!/usr/bin/env python3
"""
ONNX Model Export Script

This script exports the current multimodal model checkpoints to ONNX for
deployment and offline inference.

Usage:
    python scripts/export_onnx.py --checkpoint checkpoints/checkpoint_best.pth --output models/model.onnx

    # Export with custom dummy batch shapes
    python scripts/export_onnx.py \
        --checkpoint checkpoints/checkpoint_best.pth \
        --output models/model.onnx \
        --batch-size 1 \
        --wsi-num-patches 128 \
        --clinical-seq-len 256

    # With optimization
    python scripts/export_onnx.py \
        --checkpoint checkpoints/checkpoint_best.pth \
        --output models/model.onnx \
        --optimize \
        --opset-version 14
"""

import argparse
import copy
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.heads import ClassificationHead, SurvivalPredictionHead
from src.models.multimodal import MultimodalFusionModel

DEFAULT_WSI_NUM_PATCHES = 100


def build_default_model_config(embed_dim: int = 256, dropout: float = 0.1) -> Dict[str, Any]:
    """Return the current default multimodal model configuration."""
    return {
        "embed_dim": embed_dim,
        "dropout": dropout,
        "wsi_config": {
            "input_dim": 1024,
            "hidden_dim": 512,
            "output_dim": embed_dim,
            "num_heads": 8,
            "num_layers": 2,
            "dropout": dropout,
            "pooling": "attention",
        },
        "genomic_config": {
            "input_dim": 2000,
            "hidden_dims": [1024, 512],
            "output_dim": embed_dim,
            "dropout": dropout * 1.5,
            "use_batch_norm": True,
        },
        "clinical_config": {
            "vocab_size": 30000,
            "embed_dim": 256,
            "hidden_dim": 512,
            "output_dim": embed_dim,
            "num_heads": 8,
            "num_layers": 3,
            "max_seq_length": 512,
            "dropout": dropout,
            "pooling": "mean",
        },
        "fusion_config": {
            "embed_dim": embed_dim,
            "num_heads": 8,
            "dropout": dropout,
            "modalities": ["wsi", "genomic", "clinical"],
        },
    }


class ExportWrapper(nn.Module):
    """Wrap backbone and optional task head for ONNX export."""

    def __init__(self, backbone: MultimodalFusionModel, task_head: Optional[nn.Module] = None):
        super().__init__()
        self.backbone = backbone
        self.task_head = task_head

    def forward(
        self,
        wsi_features: torch.Tensor,
        wsi_mask: torch.Tensor,
        genomic: torch.Tensor,
        clinical_text: torch.Tensor,
        clinical_mask: torch.Tensor,
    ):
        batch = {
            "wsi_features": wsi_features,
            "wsi_mask": wsi_mask,
            "genomic": genomic,
            "clinical_text": clinical_text,
            "clinical_mask": clinical_mask,
        }
        embeddings = self.backbone(batch)

        if self.task_head is None:
            return embeddings

        logits = self.task_head(embeddings)
        return logits, embeddings


class ONNXExporter:
    """Export PyTorch checkpoints to ONNX format."""

    def __init__(
        self,
        checkpoint_path: str,
        output_path: str,
        batch_size: int = 1,
        wsi_num_patches: int = DEFAULT_WSI_NUM_PATCHES,
        clinical_seq_len: Optional[int] = None,
        opset_version: int = 14,
        optimize: bool = False,
        verbose: bool = True,
    ):
        """
        Initialize ONNX exporter.

        Args:
            checkpoint_path: Path to PyTorch checkpoint
            output_path: Path to save ONNX model
            batch_size: Batch size for export
            wsi_num_patches: Number of WSI patches in dummy export batch
            clinical_seq_len: Clinical text sequence length for dummy export batch
            opset_version: ONNX opset version
            optimize: Whether to optimize the ONNX model
            verbose: Whether to print detailed information
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.output_path = Path(output_path)
        self.batch_size = batch_size
        self.wsi_num_patches = wsi_num_patches
        self.clinical_seq_len = clinical_seq_len
        self.opset_version = opset_version
        self.optimize = optimize
        self.verbose = verbose
        self.export_model: Optional[ExportWrapper] = None
        self.model_config: Optional[Dict[str, Any]] = None
        self.output_names = ["embeddings"]

        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)

    def _get_backbone_state_dict(self, checkpoint: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        if "model_state_dict" in checkpoint:
            return checkpoint["model_state_dict"]
        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]
        return checkpoint

    def _extract_config_from_checkpoint(
        self, checkpoint: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        raw_config = checkpoint.get("config")
        if not isinstance(raw_config, dict):
            return None

        model_config = raw_config.get("model", raw_config)
        if not isinstance(model_config, dict):
            return None

        if any(
            key in model_config
            for key in (
                "wsi_config",
                "genomic_config",
                "clinical_config",
                "fusion_config",
                "embed_dim",
            )
        ):
            return model_config

        return None

    def _infer_config_from_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        config = build_default_model_config()

        wsi_input = state_dict.get("wsi_encoder.input_proj.0.weight")
        wsi_output = state_dict.get("wsi_encoder.output_proj.0.weight")
        if wsi_input is not None:
            config["wsi_config"]["input_dim"] = wsi_input.shape[1]
            config["wsi_config"]["hidden_dim"] = wsi_input.shape[0]
        if wsi_output is not None:
            config["wsi_config"]["output_dim"] = wsi_output.shape[0]
            config["embed_dim"] = wsi_output.shape[0]

        genomic_linear_keys = sorted(
            key
            for key, value in state_dict.items()
            if key.startswith("genomic_encoder.mlp.")
            and key.endswith(".weight")
            and value.dim() == 2
        )
        if genomic_linear_keys:
            genomic_weights = [state_dict[key] for key in genomic_linear_keys]
            config["genomic_config"]["input_dim"] = genomic_weights[0].shape[1]
            config["genomic_config"]["hidden_dims"] = [
                weight.shape[0] for weight in genomic_weights[:-1]
            ]
            config["genomic_config"]["output_dim"] = genomic_weights[-1].shape[0]
            config["embed_dim"] = genomic_weights[-1].shape[0]

        clinical_embedding = state_dict.get("clinical_encoder.token_embedding.weight")
        clinical_proj = state_dict.get("clinical_encoder.embed_proj.weight")
        clinical_output = state_dict.get("clinical_encoder.output_proj.0.weight")
        clinical_positional = state_dict.get("clinical_encoder.positional_encoding")
        if clinical_embedding is not None:
            config["clinical_config"]["vocab_size"] = clinical_embedding.shape[0]
            config["clinical_config"]["embed_dim"] = clinical_embedding.shape[1]
        if clinical_proj is not None:
            config["clinical_config"]["hidden_dim"] = clinical_proj.shape[0]
        if clinical_output is not None:
            config["clinical_config"]["output_dim"] = clinical_output.shape[0]
            config["embed_dim"] = clinical_output.shape[0]
        if clinical_positional is not None:
            config["clinical_config"]["max_seq_length"] = clinical_positional.shape[1]

        wsi_layers = {
            int(key.split(".")[3])
            for key in state_dict
            if key.startswith("wsi_encoder.transformer.layers.")
        }
        clinical_layers = {
            int(key.split(".")[3])
            for key in state_dict
            if key.startswith("clinical_encoder.transformer.layers.")
        }
        if wsi_layers:
            config["wsi_config"]["num_layers"] = max(wsi_layers) + 1
        if clinical_layers:
            config["clinical_config"]["num_layers"] = max(clinical_layers) + 1

        config["wsi_config"]["output_dim"] = config["embed_dim"]
        config["genomic_config"]["output_dim"] = config["embed_dim"]
        config["clinical_config"]["output_dim"] = config["embed_dim"]
        config["fusion_config"]["embed_dim"] = config["embed_dim"]

        return config

    def _resolve_model_config(self, checkpoint: Dict[str, Any]) -> Dict[str, Any]:
        state_dict = self._get_backbone_state_dict(checkpoint)
        config = self._infer_config_from_state_dict(state_dict)

        checkpoint_config = self._extract_config_from_checkpoint(checkpoint)
        if checkpoint_config is None:
            return config

        embed_dim = checkpoint_config.get("embed_dim", config["embed_dim"])
        dropout = checkpoint_config.get("dropout", config["dropout"])
        merged = build_default_model_config(embed_dim=embed_dim, dropout=dropout)

        for key in ("wsi_config", "genomic_config", "clinical_config", "fusion_config"):
            merged[key].update(config[key])
            if key in checkpoint_config and isinstance(checkpoint_config[key], dict):
                merged[key].update(checkpoint_config[key])

        merged["embed_dim"] = checkpoint_config.get("embed_dim", config["embed_dim"])
        merged["dropout"] = checkpoint_config.get("dropout", config["dropout"])
        merged["wsi_config"]["output_dim"] = merged["embed_dim"]
        merged["genomic_config"]["output_dim"] = merged["embed_dim"]
        merged["clinical_config"]["output_dim"] = merged["embed_dim"]
        merged["fusion_config"]["embed_dim"] = merged["embed_dim"]

        if "wsi_dim" in checkpoint_config:
            merged["wsi_config"]["input_dim"] = checkpoint_config["wsi_dim"]
        if "clinical_dim" in checkpoint_config:
            merged["clinical_config"]["output_dim"] = checkpoint_config["clinical_dim"]

        return merged

    def _build_backbone(self, checkpoint: Dict[str, Any]) -> MultimodalFusionModel:
        self.model_config = self._resolve_model_config(checkpoint)

        backbone = MultimodalFusionModel(
            wsi_config=copy.deepcopy(self.model_config["wsi_config"]),
            genomic_config=copy.deepcopy(self.model_config["genomic_config"]),
            clinical_config=copy.deepcopy(self.model_config["clinical_config"]),
            fusion_config=copy.deepcopy(self.model_config["fusion_config"]),
            embed_dim=self.model_config["embed_dim"],
            dropout=self.model_config["dropout"],
        )

        backbone.load_state_dict(self._get_backbone_state_dict(checkpoint))
        return backbone

    def _build_task_head(self, checkpoint: Dict[str, Any], embed_dim: int) -> Optional[nn.Module]:
        state_dict = checkpoint.get("task_head_state_dict")
        if not isinstance(state_dict, dict):
            return None

        if any(key.startswith("classifier.") for key in state_dict):
            use_hidden_layer = (
                "classifier.0.weight" in state_dict and "classifier.4.weight" in state_dict
            )
            if use_hidden_layer:
                hidden_dim = state_dict["classifier.0.weight"].shape[0]
                num_classes = state_dict["classifier.4.weight"].shape[0]
            else:
                linear_keys = sorted(
                    key
                    for key, value in state_dict.items()
                    if key.startswith("classifier.")
                    and key.endswith(".weight")
                    and value.dim() == 2
                )
                final_key = linear_keys[-1]
                hidden_dim = embed_dim
                num_classes = state_dict[final_key].shape[0]

            head = ClassificationHead(
                input_dim=embed_dim,
                hidden_dim=hidden_dim,
                num_classes=num_classes,
                use_hidden_layer=use_hidden_layer,
            )
            head.load_state_dict(state_dict)
            return head

        if any(key.startswith("predictor.") for key in state_dict):
            use_hidden_layer = (
                "predictor.0.weight" in state_dict and "predictor.4.weight" in state_dict
            )
            if use_hidden_layer:
                hidden_dim = state_dict["predictor.0.weight"].shape[0]
                output_dim = state_dict["predictor.4.weight"].shape[0]
            else:
                linear_keys = sorted(
                    key
                    for key, value in state_dict.items()
                    if key.startswith("predictor.") and key.endswith(".weight") and value.dim() == 2
                )
                final_key = linear_keys[-1]
                hidden_dim = embed_dim
                output_dim = state_dict[final_key].shape[0]

            head = SurvivalPredictionHead(
                input_dim=embed_dim,
                hidden_dim=hidden_dim,
                num_time_bins=None if output_dim == 1 else output_dim,
                use_hidden_layer=use_hidden_layer,
            )
            head.load_state_dict(state_dict)
            return head

        return None

    def load_model(self) -> ExportWrapper:
        """Load checkpoint into an exportable backbone + optional task head."""
        self._log(f"Loading checkpoint from {self.checkpoint_path}")
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")

        backbone = self._build_backbone(checkpoint)
        task_head = self._build_task_head(checkpoint, backbone.get_embedding_dim())

        self.export_model = ExportWrapper(backbone, task_head)
        self.export_model.eval()
        self.output_names = ["logits", "embeddings"] if task_head is not None else ["embeddings"]

        self._log("Model loaded successfully")
        self._log(f"  Embedding dim: {backbone.get_embedding_dim()}")
        self._log(f"  Task head: {'present' if task_head is not None else 'none'}")

        return self.export_model

    def create_dummy_inputs(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create dummy batch tensors for ONNX tracing."""
        if self.model_config is None:
            raise RuntimeError("Model must be loaded before creating dummy inputs")

        clinical_seq_len = (
            self.clinical_seq_len or self.model_config["clinical_config"]["max_seq_length"]
        )
        clinical_seq_len = min(
            clinical_seq_len, self.model_config["clinical_config"]["max_seq_length"]
        )

        wsi_features = torch.randn(
            self.batch_size,
            self.wsi_num_patches,
            self.model_config["wsi_config"]["input_dim"],
        )
        wsi_mask = torch.ones(self.batch_size, self.wsi_num_patches, dtype=torch.bool)
        genomic = torch.randn(self.batch_size, self.model_config["genomic_config"]["input_dim"])
        clinical_text = torch.randint(
            low=1,
            high=self.model_config["clinical_config"]["vocab_size"],
            size=(self.batch_size, clinical_seq_len),
            dtype=torch.long,
        )
        clinical_mask = torch.ones(self.batch_size, clinical_seq_len, dtype=torch.bool)

        self._log("\nDummy input shapes:")
        self._log(f"  WSI features: {tuple(wsi_features.shape)}")
        self._log(f"  WSI mask: {tuple(wsi_mask.shape)}")
        self._log(f"  Genomic: {tuple(genomic.shape)}")
        self._log(f"  Clinical text: {tuple(clinical_text.shape)}")
        self._log(f"  Clinical mask: {tuple(clinical_mask.shape)}")

        return wsi_features, wsi_mask, genomic, clinical_text, clinical_mask

    def export(self) -> None:
        """Export model to ONNX format."""
        model = self.load_model()
        dummy_inputs = self.create_dummy_inputs()

        self._log("\nExporting to ONNX...")
        self._log(f"  Output path: {self.output_path}")
        self._log(f"  Opset version: {self.opset_version}")

        dynamic_axes = {
            "wsi_features": {0: "batch_size", 1: "num_patches"},
            "wsi_mask": {0: "batch_size", 1: "num_patches"},
            "genomic": {0: "batch_size"},
            "clinical_text": {0: "batch_size", 1: "seq_len"},
            "clinical_mask": {0: "batch_size", 1: "seq_len"},
            "embeddings": {0: "batch_size"},
        }
        if "logits" in self.output_names:
            dynamic_axes["logits"] = {0: "batch_size"}

        torch.onnx.export(
            model,
            dummy_inputs,
            str(self.output_path),
            export_params=True,
            opset_version=self.opset_version,
            do_constant_folding=True,
            input_names=["wsi_features", "wsi_mask", "genomic", "clinical_text", "clinical_mask"],
            output_names=self.output_names,
            dynamic_axes=dynamic_axes,
            verbose=self.verbose,
        )

        self._log(f"Model exported successfully to {self.output_path}")

        if self.optimize:
            self.optimize_model()

        self.verify_export()

    def optimize_model(self) -> None:
        """Optimize ONNX model."""
        try:
            import onnx
            from onnxruntime.transformers import optimizer

            self._log("\nOptimizing ONNX model...")

            onnx_model = onnx.load(str(self.output_path))
            hidden_size = self.model_config["embed_dim"] if self.model_config is not None else 256
            num_heads = (
                self.model_config["fusion_config"].get("num_heads", 8) if self.model_config else 8
            )

            optimized_model = optimizer.optimize_model(
                str(self.output_path),
                model_type="bert",
                num_heads=num_heads,
                hidden_size=hidden_size,
            )

            optimized_path = self.output_path.with_suffix(".optimized.onnx")
            optimized_model.save_model_to_file(str(optimized_path))
            self._log(f"Optimized model saved to {optimized_path}")

        except ImportError:
            self._log("Warning: onnxruntime not installed, skipping optimization")
            self._log("Install with: pip install onnxruntime")

    def verify_export(self) -> None:
        """Verify ONNX export by running inference."""
        try:
            import onnx
            import onnxruntime as ort

            self._log("\nVerifying ONNX export...")

            onnx_model = onnx.load(str(self.output_path))
            onnx.checker.check_model(onnx_model)
            self._log("ONNX model is valid")

            ort_session = ort.InferenceSession(str(self.output_path))
            wsi_features, wsi_mask, genomic, clinical_text, clinical_mask = (
                self.create_dummy_inputs()
            )

            outputs = ort_session.run(
                None,
                {
                    "wsi_features": wsi_features.numpy(),
                    "wsi_mask": wsi_mask.numpy(),
                    "genomic": genomic.numpy(),
                    "clinical_text": clinical_text.numpy(),
                    "clinical_mask": clinical_mask.numpy(),
                },
            )

            self._log("ONNX inference successful")
            self._log("  Output shapes:")
            for name, output in zip(self.output_names, outputs):
                self._log(f"    {name}: {output.shape}")

            self._log("\nModel information:")
            self._log(f"  File size: {self.output_path.stat().st_size / 1024 / 1024:.2f} MB")
            self._log(f"  Inputs: {[inp.name for inp in ort_session.get_inputs()]}")
            self._log(f"  Outputs: {[out.name for out in ort_session.get_outputs()]}")

        except ImportError:
            self._log("Warning: onnx or onnxruntime not installed, skipping verification")
            self._log("Install with: pip install onnx onnxruntime")
        except Exception as exc:
            print(f"Error during verification: {exc}")
            raise


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Export PyTorch multimodal checkpoint to ONNX format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to PyTorch checkpoint file",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save ONNX model",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for export",
    )
    parser.add_argument(
        "--wsi-num-patches",
        type=int,
        default=DEFAULT_WSI_NUM_PATCHES,
        help="Number of WSI patches in the dummy export batch",
    )
    parser.add_argument(
        "--clinical-seq-len",
        type=int,
        default=None,
        help="Clinical sequence length for the dummy export batch",
    )
    parser.add_argument(
        "--opset-version",
        type=int,
        default=14,
        help="ONNX opset version",
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Optimize ONNX model",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    exporter = ONNXExporter(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        batch_size=args.batch_size,
        wsi_num_patches=args.wsi_num_patches,
        clinical_seq_len=args.clinical_seq_len,
        opset_version=args.opset_version,
        optimize=args.optimize,
        verbose=not args.quiet,
    )

    try:
        exporter.export()
        print("\nExport completed successfully!")
    except Exception as exc:
        print(f"\nExport failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
