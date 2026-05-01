"""
Export trained models to TorchScript for optimized inference.

TorchScript provides:
- 2-3x faster inference
- No Python GIL overhead
- Cross-platform deployment (C++, mobile)
- Production-ready optimization

Usage:
    python scripts/export_torchscript.py \
        --checkpoint checkpoints/best_model.pth \
        --output models/model_scripted.pt \
        --optimize
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

sys.path.append(str(Path(__file__).parent.parent))

from src.models import ClassificationHead, MultimodalFusionModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_checkpoint(
    checkpoint_path: Path, device: torch.device
) -> Tuple[nn.Module, nn.Module, Dict]:
    """Load model and task head from checkpoint."""
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get config
    config = checkpoint.get("config", {})
    
    # Initialize model
    model = MultimodalFusionModel(
        wsi_feature_dim=config.get("wsi_feature_dim", 1024),
        genomic_feature_dim=config.get("genomic_feature_dim", 2000),
        clinical_vocab_size=config.get("clinical_vocab_size", 30000),
        fusion_dim=config.get("embed_dim", 256),
    )
    
    # Initialize task head
    num_classes = config.get("num_classes", 2)
    task_head = ClassificationHead(
        input_dim=config.get("embed_dim", 256),
        num_classes=num_classes,
    )
    
    # Load state dicts
    model.load_state_dict(checkpoint["model_state_dict"])
    task_head.load_state_dict(checkpoint["task_head_state_dict"])
    
    model.to(device)
    task_head.to(device)
    
    logger.info("Checkpoint loaded successfully")
    return model, task_head, config


class ScriptableModel(nn.Module):
    """Wrapper to make model scriptable with TorchScript."""
    
    def __init__(self, fusion_model: nn.Module, task_head: nn.Module):
        super().__init__()
        self.fusion_model = fusion_model
        self.task_head = task_head
    
    def forward(
        self,
        wsi_features: Optional[torch.Tensor] = None,
        genomic: Optional[torch.Tensor] = None,
        clinical_text: Optional[torch.Tensor] = None,
        wsi_mask: Optional[torch.Tensor] = None,
        clinical_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for inference."""
        # Create batch dict
        batch = {}
        if wsi_features is not None:
            batch["wsi_features"] = wsi_features
            if wsi_mask is not None:
                batch["wsi_mask"] = wsi_mask
        if genomic is not None:
            batch["genomic"] = genomic
        if clinical_text is not None:
            batch["clinical_text"] = clinical_text
            if clinical_mask is not None:
                batch["clinical_mask"] = clinical_mask
        
        # Get embeddings
        embeddings = self.fusion_model(batch)
        
        # Get logits
        logits = self.task_head(embeddings)
        
        return logits


def export_torchscript(
    model: nn.Module,
    task_head: nn.Module,
    output_path: Path,
    example_inputs: Dict[str, torch.Tensor],
    optimize: bool = True,
) -> None:
    """Export model to TorchScript format."""
    logger.info("Creating scriptable model wrapper...")
    
    # Create scriptable wrapper
    scriptable_model = ScriptableModel(model, task_head)
    scriptable_model.eval()
    
    # Trace model with example inputs
    logger.info("Tracing model with example inputs...")
    try:
        with torch.no_grad():
            traced_model = torch.jit.trace(
                scriptable_model,
                example_inputs=(
                    example_inputs.get("wsi_features"),
                    example_inputs.get("genomic"),
                    example_inputs.get("clinical_text"),
                    example_inputs.get("wsi_mask"),
                    example_inputs.get("clinical_mask"),
                ),
            )
        
        # Optimize for inference
        if optimize:
            logger.info("Optimizing traced model...")
            traced_model = torch.jit.optimize_for_inference(traced_model)
        
        # Save
        logger.info(f"Saving TorchScript model to {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.jit.save(traced_model, str(output_path))
        
        logger.info("✓ TorchScript export successful")
        
        # Verify
        logger.info("Verifying exported model...")
        loaded_model = torch.jit.load(str(output_path))
        with torch.no_grad():
            original_output = scriptable_model(**example_inputs)
            loaded_output = loaded_model(
                example_inputs.get("wsi_features"),
                example_inputs.get("genomic"),
                example_inputs.get("clinical_text"),
                example_inputs.get("wsi_mask"),
                example_inputs.get("clinical_mask"),
            )
        
        # Check outputs match
        if torch.allclose(original_output, loaded_output, rtol=1e-4, atol=1e-4):
            logger.info("✓ Verification passed - outputs match")
        else:
            logger.warning("⚠ Verification warning - outputs differ slightly")
            max_diff = (original_output - loaded_output).abs().max().item()
            logger.warning(f"  Max difference: {max_diff:.6f}")
        
    except Exception as e:
        logger.error(f"TorchScript export failed: {e}")
        raise


def create_example_inputs(
    config: Dict, device: torch.device, batch_size: int = 1
) -> Dict[str, Optional[torch.Tensor]]:
    """Create example inputs for tracing."""
    logger.info("Creating example inputs for tracing...")
    
    example_inputs = {}
    
    # WSI features
    if config.get("wsi_enabled", True):
        num_patches = 100
        wsi_feature_dim = config.get("wsi_feature_dim", 1024)
        example_inputs["wsi_features"] = torch.randn(
            batch_size, num_patches, wsi_feature_dim, device=device
        )
        example_inputs["wsi_mask"] = torch.ones(
            batch_size, num_patches, dtype=torch.bool, device=device
        )
    else:
        example_inputs["wsi_features"] = None
        example_inputs["wsi_mask"] = None
    
    # Genomic features
    if config.get("genomic_enabled", True):
        genomic_feature_dim = config.get("genomic_feature_dim", 2000)
        example_inputs["genomic"] = torch.randn(
            batch_size, genomic_feature_dim, device=device
        )
    else:
        example_inputs["genomic"] = None
    
    # Clinical text
    if config.get("clinical_text_enabled", True):
        max_text_length = config.get("max_text_length", 512)
        example_inputs["clinical_text"] = torch.randint(
            0, config.get("clinical_vocab_size", 30000),
            (batch_size, max_text_length),
            device=device,
        )
        example_inputs["clinical_mask"] = torch.ones(
            batch_size, max_text_length, dtype=torch.bool, device=device
        )
    else:
        example_inputs["clinical_text"] = None
        example_inputs["clinical_mask"] = None
    
    return example_inputs


def benchmark_inference(
    original_model: nn.Module,
    scripted_model: torch.jit.ScriptModule,
    example_inputs: Dict[str, torch.Tensor],
    num_iterations: int = 100,
) -> None:
    """Benchmark inference speed comparison."""
    logger.info(f"\nBenchmarking inference speed ({num_iterations} iterations)...")
    
    import time
    
    device = next(original_model.parameters()).device
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = original_model(**example_inputs)
            _ = scripted_model(
                example_inputs.get("wsi_features"),
                example_inputs.get("genomic"),
                example_inputs.get("clinical_text"),
                example_inputs.get("wsi_mask"),
                example_inputs.get("clinical_mask"),
            )
    
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    # Benchmark original model
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = original_model(**example_inputs)
    if device.type == "cuda":
        torch.cuda.synchronize()
    original_time = (time.perf_counter() - start) / num_iterations * 1000
    
    # Benchmark scripted model
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = scripted_model(
                example_inputs.get("wsi_features"),
                example_inputs.get("genomic"),
                example_inputs.get("clinical_text"),
                example_inputs.get("wsi_mask"),
                example_inputs.get("clinical_mask"),
            )
    if device.type == "cuda":
        torch.cuda.synchronize()
    scripted_time = (time.perf_counter() - start) / num_iterations * 1000
    
    # Report results
    speedup = original_time / scripted_time
    logger.info(f"\nInference Speed Comparison:")
    logger.info(f"  Original model: {original_time:.2f} ms/sample")
    logger.info(f"  TorchScript:    {scripted_time:.2f} ms/sample")
    logger.info(f"  Speedup:        {speedup:.2f}x")


def main():
    parser = argparse.ArgumentParser(description="Export model to TorchScript")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/model_scripted.pt",
        help="Output path for TorchScript model",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda/cpu)",
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Optimize model for inference",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Benchmark inference speed",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for example inputs",
    )
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load checkpoint
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return
    
    model, task_head, config = load_checkpoint(checkpoint_path, device)
    
    # Set to eval mode
    model.eval()
    task_head.eval()
    
    # Create example inputs
    example_inputs = create_example_inputs(config, device, args.batch_size)
    
    # Create scriptable wrapper
    scriptable_model = ScriptableModel(model, task_head)
    scriptable_model.eval()
    
    # Export to TorchScript
    output_path = Path(args.output)
    export_torchscript(
        model,
        task_head,
        output_path,
        example_inputs,
        optimize=args.optimize,
    )
    
    # Benchmark if requested
    if args.benchmark:
        scripted_model = torch.jit.load(str(output_path))
        benchmark_inference(scriptable_model, scripted_model, example_inputs)
    
    # Print usage instructions
    logger.info(f"\n{'='*60}")
    logger.info("Export complete! Usage:")
    logger.info(f"{'='*60}")
    logger.info(f"\nPython:")
    logger.info(f"  model = torch.jit.load('{output_path}')")
    logger.info(f"  output = model(wsi_features, genomic, clinical_text, wsi_mask, clinical_mask)")
    logger.info(f"\nC++:")
    logger.info(f"  torch::jit::script::Module model = torch::jit::load('{output_path}');")
    logger.info(f"  auto output = model.forward({{wsi_features, genomic, clinical_text, wsi_mask, clinical_mask}});")
    logger.info(f"\n{'='*60}")


if __name__ == "__main__":
    main()
