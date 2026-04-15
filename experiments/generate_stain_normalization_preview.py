"""
Generate stain-normalization before/after preview artifacts from a checkpoint.

This utility loads a trained ``StainNormalizationTransformer`` checkpoint plus
its config, runs normalization on one image (optionally conditioned on a
reference style image), and writes preview-ready outputs to disk.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Tuple, Union

import numpy as np
import torch
import yaml
from PIL import Image

# Add repo root to path for direct script execution.
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.stain_normalization import StainNormalizationTransformer


def load_stain_normalization_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load stain-normalization YAML config."""
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_stain_normalization_model(config: Dict[str, Any]) -> StainNormalizationTransformer:
    """Construct a stain-normalization model from config."""
    model_cfg = config["model"]
    encoder_cfg = model_cfg.get("encoder", {})
    decoder_cfg = model_cfg.get("decoder", {})
    return StainNormalizationTransformer(
        patch_size=model_cfg.get("patch_size", 16),
        in_channels=model_cfg.get("in_channels", 3),
        embed_dim=model_cfg.get("embed_dim", 256),
        num_encoder_layers=encoder_cfg.get("num_layers", 4),
        num_decoder_layers=decoder_cfg.get("num_layers", 4),
        num_heads=encoder_cfg.get("num_heads", 8),
        mlp_ratio=encoder_cfg.get("mlp_ratio", 4.0),
        dropout=encoder_cfg.get("dropout", 0.1),
        style_dim=model_cfg.get("style_conditioner", {}).get("style_dim", 128),
    )


def load_stain_normalization_model(
    checkpoint_path: Union[str, Path],
    config_path: Union[str, Path],
    device: Union[str, torch].device = "cpu",
) -> StainNormalizationTransformer:
    """Load a stain-normalization checkpoint into eval mode."""
    config = load_stain_normalization_config(config_path)
    model = build_stain_normalization_model(config)
    checkpoint = torch.load(checkpoint_path, map_location=str(device))
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model


def load_preview_image(
    image_path: Union[str, Path], image_size: int
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Load an RGB image, resize for preview inference, and convert to [-1, 1]."""
    image = Image.open(image_path).convert("RGB")
    original_size = image.size
    if image.size != (image_size, image_size):
        image = image.resize((image_size, image_size), Image.Resampling.BICUBIC)
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    tensor = tensor * 2.0 - 1.0
    return tensor.unsqueeze(0), original_size


def tensor_to_pil_image(tensor: torch.Tensor) -> Image.Image:
    """Convert a single-image tensor in [-1, 1] to PIL RGB."""
    if tensor.ndim == 4:
        tensor = tensor[0]
    array = tensor.detach().cpu().clamp(-1.0, 1.0)
    array = ((array + 1.0) / 2.0).permute(1, 2, 0).numpy()
    array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def build_comparison_panel(
    input_image: Image.Image,
    normalized_image: Image.Image,
    reference_image: Image.Image | None = None,
) -> Image.Image:
    """Create a side-by-side preview panel."""
    images = [input_image]
    if reference_image is not None:
        images.append(reference_image)
    images.append(normalized_image)

    width = sum(image.width for image in images)
    height = max(image.height for image in images)
    panel = Image.new("RGB", (width, height), color=(255, 255, 255))

    offset_x = 0
    for image in images:
        panel.paste(image, (offset_x, 0))
        offset_x += image.width

    return panel


def generate_stain_normalization_preview(
    *,
    checkpoint_path: Union[str, Path],
    config_path: Union[str, Path],
    input_image_path: Union[str, Path],
    output_dir: Union[str, Path],
    reference_image_path: Union[Union[str, Path], None] = None,
    device: Union[str, torch].device = "cpu",
) -> Dict[str, Any]:
    """Generate normalized preview images plus summary metadata."""
    config = load_stain_normalization_config(config_path)
    image_size = int(config["model"].get("image_size", config["data"].get("image_size", 256)))

    model = load_stain_normalization_model(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        device=device,
    )

    input_tensor, input_original_size = load_preview_image(input_image_path, image_size)
    reference_tensor = None
    reference_original_size = None
    if reference_image_path is not None:
        reference_tensor, reference_original_size = load_preview_image(
            reference_image_path, image_size
        )

    input_tensor = input_tensor.to(device)
    if reference_tensor is not None:
        reference_tensor = reference_tensor.to(device)

    with torch.no_grad():
        normalized_tensor = model(input_tensor, reference_tensor)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_preview = tensor_to_pil_image(input_tensor)
    normalized_preview = tensor_to_pil_image(normalized_tensor)
    reference_preview = (
        tensor_to_pil_image(reference_tensor) if reference_tensor is not None else None
    )
    comparison_panel = build_comparison_panel(
        input_image=input_preview,
        normalized_image=normalized_preview,
        reference_image=reference_preview,
    )

    normalized_path = output_dir / "normalized.png"
    comparison_path = output_dir / "comparison.png"
    input_preview_path = output_dir / "input_preview.png"

    input_preview.save(input_preview_path)
    normalized_preview.save(normalized_path)
    comparison_panel.save(comparison_path)

    summary = {
        "checkpoint_path": Path(checkpoint_path).as_posix(),
        "config_path": Path(config_path).as_posix(),
        "input_image_path": Path(input_image_path).as_posix(),
        "reference_image_path": (
            Path(reference_image_path).as_posix() if reference_image_path else None
        ),
        "input_original_size": list(input_original_size),
        "reference_original_size": (
            list(reference_original_size) if reference_original_size else None
        ),
        "preview_image_size": [image_size, image_size],
        "artifacts": {
            "input_preview": input_preview_path.as_posix(),
            "normalized": normalized_path.as_posix(),
            "comparison": comparison_path.as_posix(),
        },
    }
    if reference_preview is not None:
        reference_preview_path = output_dir / "reference_preview.png"
        reference_preview.save(reference_preview_path)
        summary["artifacts"]["reference_preview"] = reference_preview_path.as_posix()

    summary_path = output_dir / "stain_normalization_preview_summary.json"
    summary["summary_path"] = summary_path.as_posix()
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate stain-normalization before/after preview artifacts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python experiments/generate_stain_normalization_preview.py ^
      --checkpoint checkpoints/stain_norm/best_model.pth ^
      --config experiments/configs/stain_norm.yaml ^
      --input-image data/stain_norm/example_input.png ^
      --output-dir results/stain_norm/example_preview

  python experiments/generate_stain_normalization_preview.py ^
      --checkpoint checkpoints/stain_norm/best_model.pth ^
      --config experiments/configs/stain_norm.yaml ^
      --input-image data/stain_norm/example_input.png ^
      --reference-image data/stain_norm/example_reference.png ^
      --output-dir results/stain_norm/example_preview
        """,
    )
    parser.add_argument(
        "--checkpoint", required=True, help="Path to trained stain-normalization checkpoint"
    )
    parser.add_argument("--config", required=True, help="Path to stain-normalization YAML config")
    parser.add_argument("--input-image", required=True, help="Input RGB image to normalize")
    parser.add_argument(
        "--reference-image", default=None, help="Optional reference style RGB image"
    )
    parser.add_argument("--output-dir", required=True, help="Directory for preview artifacts")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Inference device")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")

    summary = generate_stain_normalization_preview(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        input_image_path=args.input_image,
        output_dir=args.output_dir,
        reference_image_path=args.reference_image,
        device=args.device,
    )
    print(f"Saved stain-normalization preview summary to {summary['summary_path']}")


if __name__ == "__main__":
    main()
