"""Tests for stain-normalization preview generation."""

import json
from pathlib import Path

import torch
import yaml
from PIL import Image

from experiments.generate_stain_normalization_preview import (
    generate_stain_normalization_preview,
    load_stain_normalization_model,
)
from src.models.stain_normalization import StainNormalizationTransformer


def _write_preview_config(config_path: Path) -> dict:
    config = {
        "model": {
            "name": "stain_normalization_transformer",
            "patch_size": 8,
            "image_size": 16,
            "in_channels": 3,
            "embed_dim": 32,
            "encoder": {
                "num_layers": 1,
                "num_heads": 4,
                "mlp_ratio": 2.0,
                "dropout": 0.0,
            },
            "style_conditioner": {
                "style_dim": 16,
            },
            "decoder": {
                "num_layers": 1,
                "num_heads": 4,
                "mlp_ratio": 2.0,
                "dropout": 0.0,
            },
        },
        "data": {
            "image_size": 16,
        },
    }
    with open(config_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle)
    return config


def _write_preview_checkpoint(checkpoint_path: Path) -> None:
    model = StainNormalizationTransformer(
        patch_size=8,
        embed_dim=32,
        num_encoder_layers=1,
        num_decoder_layers=1,
        num_heads=4,
        mlp_ratio=2.0,
        dropout=0.0,
        style_dim=16,
    )
    torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)


def _write_preview_image(path: Path, color: tuple[int, int, int]) -> None:
    Image.new("RGB", (20, 12), color=color).save(path)


def test_load_stain_normalization_model(tmp_path):
    config_path = tmp_path / "stain_norm.yaml"
    checkpoint_path = tmp_path / "checkpoint.pth"
    _write_preview_config(config_path)
    _write_preview_checkpoint(checkpoint_path)

    model = load_stain_normalization_model(checkpoint_path, config_path)
    assert isinstance(model, StainNormalizationTransformer)
    assert not model.training


def test_generate_stain_normalization_preview_without_reference(tmp_path):
    config_path = tmp_path / "stain_norm.yaml"
    checkpoint_path = tmp_path / "checkpoint.pth"
    input_image_path = tmp_path / "input.png"
    _write_preview_config(config_path)
    _write_preview_checkpoint(checkpoint_path)
    _write_preview_image(input_image_path, (170, 90, 120))

    summary = generate_stain_normalization_preview(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        input_image_path=input_image_path,
        output_dir=tmp_path / "preview",
    )

    assert Path(summary["artifacts"]["input_preview"]).exists()
    assert Path(summary["artifacts"]["normalized"]).exists()
    assert Path(summary["artifacts"]["comparison"]).exists()
    assert Path(summary["summary_path"]).exists()
    assert summary["reference_image_path"] is None

    saved_summary = json.loads(Path(summary["summary_path"]).read_text(encoding="utf-8"))
    assert saved_summary["preview_image_size"] == [16, 16]
    assert "reference_preview" not in saved_summary["artifacts"]


def test_generate_stain_normalization_preview_with_reference(tmp_path):
    config_path = tmp_path / "stain_norm.yaml"
    checkpoint_path = tmp_path / "checkpoint.pth"
    input_image_path = tmp_path / "input.png"
    reference_image_path = tmp_path / "reference.png"
    _write_preview_config(config_path)
    _write_preview_checkpoint(checkpoint_path)
    _write_preview_image(input_image_path, (170, 90, 120))
    _write_preview_image(reference_image_path, (90, 150, 180))

    summary = generate_stain_normalization_preview(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        input_image_path=input_image_path,
        output_dir=tmp_path / "preview_with_reference",
        reference_image_path=reference_image_path,
    )

    assert Path(summary["artifacts"]["reference_preview"]).exists()
    assert Path(summary["artifacts"]["comparison"]).exists()
    saved_summary = json.loads(Path(summary["summary_path"]).read_text(encoding="utf-8"))
    assert saved_summary["reference_image_path"] == reference_image_path.as_posix()
