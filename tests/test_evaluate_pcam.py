"""Targeted tests for optional PCam interpretability integration during evaluation."""

from argparse import Namespace
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import experiments.evaluate_pcam as evaluate_pcam


def _make_args(**overrides):
    base = {
        "checkpoint": "checkpoints/pcam/best_model.pth",
        "data_root": "data/pcam",
        "batch_size": 64,
        "generate_interpretability": False,
        "interpretability_output_dir": None,
        "interpretability_max_samples": 32,
        "interpretability_top_k": 10,
    }
    base.update(overrides)
    return Namespace(**base)


def test_build_interpretability_metadata_includes_eval_context(tmp_path):
    args = _make_args(interpretability_max_samples=16)
    config = {"experiment": {"name": "patchcamelyon"}}

    metadata = evaluate_pcam.build_interpretability_metadata(args, config, tmp_path)

    assert metadata["checkpoint_path"].endswith("best_model.pth")
    assert metadata["data_root"] == "data/pcam"
    assert metadata["split"] == "test"
    assert metadata["batch_size"] == 64
    assert metadata["max_samples"] == 16
    assert metadata["experiment_name"] == "patchcamelyon"
    assert metadata["evaluation_output_dir"] == str(tmp_path)


def test_maybe_generate_interpretability_artifacts_skips_when_disabled(tmp_path):
    dataloader = DataLoader(TensorDataset(torch.randn(2, 3)), batch_size=1)

    summary = evaluate_pcam.maybe_generate_interpretability_artifacts(
        args=_make_args(),
        config={"experiment": {"name": "patchcamelyon"}},
        feature_extractor=nn.Identity(),
        encoder=nn.Identity(),
        head=nn.Identity(),
        dataloader=dataloader,
        output_dir=tmp_path,
        device=torch.device("cpu"),
    )

    assert summary is None


def test_maybe_generate_interpretability_artifacts_uses_default_subdir(tmp_path, monkeypatch):
    captured = {}

    def _fake_builder(**kwargs):
        captured.update(kwargs)
        output_dir = Path(kwargs["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = output_dir / "interpretability_summary.json"
        summary_path.write_text("{}", encoding="utf-8")
        return {
            "summary_path": str(summary_path),
            "artifacts": {"pca_plot": str(output_dir / "pcam_embeddings_pca.png")},
            "metadata": kwargs["metadata"],
        }

    monkeypatch.setattr(evaluate_pcam, "build_pcam_interpretability_artifacts", _fake_builder)

    dataloader = DataLoader(TensorDataset(torch.randn(2, 3)), batch_size=1)
    args = _make_args(generate_interpretability=True, interpretability_max_samples=24)

    summary = evaluate_pcam.maybe_generate_interpretability_artifacts(
        args=args,
        config={"experiment": {"name": "pcam-baseline"}},
        feature_extractor=nn.Identity(),
        encoder=nn.Identity(),
        head=nn.Identity(),
        dataloader=dataloader,
        output_dir=tmp_path,
        device=torch.device("cpu"),
    )

    assert summary is not None
    assert Path(summary["summary_path"]).parent == tmp_path / "interpretability"
    assert captured["max_samples"] == 24
    assert captured["top_k"] == 10
    assert captured["metadata"]["experiment_name"] == "pcam-baseline"
    assert captured["metadata"]["evaluation_output_dir"] == str(tmp_path)
