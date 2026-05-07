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
    args = _make_args(interpretability_max_samples=16, interpretability_top_k=7)
    config = {"experiment": {"name": "patchcamelyon"}}

    metadata = evaluate_pcam.build_interpretability_metadata(args, config, tmp_path)

    assert metadata["checkpoint_path"].endswith("best_model.pth")
    assert metadata["data_root"] == "data/pcam"
    assert metadata["split"] == "test"
    assert metadata["batch_size"] == 64
    assert metadata["max_samples"] == 16
    assert metadata["top_k"] == 7
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
            "report_path": str(output_dir / "interpretability_report.md"),
            "artifacts": {
                "pca_plot": str(output_dir / "pcam_embeddings_pca.png"),
                "report": str(output_dir / "interpretability_report.md"),
            },
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
    assert summary["status"] == "success"
    assert Path(summary["summary_path"]).parent == tmp_path / "interpretability"
    assert summary["report_path"].endswith("interpretability_report.md")
    assert summary["artifacts"]["report"].endswith("interpretability_report.md")
    assert captured["max_samples"] == 24
    assert captured["top_k"] == 10
    assert captured["metadata"]["experiment_name"] == "pcam-baseline"
    assert captured["metadata"]["evaluation_output_dir"] == str(tmp_path)
    assert captured["metadata"]["top_k"] == 10


def test_maybe_generate_interpretability_artifacts_returns_failure_payload(tmp_path, monkeypatch):
    def _failing_builder(**kwargs):
        del kwargs
        raise ValueError("not enough samples for interpretability")

    monkeypatch.setattr(evaluate_pcam, "build_pcam_interpretability_artifacts", _failing_builder)

    dataloader = DataLoader(TensorDataset(torch.randn(2, 3)), batch_size=1)
    args = _make_args(generate_interpretability=True)

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
    assert summary["status"] == "failed"
    assert "not enough samples" in summary["error"]
    assert summary["output_dir"].endswith("interpretability")
    assert summary["metadata"]["experiment_name"] == "pcam-baseline"


def test_log_evaluation_summary_does_not_require_plots(tmp_path, monkeypatch):
    messages = []

    def _capture(message, *args):
        if args:
            message = message % args
        messages.append(message)

    monkeypatch.setattr(evaluate_pcam.logger, "info", _capture)

    test_metrics = {
        "accuracy": 0.94,
        "auc": 1.0,
        "precision": 0.95,
        "recall": 0.93,
        "f1": 0.94,
        "confusion_matrix": [[39, 6], [0, 55]],
        "per_class_metrics": {
            "class_0": {"precision": 1.0, "recall": 0.867, "f1": 0.929},
            "class_1": {"precision": 0.902, "recall": 1.0, "f1": 0.948},
        },
    }

    evaluate_pcam.log_evaluation_summary(
        checkpoint_path="checkpoints/pcam/best_model.pth",
        epoch=3,
        test_dataset_size=100,
        inference_time=0.81,
        test_metrics=test_metrics,
        output_dir=tmp_path,
        confusion_matrix_generated=False,
        roc_curve_generated=False,
        interpretability_summary=None,
    )

    joined = "\n".join(messages)
    assert "EVALUATION SUMMARY" in joined
    assert "Confusion Matrix:" in joined
    assert "TN=39" in joined
    assert "Metrics:" in joined


def test_compute_metrics_handles_single_class_labels():
    metrics = evaluate_pcam.compute_metrics(
        predictions=torch.tensor([0, 0, 0]).numpy(),
        probabilities=torch.tensor([0.1, 0.2, 0.3]).numpy(),
        labels=torch.tensor([0, 0, 0]).numpy(),
    )

    assert metrics["accuracy"] == 1.0
    assert metrics["auc"] is None
    assert metrics["confusion_matrix"] == [[3, 0], [0, 0]]
    assert metrics["per_class_metrics"]["class_0"]["recall"] == 1.0
    assert metrics["per_class_metrics"]["class_1"]["precision"] == 0.0
    assert metrics["precision_binary"] == 0.0
    assert metrics["recall_binary"] == 0.0


def test_log_evaluation_summary_reports_undefined_auc(tmp_path, monkeypatch):
    messages = []

    def _capture(message, *args):
        if args:
            message = message % args
        messages.append(message)

    monkeypatch.setattr(evaluate_pcam.logger, "info", _capture)

    test_metrics = {
        "accuracy": 1.0,
        "auc": None,
        "precision": 0.5,
        "recall": 0.5,
        "f1": 0.5,
        "confusion_matrix": [[3, 0], [0, 0]],
        "per_class_metrics": {
            "class_0": {"precision": 1.0, "recall": 1.0, "f1": 1.0},
            "class_1": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
        },
    }

    evaluate_pcam.log_evaluation_summary(
        checkpoint_path="checkpoints/pcam/best_model.pth",
        epoch=3,
        test_dataset_size=3,
        inference_time=0.1,
        test_metrics=test_metrics,
        output_dir=tmp_path,
        confusion_matrix_generated=False,
        roc_curve_generated=False,
        interpretability_summary=None,
    )

    joined = "\n".join(messages)
    assert "AUC:       undefined" in joined


def test_log_evaluation_summary_only_lists_artifacts_that_exist(tmp_path, monkeypatch):
    messages = []

    def _capture(message, *args):
        if args:
            message = message % args
        messages.append(message)

    monkeypatch.setattr(evaluate_pcam.logger, "info", _capture)

    test_metrics = {
        "accuracy": 1.0,
        "auc": None,
        "precision": 0.5,
        "recall": 0.5,
        "f1": 0.5,
        "confusion_matrix": [[3, 0], [0, 0]],
        "per_class_metrics": {
            "class_0": {"precision": 1.0, "recall": 1.0, "f1": 1.0},
            "class_1": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
        },
    }

    evaluate_pcam.log_evaluation_summary(
        checkpoint_path="checkpoints/pcam/best_model.pth",
        epoch=3,
        test_dataset_size=3,
        inference_time=0.1,
        test_metrics=test_metrics,
        output_dir=tmp_path,
        confusion_matrix_generated=True,
        roc_curve_generated=False,
        interpretability_summary=None,
    )

    joined = "\n".join(messages)
    assert "Confusion matrix:" in joined
    assert "ROC curve:" not in joined
