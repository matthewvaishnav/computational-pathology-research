"""Regression tests for the ONNX exporter."""

import torch

from scripts.export_onnx import ONNXExporter
from src.models.heads import ClassificationHead
from src.models.multimodal import MultimodalFusionModel


def test_exporter_loads_state_dict_only_checkpoint(tmp_path):
    """Exporter should load the standard backbone-only training checkpoint path."""
    checkpoint_path = tmp_path / "backbone_only.pth"
    output_path = tmp_path / "model.onnx"

    model = MultimodalFusionModel()
    torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)

    exporter = ONNXExporter(
        checkpoint_path=str(checkpoint_path),
        output_path=str(output_path),
        batch_size=2,
        wsi_num_patches=12,
        clinical_seq_len=64,
        verbose=False,
    )

    export_model = exporter.load_model()
    dummy_inputs = exporter.create_dummy_inputs()

    embeddings = export_model(*dummy_inputs)

    assert exporter.output_names == ["embeddings"]
    assert embeddings.shape == (2, model.get_embedding_dim())
    assert exporter.model_config["genomic_config"]["hidden_dims"] == [1024, 512]
    assert exporter.model_config["clinical_config"]["max_seq_length"] == 512


def test_exporter_loads_checkpoint_with_task_head(tmp_path):
    """Exporter should reconstruct the task head from checkpoint weights."""
    checkpoint_path = tmp_path / "full_checkpoint.pth"
    output_path = tmp_path / "model.onnx"

    model = MultimodalFusionModel(embed_dim=64)
    head = ClassificationHead(input_dim=64, hidden_dim=32, num_classes=3)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "task_head_state_dict": head.state_dict(),
        },
        checkpoint_path,
    )

    exporter = ONNXExporter(
        checkpoint_path=str(checkpoint_path),
        output_path=str(output_path),
        batch_size=3,
        wsi_num_patches=10,
        clinical_seq_len=32,
        verbose=False,
    )

    export_model = exporter.load_model()
    dummy_inputs = exporter.create_dummy_inputs()

    logits, embeddings = export_model(*dummy_inputs)

    assert exporter.output_names == ["logits", "embeddings"]
    assert logits.shape == (3, 3)
    assert embeddings.shape == (3, 64)


def test_exporter_clamps_dummy_sequence_length_to_model_max(tmp_path):
    """Dummy clinical sequence length should not exceed the model positional encoding."""
    checkpoint_path = tmp_path / "checkpoint.pth"
    output_path = tmp_path / "model.onnx"

    model = MultimodalFusionModel()
    torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)

    exporter = ONNXExporter(
        checkpoint_path=str(checkpoint_path),
        output_path=str(output_path),
        clinical_seq_len=2048,
        verbose=False,
    )
    exporter.load_model()

    _, _, _, clinical_text, clinical_mask = exporter.create_dummy_inputs()

    assert clinical_text.shape[1] == 512
    assert clinical_mask.shape[1] == 512
