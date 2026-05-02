"""
Property-based tests for serialization round-trip preservation.

Feature: competitor-benchmark-system
Property 2: Serialization Round-Trip Preservation

Tests that BenchmarkConfig and TrainingResult instances can be serialized
to JSON and deserialized back without data loss.
"""

import json
from datetime import datetime
from pathlib import Path

import pytest
from hypothesis import given, settings, strategies as st

from experiments.benchmark_system.models import (
    BenchmarkConfig,
    TaskSpecification,
    TrainingResult,
)


# Hypothesis strategies for generating test data
@st.composite
def task_specification_strategy(draw):
    """Generate random TaskSpecification instances."""
    return TaskSpecification(
        dataset_name=draw(st.text(min_size=1, max_size=50)),
        data_root=Path(draw(st.text(min_size=1, max_size=100))),
        train_split=draw(st.floats(min_value=0.1, max_value=0.9)),
        val_split=draw(st.floats(min_value=0.05, max_value=0.2)),
        test_split=draw(st.floats(min_value=0.05, max_value=0.2)),
        model_architecture=draw(st.sampled_from(["resnet18", "resnet50", "vit"])),
        feature_dim=draw(st.integers(min_value=128, max_value=2048)),
        num_classes=draw(st.integers(min_value=2, max_value=10)),
        num_epochs=draw(st.integers(min_value=1, max_value=100)),
        batch_size=draw(st.integers(min_value=1, max_value=256)),
        learning_rate=draw(st.floats(min_value=1e-6, max_value=1e-2)),
        weight_decay=draw(st.floats(min_value=0.0, max_value=1e-3)),
        optimizer=draw(st.sampled_from(["Adam", "AdamW", "SGD"])),
        random_seed=draw(st.integers(min_value=0, max_value=10000)),
        augmentation_config=draw(st.dictionaries(
            st.text(min_size=1, max_size=20),
            st.one_of(st.booleans(), st.floats(), st.integers(), st.text())
        )),
        metrics=draw(st.lists(
            st.sampled_from(["accuracy", "auc", "f1", "precision", "recall"]),
            min_size=1,
            max_size=5,
            unique=True
        ))
    )


@st.composite
def benchmark_config_strategy(draw):
    """Generate random BenchmarkConfig instances."""
    # Ensure splits sum to approximately 1.0
    train_split = draw(st.floats(min_value=0.6, max_value=0.8))
    remaining = 1.0 - train_split
    val_split = draw(st.floats(min_value=0.1, max_value=remaining - 0.1))
    test_split = 1.0 - train_split - val_split
    
    task_spec = TaskSpecification(
        dataset_name=draw(st.text(min_size=1, max_size=50)),
        data_root=Path(draw(st.text(min_size=1, max_size=100))),
        train_split=train_split,
        val_split=val_split,
        test_split=test_split,
        model_architecture=draw(st.sampled_from(["resnet18", "resnet50", "vit"])),
        feature_dim=draw(st.integers(min_value=128, max_value=2048)),
        num_classes=draw(st.integers(min_value=2, max_value=10)),
        num_epochs=draw(st.integers(min_value=1, max_value=100)),
        batch_size=draw(st.integers(min_value=1, max_value=256)),
        learning_rate=draw(st.floats(min_value=1e-6, max_value=1e-2)),
        weight_decay=draw(st.floats(min_value=0.0, max_value=1e-3)),
        optimizer=draw(st.sampled_from(["Adam", "AdamW", "SGD"])),
        random_seed=draw(st.integers(min_value=0, max_value=10000)),
    )
    
    return BenchmarkConfig(
        mode=draw(st.sampled_from(["quick", "full"])),
        frameworks=draw(st.lists(
            st.sampled_from(["HistoCore", "PathML", "CLAM", "PyTorch"]),
            min_size=1,
            max_size=4,
            unique=True
        )),
        task_spec=task_spec,
        quick_mode_epochs=draw(st.integers(min_value=1, max_value=10)),
        quick_mode_samples=draw(st.integers(min_value=100, max_value=10000)),
        max_gpu_memory_mb=draw(st.integers(min_value=1000, max_value=24000)),
        max_gpu_temperature=draw(st.floats(min_value=60.0, max_value=90.0)),
        timeout_hours=draw(st.floats(min_value=1.0, max_value=100.0)),
        checkpoint_interval_minutes=draw(st.integers(min_value=1, max_value=120)),
        output_dir=Path(draw(st.text(min_size=1, max_value=100))),
        random_seed=draw(st.integers(min_value=0, max_value=10000)),
        bootstrap_samples=draw(st.integers(min_value=100, max_value=10000)),
        confidence_level=draw(st.floats(min_value=0.8, max_value=0.99)),
    )


@st.composite
def training_result_strategy(draw):
    """Generate random TrainingResult instances."""
    # Ensure splits sum to approximately 1.0
    train_split = draw(st.floats(min_value=0.6, max_value=0.8))
    remaining = 1.0 - train_split
    val_split = draw(st.floats(min_value=0.1, max_value=remaining - 0.1))
    test_split = 1.0 - train_split - val_split
    
    task_spec = TaskSpecification(
        dataset_name=draw(st.text(min_size=1, max_size=50)),
        data_root=Path(draw(st.text(min_size=1, max_size=100))),
        train_split=train_split,
        val_split=val_split,
        test_split=test_split,
        model_architecture=draw(st.sampled_from(["resnet18", "resnet50", "vit"])),
        feature_dim=draw(st.integers(min_value=128, max_value=2048)),
        num_classes=draw(st.integers(min_value=2, max_value=10)),
        num_epochs=draw(st.integers(min_value=1, max_value=100)),
        batch_size=draw(st.integers(min_value=1, max_value=256)),
        learning_rate=draw(st.floats(min_value=1e-6, max_value=1e-2)),
        weight_decay=draw(st.floats(min_value=0.0, max_value=1e-3)),
        optimizer=draw(st.sampled_from(["Adam", "AdamW", "SGD"])),
        random_seed=draw(st.integers(min_value=0, max_value=10000)),
    )
    
    # Generate confidence intervals
    accuracy = draw(st.floats(min_value=0.0, max_value=1.0))
    auc = draw(st.floats(min_value=0.0, max_value=1.0))
    f1 = draw(st.floats(min_value=0.0, max_value=1.0))
    
    accuracy_ci_lower = max(0.0, accuracy - draw(st.floats(min_value=0.0, max_value=0.1)))
    accuracy_ci_upper = min(1.0, accuracy + draw(st.floats(min_value=0.0, max_value=0.1)))
    
    auc_ci_lower = max(0.0, auc - draw(st.floats(min_value=0.0, max_value=0.1)))
    auc_ci_upper = min(1.0, auc + draw(st.floats(min_value=0.0, max_value=0.1)))
    
    f1_ci_lower = max(0.0, f1 - draw(st.floats(min_value=0.0, max_value=0.1)))
    f1_ci_upper = min(1.0, f1 + draw(st.floats(min_value=0.0, max_value=0.1)))
    
    return TrainingResult(
        framework_name=draw(st.sampled_from(["HistoCore", "PathML", "CLAM", "PyTorch"])),
        task_spec=task_spec,
        training_time_seconds=draw(st.floats(min_value=1.0, max_value=100000.0)),
        epochs_completed=draw(st.integers(min_value=1, max_value=100)),
        final_train_loss=draw(st.floats(min_value=0.0, max_value=10.0)),
        final_val_loss=draw(st.floats(min_value=0.0, max_value=10.0)),
        test_accuracy=accuracy,
        test_auc=auc,
        test_f1=f1,
        test_precision=draw(st.floats(min_value=0.0, max_value=1.0)),
        test_recall=draw(st.floats(min_value=0.0, max_value=1.0)),
        accuracy_ci=(accuracy_ci_lower, accuracy_ci_upper),
        auc_ci=(auc_ci_lower, auc_ci_upper),
        f1_ci=(f1_ci_lower, f1_ci_upper),
        peak_gpu_memory_mb=draw(st.floats(min_value=100.0, max_value=24000.0)),
        avg_gpu_utilization=draw(st.floats(min_value=0.0, max_value=100.0)),
        peak_gpu_temperature=draw(st.floats(min_value=30.0, max_value=90.0)),
        samples_per_second=draw(st.floats(min_value=1.0, max_value=10000.0)),
        inference_time_ms=draw(st.floats(min_value=0.1, max_value=1000.0)),
        model_parameters=draw(st.integers(min_value=1000, max_value=1000000000)),
        checkpoint_path=Path(draw(st.text(min_size=1, max_size=100))),
        metrics_path=Path(draw(st.text(min_size=1, max_size=100))),
        log_path=Path(draw(st.text(min_size=1, max_size=100))),
        status=draw(st.sampled_from(["success", "failed", "timeout"])),
        error_message=draw(st.one_of(st.none(), st.text(min_size=1, max_size=200))),
    )


def serialize_benchmark_config(config: BenchmarkConfig) -> str:
    """Serialize BenchmarkConfig to JSON string."""
    data = {
        "mode": config.mode,
        "frameworks": config.frameworks,
        "task_spec": {
            "dataset_name": config.task_spec.dataset_name if config.task_spec else None,
            "data_root": str(config.task_spec.data_root) if config.task_spec else None,
            "train_split": config.task_spec.train_split if config.task_spec else None,
            "val_split": config.task_spec.val_split if config.task_spec else None,
            "test_split": config.task_spec.test_split if config.task_spec else None,
            "model_architecture": config.task_spec.model_architecture if config.task_spec else None,
            "feature_dim": config.task_spec.feature_dim if config.task_spec else None,
            "num_classes": config.task_spec.num_classes if config.task_spec else None,
            "num_epochs": config.task_spec.num_epochs if config.task_spec else None,
            "batch_size": config.task_spec.batch_size if config.task_spec else None,
            "learning_rate": config.task_spec.learning_rate if config.task_spec else None,
            "weight_decay": config.task_spec.weight_decay if config.task_spec else None,
            "optimizer": config.task_spec.optimizer if config.task_spec else None,
            "random_seed": config.task_spec.random_seed if config.task_spec else None,
            "augmentation_config": config.task_spec.augmentation_config if config.task_spec else {},
            "metrics": config.task_spec.metrics if config.task_spec else [],
        } if config.task_spec else None,
        "quick_mode_epochs": config.quick_mode_epochs,
        "quick_mode_samples": config.quick_mode_samples,
        "max_gpu_memory_mb": config.max_gpu_memory_mb,
        "max_gpu_temperature": config.max_gpu_temperature,
        "timeout_hours": config.timeout_hours,
        "checkpoint_interval_minutes": config.checkpoint_interval_minutes,
        "output_dir": str(config.output_dir),
        "random_seed": config.random_seed,
        "bootstrap_samples": config.bootstrap_samples,
        "confidence_level": config.confidence_level,
    }
    return json.dumps(data)


def deserialize_benchmark_config(json_str: str) -> BenchmarkConfig:
    """Deserialize BenchmarkConfig from JSON string."""
    data = json.loads(json_str)
    
    task_spec = None
    if data["task_spec"]:
        task_spec = TaskSpecification(
            dataset_name=data["task_spec"]["dataset_name"],
            data_root=Path(data["task_spec"]["data_root"]),
            train_split=data["task_spec"]["train_split"],
            val_split=data["task_spec"]["val_split"],
            test_split=data["task_spec"]["test_split"],
            model_architecture=data["task_spec"]["model_architecture"],
            feature_dim=data["task_spec"]["feature_dim"],
            num_classes=data["task_spec"]["num_classes"],
            num_epochs=data["task_spec"]["num_epochs"],
            batch_size=data["task_spec"]["batch_size"],
            learning_rate=data["task_spec"]["learning_rate"],
            weight_decay=data["task_spec"]["weight_decay"],
            optimizer=data["task_spec"]["optimizer"],
            random_seed=data["task_spec"]["random_seed"],
            augmentation_config=data["task_spec"]["augmentation_config"],
            metrics=data["task_spec"]["metrics"],
        )
    
    return BenchmarkConfig(
        mode=data["mode"],
        frameworks=data["frameworks"],
        task_spec=task_spec,
        quick_mode_epochs=data["quick_mode_epochs"],
        quick_mode_samples=data["quick_mode_samples"],
        max_gpu_memory_mb=data["max_gpu_memory_mb"],
        max_gpu_temperature=data["max_gpu_temperature"],
        timeout_hours=data["timeout_hours"],
        checkpoint_interval_minutes=data["checkpoint_interval_minutes"],
        output_dir=Path(data["output_dir"]),
        random_seed=data["random_seed"],
        bootstrap_samples=data["bootstrap_samples"],
        confidence_level=data["confidence_level"],
    )


def serialize_training_result(result: TrainingResult) -> str:
    """Serialize TrainingResult to JSON string."""
    data = {
        "framework_name": result.framework_name,
        "task_spec": {
            "dataset_name": result.task_spec.dataset_name,
            "data_root": str(result.task_spec.data_root),
            "train_split": result.task_spec.train_split,
            "val_split": result.task_spec.val_split,
            "test_split": result.task_spec.test_split,
            "model_architecture": result.task_spec.model_architecture,
            "feature_dim": result.task_spec.feature_dim,
            "num_classes": result.task_spec.num_classes,
            "num_epochs": result.task_spec.num_epochs,
            "batch_size": result.task_spec.batch_size,
            "learning_rate": result.task_spec.learning_rate,
            "weight_decay": result.task_spec.weight_decay,
            "optimizer": result.task_spec.optimizer,
            "random_seed": result.task_spec.random_seed,
            "augmentation_config": result.task_spec.augmentation_config,
            "metrics": result.task_spec.metrics,
        },
        "training_time_seconds": result.training_time_seconds,
        "epochs_completed": result.epochs_completed,
        "final_train_loss": result.final_train_loss,
        "final_val_loss": result.final_val_loss,
        "test_accuracy": result.test_accuracy,
        "test_auc": result.test_auc,
        "test_f1": result.test_f1,
        "test_precision": result.test_precision,
        "test_recall": result.test_recall,
        "accuracy_ci": list(result.accuracy_ci),
        "auc_ci": list(result.auc_ci),
        "f1_ci": list(result.f1_ci),
        "peak_gpu_memory_mb": result.peak_gpu_memory_mb,
        "avg_gpu_utilization": result.avg_gpu_utilization,
        "peak_gpu_temperature": result.peak_gpu_temperature,
        "samples_per_second": result.samples_per_second,
        "inference_time_ms": result.inference_time_ms,
        "model_parameters": result.model_parameters,
        "checkpoint_path": str(result.checkpoint_path),
        "metrics_path": str(result.metrics_path),
        "log_path": str(result.log_path),
        "status": result.status,
        "error_message": result.error_message,
    }
    return json.dumps(data)


def deserialize_training_result(json_str: str) -> TrainingResult:
    """Deserialize TrainingResult from JSON string."""
    data = json.loads(json_str)
    
    task_spec = TaskSpecification(
        dataset_name=data["task_spec"]["dataset_name"],
        data_root=Path(data["task_spec"]["data_root"]),
        train_split=data["task_spec"]["train_split"],
        val_split=data["task_spec"]["val_split"],
        test_split=data["task_spec"]["test_split"],
        model_architecture=data["task_spec"]["model_architecture"],
        feature_dim=data["task_spec"]["feature_dim"],
        num_classes=data["task_spec"]["num_classes"],
        num_epochs=data["task_spec"]["num_epochs"],
        batch_size=data["task_spec"]["batch_size"],
        learning_rate=data["task_spec"]["learning_rate"],
        weight_decay=data["task_spec"]["weight_decay"],
        optimizer=data["task_spec"]["optimizer"],
        random_seed=data["task_spec"]["random_seed"],
        augmentation_config=data["task_spec"]["augmentation_config"],
        metrics=data["task_spec"]["metrics"],
    )
    
    return TrainingResult(
        framework_name=data["framework_name"],
        task_spec=task_spec,
        training_time_seconds=data["training_time_seconds"],
        epochs_completed=data["epochs_completed"],
        final_train_loss=data["final_train_loss"],
        final_val_loss=data["final_val_loss"],
        test_accuracy=data["test_accuracy"],
        test_auc=data["test_auc"],
        test_f1=data["test_f1"],
        test_precision=data["test_precision"],
        test_recall=data["test_recall"],
        accuracy_ci=tuple(data["accuracy_ci"]),
        auc_ci=tuple(data["auc_ci"]),
        f1_ci=tuple(data["f1_ci"]),
        peak_gpu_memory_mb=data["peak_gpu_memory_mb"],
        avg_gpu_utilization=data["avg_gpu_utilization"],
        peak_gpu_temperature=data["peak_gpu_temperature"],
        samples_per_second=data["samples_per_second"],
        inference_time_ms=data["inference_time_ms"],
        model_parameters=data["model_parameters"],
        checkpoint_path=Path(data["checkpoint_path"]),
        metrics_path=Path(data["metrics_path"]),
        log_path=Path(data["log_path"]),
        status=data["status"],
        error_message=data["error_message"],
    )


# Feature: competitor-benchmark-system, Property 2: Serialization Round-Trip Preservation
@settings(max_examples=100)
@given(config=benchmark_config_strategy())
def test_benchmark_config_serialization_roundtrip(config):
    """
    Property: Serialization Round-Trip Preservation for BenchmarkConfig
    
    For any BenchmarkConfig instance, serializing to JSON and deserializing
    SHALL produce an equivalent object with all data preserved.
    
    Validates: Requirements 4.9, 9.8
    """
    # Serialize
    json_str = serialize_benchmark_config(config)
    
    # Deserialize
    restored_config = deserialize_benchmark_config(json_str)
    
    # Verify equivalence
    assert restored_config.mode == config.mode
    assert restored_config.frameworks == config.frameworks
    assert restored_config.quick_mode_epochs == config.quick_mode_epochs
    assert restored_config.quick_mode_samples == config.quick_mode_samples
    assert restored_config.max_gpu_memory_mb == config.max_gpu_memory_mb
    assert restored_config.max_gpu_temperature == config.max_gpu_temperature
    assert restored_config.timeout_hours == config.timeout_hours
    assert restored_config.checkpoint_interval_minutes == config.checkpoint_interval_minutes
    assert restored_config.output_dir == config.output_dir
    assert restored_config.random_seed == config.random_seed
    assert restored_config.bootstrap_samples == config.bootstrap_samples
    assert restored_config.confidence_level == config.confidence_level
    
    # Verify task_spec if present
    if config.task_spec:
        assert restored_config.task_spec is not None
        assert restored_config.task_spec.dataset_name == config.task_spec.dataset_name
        assert restored_config.task_spec.data_root == config.task_spec.data_root
        assert restored_config.task_spec.train_split == config.task_spec.train_split
        assert restored_config.task_spec.val_split == config.task_spec.val_split
        assert restored_config.task_spec.test_split == config.task_spec.test_split
        assert restored_config.task_spec.model_architecture == config.task_spec.model_architecture
        assert restored_config.task_spec.feature_dim == config.task_spec.feature_dim
        assert restored_config.task_spec.num_classes == config.task_spec.num_classes
        assert restored_config.task_spec.num_epochs == config.task_spec.num_epochs
        assert restored_config.task_spec.batch_size == config.task_spec.batch_size
        assert restored_config.task_spec.learning_rate == config.task_spec.learning_rate
        assert restored_config.task_spec.weight_decay == config.task_spec.weight_decay
        assert restored_config.task_spec.optimizer == config.task_spec.optimizer
        assert restored_config.task_spec.random_seed == config.task_spec.random_seed
        assert restored_config.task_spec.augmentation_config == config.task_spec.augmentation_config
        assert restored_config.task_spec.metrics == config.task_spec.metrics


# Feature: competitor-benchmark-system, Property 2: Serialization Round-Trip Preservation
@settings(max_examples=100)
@given(result=training_result_strategy())
def test_training_result_serialization_roundtrip(result):
    """
    Property: Serialization Round-Trip Preservation for TrainingResult
    
    For any TrainingResult instance, serializing to JSON and deserializing
    SHALL produce an equivalent object with all data preserved.
    
    Validates: Requirements 4.9, 9.8
    """
    # Serialize
    json_str = serialize_training_result(result)
    
    # Deserialize
    restored_result = deserialize_training_result(json_str)
    
    # Verify equivalence
    assert restored_result.framework_name == result.framework_name
    assert restored_result.training_time_seconds == result.training_time_seconds
    assert restored_result.epochs_completed == result.epochs_completed
    assert restored_result.final_train_loss == result.final_train_loss
    assert restored_result.final_val_loss == result.final_val_loss
    assert restored_result.test_accuracy == result.test_accuracy
    assert restored_result.test_auc == result.test_auc
    assert restored_result.test_f1 == result.test_f1
    assert restored_result.test_precision == result.test_precision
    assert restored_result.test_recall == result.test_recall
    assert restored_result.accuracy_ci == result.accuracy_ci
    assert restored_result.auc_ci == result.auc_ci
    assert restored_result.f1_ci == result.f1_ci
    assert restored_result.peak_gpu_memory_mb == result.peak_gpu_memory_mb
    assert restored_result.avg_gpu_utilization == result.avg_gpu_utilization
    assert restored_result.peak_gpu_temperature == result.peak_gpu_temperature
    assert restored_result.samples_per_second == result.samples_per_second
    assert restored_result.inference_time_ms == result.inference_time_ms
    assert restored_result.model_parameters == result.model_parameters
    assert restored_result.checkpoint_path == result.checkpoint_path
    assert restored_result.metrics_path == result.metrics_path
    assert restored_result.log_path == result.log_path
    assert restored_result.status == result.status
    assert restored_result.error_message == result.error_message
    
    # Verify task_spec
    assert restored_result.task_spec.dataset_name == result.task_spec.dataset_name
    assert restored_result.task_spec.data_root == result.task_spec.data_root
    assert restored_result.task_spec.train_split == result.task_spec.train_split
    assert restored_result.task_spec.val_split == result.task_spec.val_split
    assert restored_result.task_spec.test_split == result.task_spec.test_split
    assert restored_result.task_spec.model_architecture == result.task_spec.model_architecture
    assert restored_result.task_spec.feature_dim == result.task_spec.feature_dim
    assert restored_result.task_spec.num_classes == result.task_spec.num_classes
    assert restored_result.task_spec.num_epochs == result.task_spec.num_epochs
    assert restored_result.task_spec.batch_size == result.task_spec.batch_size
    assert restored_result.task_spec.learning_rate == result.task_spec.learning_rate
    assert restored_result.task_spec.weight_decay == result.task_spec.weight_decay
    assert restored_result.task_spec.optimizer == result.task_spec.optimizer
    assert restored_result.task_spec.random_seed == result.task_spec.random_seed
    assert restored_result.task_spec.augmentation_config == result.task_spec.augmentation_config
    assert restored_result.task_spec.metrics == result.task_spec.metrics
