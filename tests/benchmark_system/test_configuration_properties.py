"""
Property-based tests for configuration equivalence across frameworks.

Feature: competitor-benchmark-system
Property 1: Configuration Equivalence Across Frameworks

Tests that all frameworks receive equivalent configurations when configured
from the same TaskSpecification, ensuring fair comparisons.
"""

from pathlib import Path

import pytest
from hypothesis import given, settings, strategies as st

from experiments.benchmark_system.models import TaskSpecification
from experiments.benchmark_system.task_executor import TrainingTaskExecutor


# Hypothesis strategies for generating test data
@st.composite
def task_specification_strategy(draw):
    """Generate random TaskSpecification instances with valid splits."""
    # Ensure splits sum to approximately 1.0
    # Use fixed splits to avoid floating-point precision issues
    split_choice = draw(st.sampled_from([
        (0.8, 0.1, 0.1),
        (0.7, 0.2, 0.1),
        (0.7, 0.15, 0.15),
        (0.6, 0.2, 0.2),
        (0.75, 0.15, 0.1),
    ]))
    train_split, val_split, test_split = split_choice
    
    return TaskSpecification(
        dataset_name=draw(st.sampled_from(["PatchCamelyon", "Camelyon16", "TCGA"])),
        data_root=Path(draw(st.text(min_size=1, max_size=50))),
        train_split=train_split,
        val_split=val_split,
        test_split=test_split,
        model_architecture=draw(st.sampled_from([
            "resnet18_transformer",
            "resnet50_transformer",
            "vit"
        ])),
        feature_dim=draw(st.sampled_from([256, 512, 1024, 2048])),
        num_classes=draw(st.integers(min_value=2, max_value=10)),
        num_epochs=draw(st.integers(min_value=1, max_value=100)),
        batch_size=draw(st.sampled_from([8, 16, 32, 64, 128])),
        learning_rate=draw(st.floats(min_value=1e-6, max_value=1e-2)),
        weight_decay=draw(st.floats(min_value=0.0, max_value=1e-3)),
        optimizer=draw(st.sampled_from(["Adam", "AdamW", "SGD"])),
        random_seed=draw(st.integers(min_value=0, max_value=10000)),
        augmentation_config=draw(st.dictionaries(
            st.sampled_from(["flip", "rotate", "color_jitter", "normalize"]),
            st.one_of(st.booleans(), st.floats(min_value=0.0, max_value=1.0)),
            min_size=0,
            max_size=4
        )),
        metrics=draw(st.lists(
            st.sampled_from(["accuracy", "auc", "f1", "precision", "recall"]),
            min_size=1,
            max_size=5,
            unique=True
        ))
    )


# Feature: competitor-benchmark-system, Property 1: Configuration Equivalence Across Frameworks
@settings(max_examples=100)
@given(task_spec=task_specification_strategy())
def test_configuration_equivalence_random_seeds(task_spec):
    """
    Property: All frameworks receive identical random seeds.
    
    For any TaskSpecification, when configured for multiple frameworks,
    all frameworks SHALL receive the same random seed value.
    
    Validates: Requirement 2.2
    """
    executor = TrainingTaskExecutor()
    
    # Configure for all frameworks
    frameworks = ["HistoCore", "PathML", "CLAM", "PyTorch"]
    configs = [executor.configure_task(task_spec, fw) for fw in frameworks]
    
    # Verify all have the same random seed
    reference_seed = configs[0].random_seed
    for config in configs:
        assert config.random_seed == reference_seed, (
            f"Random seed mismatch: {config.framework_name} has seed "
            f"{config.random_seed}, expected {reference_seed}"
        )
    
    # Verify seed matches task specification
    assert reference_seed == task_spec.random_seed, (
        f"Random seed {reference_seed} does not match task spec seed "
        f"{task_spec.random_seed}"
    )


# Feature: competitor-benchmark-system, Property 1: Configuration Equivalence Across Frameworks
@settings(max_examples=100)
@given(task_spec=task_specification_strategy())
def test_configuration_equivalence_data_splits(task_spec):
    """
    Property: All frameworks receive identical data splits.
    
    For any TaskSpecification, when configured for multiple frameworks,
    all frameworks SHALL receive the same data split configuration
    (train/val/test ratios and data split hash).
    
    Validates: Requirement 2.3
    """
    executor = TrainingTaskExecutor()
    
    # Configure for all frameworks
    frameworks = ["HistoCore", "PathML", "CLAM", "PyTorch"]
    configs = [executor.configure_task(task_spec, fw) for fw in frameworks]
    
    # Verify all have the same data split hash
    reference_hash = configs[0].data_split_hash
    for config in configs:
        assert config.data_split_hash == reference_hash, (
            f"Data split hash mismatch: {config.framework_name} has hash "
            f"{config.data_split_hash}, expected {reference_hash}"
        )
    
    # Verify all have the same split ratios
    for config in configs:
        assert config.task_spec.train_split == task_spec.train_split, (
            f"Train split mismatch: {config.framework_name} has "
            f"{config.task_spec.train_split}, expected {task_spec.train_split}"
        )
        assert config.task_spec.val_split == task_spec.val_split, (
            f"Val split mismatch: {config.framework_name} has "
            f"{config.task_spec.val_split}, expected {task_spec.val_split}"
        )
        assert config.task_spec.test_split == task_spec.test_split, (
            f"Test split mismatch: {config.framework_name} has "
            f"{config.task_spec.test_split}, expected {task_spec.test_split}"
        )


# Feature: competitor-benchmark-system, Property 1: Configuration Equivalence Across Frameworks
@settings(max_examples=100)
@given(task_spec=task_specification_strategy())
def test_configuration_equivalence_model_parameters(task_spec):
    """
    Property: All frameworks receive equivalent model architecture parameters.
    
    For any TaskSpecification, when configured for multiple frameworks,
    all frameworks SHALL receive the same feature dimension and number of classes.
    Model architecture names may differ due to framework-specific translations.
    
    Validates: Requirement 2.4
    """
    executor = TrainingTaskExecutor()
    
    # Configure for all frameworks
    frameworks = ["HistoCore", "PathML", "CLAM", "PyTorch"]
    configs = [executor.configure_task(task_spec, fw) for fw in frameworks]
    
    # Verify all have the same feature dimension
    for config in configs:
        assert config.task_spec.feature_dim == task_spec.feature_dim, (
            f"Feature dimension mismatch: {config.framework_name} has "
            f"{config.task_spec.feature_dim}, expected {task_spec.feature_dim}"
        )
    
    # Verify all have the same number of classes
    for config in configs:
        assert config.task_spec.num_classes == task_spec.num_classes, (
            f"Number of classes mismatch: {config.framework_name} has "
            f"{config.task_spec.num_classes}, expected {task_spec.num_classes}"
        )


# Feature: competitor-benchmark-system, Property 1: Configuration Equivalence Across Frameworks
@settings(max_examples=100)
@given(task_spec=task_specification_strategy())
def test_configuration_equivalence_augmentation(task_spec):
    """
    Property: All frameworks receive identical augmentation pipelines.
    
    For any TaskSpecification, when configured for multiple frameworks,
    all frameworks SHALL receive the same augmentation configuration.
    
    Validates: Requirement 2.5
    """
    executor = TrainingTaskExecutor()
    
    # Configure for all frameworks
    frameworks = ["HistoCore", "PathML", "CLAM", "PyTorch"]
    configs = [executor.configure_task(task_spec, fw) for fw in frameworks]
    
    # Verify all have the same augmentation config
    for config in configs:
        assert config.task_spec.augmentation_config == task_spec.augmentation_config, (
            f"Augmentation config mismatch: {config.framework_name} has "
            f"{config.task_spec.augmentation_config}, expected "
            f"{task_spec.augmentation_config}"
        )


# Feature: competitor-benchmark-system, Property 1: Configuration Equivalence Across Frameworks
@settings(max_examples=100)
@given(task_spec=task_specification_strategy())
def test_configuration_equivalence_hyperparameters(task_spec):
    """
    Property: All frameworks receive identical optimizer hyperparameters.
    
    For any TaskSpecification, when configured for multiple frameworks,
    all frameworks SHALL receive the same learning rate, weight decay,
    batch size, and number of epochs. Optimizer names may differ due to
    framework-specific translations.
    
    Validates: Requirement 2.6
    """
    executor = TrainingTaskExecutor()
    
    # Configure for all frameworks
    frameworks = ["HistoCore", "PathML", "CLAM", "PyTorch"]
    configs = [executor.configure_task(task_spec, fw) for fw in frameworks]
    
    # Verify all have the same learning rate
    for config in configs:
        assert config.task_spec.learning_rate == task_spec.learning_rate, (
            f"Learning rate mismatch: {config.framework_name} has "
            f"{config.task_spec.learning_rate}, expected {task_spec.learning_rate}"
        )
    
    # Verify all have the same weight decay
    for config in configs:
        assert config.task_spec.weight_decay == task_spec.weight_decay, (
            f"Weight decay mismatch: {config.framework_name} has "
            f"{config.task_spec.weight_decay}, expected {task_spec.weight_decay}"
        )
    
    # Verify all have the same batch size
    for config in configs:
        assert config.task_spec.batch_size == task_spec.batch_size, (
            f"Batch size mismatch: {config.framework_name} has "
            f"{config.task_spec.batch_size}, expected {task_spec.batch_size}"
        )
    
    # Verify all have the same number of epochs
    for config in configs:
        assert config.task_spec.num_epochs == task_spec.num_epochs, (
            f"Number of epochs mismatch: {config.framework_name} has "
            f"{config.task_spec.num_epochs}, expected {task_spec.num_epochs}"
        )


# Feature: competitor-benchmark-system, Property 1: Configuration Equivalence Across Frameworks
@settings(max_examples=100)
@given(task_spec=task_specification_strategy())
def test_configuration_equivalence_validation(task_spec):
    """
    Property: validate_equivalence() correctly identifies equivalent configurations.
    
    For any TaskSpecification, when configured for multiple frameworks,
    the validate_equivalence() method SHALL report that all configurations
    are equivalent (no critical differences).
    
    Validates: Requirements 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 3.7
    """
    executor = TrainingTaskExecutor()
    
    # Configure for all frameworks
    frameworks = ["HistoCore", "PathML", "CLAM", "PyTorch"]
    configs = [executor.configure_task(task_spec, fw) for fw in frameworks]
    
    # Validate equivalence
    report = executor.validate_equivalence(configs)
    
    # Should have no critical differences
    # (warnings are acceptable for framework-specific translations)
    assert report.is_equivalent, (
        f"Configurations should be equivalent, but found differences: "
        f"{report.differences}"
    )
    
    # Verify all random seeds are tracked and identical
    assert len(report.random_seeds) == len(frameworks)
    seed_values = list(report.random_seeds.values())
    assert all(seed == seed_values[0] for seed in seed_values), (
        f"Random seeds differ: {report.random_seeds}"
    )
    
    # Verify all data splits are tracked and identical
    assert len(report.data_splits) == len(frameworks)
    split_values = list(report.data_splits.values())
    assert all(split == split_values[0] for split in split_values), (
        f"Data splits differ: {report.data_splits}"
    )
    
    # Verify all hyperparameters are tracked
    assert len(report.hyperparameters) == len(frameworks)
    for fw, params in report.hyperparameters.items():
        assert params["num_epochs"] == task_spec.num_epochs
        assert params["batch_size"] == task_spec.batch_size
        assert params["learning_rate"] == task_spec.learning_rate
        assert params["weight_decay"] == task_spec.weight_decay


# Feature: competitor-benchmark-system, Property 1: Configuration Equivalence Across Frameworks
@settings(max_examples=50)
@given(
    task_spec=task_specification_strategy(),
    modified_seed=st.integers(min_value=0, max_value=10000)
)
def test_configuration_equivalence_detects_seed_mismatch(task_spec, modified_seed):
    """
    Property: validate_equivalence() detects random seed mismatches.
    
    When configurations have different random seeds, validate_equivalence()
    SHALL report that configurations are not equivalent and identify the
    seed mismatch.
    
    Validates: Requirement 2.2
    """
    # Skip if modified seed happens to match original
    if modified_seed == task_spec.random_seed:
        return
    
    executor = TrainingTaskExecutor()
    
    # Configure for two frameworks
    config1 = executor.configure_task(task_spec, "HistoCore")
    
    # Create modified task spec with different seed
    modified_spec = TaskSpecification(
        dataset_name=task_spec.dataset_name,
        data_root=task_spec.data_root,
        train_split=task_spec.train_split,
        val_split=task_spec.val_split,
        test_split=task_spec.test_split,
        model_architecture=task_spec.model_architecture,
        feature_dim=task_spec.feature_dim,
        num_classes=task_spec.num_classes,
        num_epochs=task_spec.num_epochs,
        batch_size=task_spec.batch_size,
        learning_rate=task_spec.learning_rate,
        weight_decay=task_spec.weight_decay,
        optimizer=task_spec.optimizer,
        random_seed=modified_seed,  # Different seed
        augmentation_config=task_spec.augmentation_config,
        metrics=task_spec.metrics,
    )
    config2 = executor.configure_task(modified_spec, "PathML")
    
    # Validate equivalence
    report = executor.validate_equivalence([config1, config2])
    
    # Should detect the mismatch
    assert not report.is_equivalent, (
        "Should detect random seed mismatch"
    )
    
    # Should have a difference about random seeds
    seed_diff_found = any("seed" in diff.lower() for diff in report.differences)
    assert seed_diff_found, (
        f"Should report random seed difference, but got: {report.differences}"
    )


# Feature: competitor-benchmark-system, Property 1: Configuration Equivalence Across Frameworks
@settings(max_examples=50)
@given(
    task_spec=task_specification_strategy(),
    modified_lr=st.floats(min_value=1e-6, max_value=1e-2)
)
def test_configuration_equivalence_detects_hyperparameter_mismatch(task_spec, modified_lr):
    """
    Property: validate_equivalence() detects hyperparameter mismatches.
    
    When configurations have different hyperparameters (e.g., learning rate),
    validate_equivalence() SHALL report that configurations are not equivalent
    and identify the hyperparameter mismatch.
    
    Validates: Requirement 2.6
    """
    # Skip if modified learning rate happens to match original
    if abs(modified_lr - task_spec.learning_rate) < 1e-9:
        return
    
    executor = TrainingTaskExecutor()
    
    # Configure for two frameworks
    config1 = executor.configure_task(task_spec, "HistoCore")
    
    # Create modified task spec with different learning rate
    modified_spec = TaskSpecification(
        dataset_name=task_spec.dataset_name,
        data_root=task_spec.data_root,
        train_split=task_spec.train_split,
        val_split=task_spec.val_split,
        test_split=task_spec.test_split,
        model_architecture=task_spec.model_architecture,
        feature_dim=task_spec.feature_dim,
        num_classes=task_spec.num_classes,
        num_epochs=task_spec.num_epochs,
        batch_size=task_spec.batch_size,
        learning_rate=modified_lr,  # Different learning rate
        weight_decay=task_spec.weight_decay,
        optimizer=task_spec.optimizer,
        random_seed=task_spec.random_seed,
        augmentation_config=task_spec.augmentation_config,
        metrics=task_spec.metrics,
    )
    config2 = executor.configure_task(modified_spec, "PathML")
    
    # Validate equivalence
    report = executor.validate_equivalence([config1, config2])
    
    # Should detect the mismatch
    assert not report.is_equivalent, (
        "Should detect learning rate mismatch"
    )
    
    # Should have a difference about learning rate
    lr_diff_found = any("learning rate" in diff.lower() for diff in report.differences)
    assert lr_diff_found, (
        f"Should report learning rate difference, but got: {report.differences}"
    )
