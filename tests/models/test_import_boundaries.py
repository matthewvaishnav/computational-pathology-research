"""Regression tests for model package import boundaries."""


def test_adaptive_pruning_import_does_not_cycle() -> None:
    """Adaptive-pruning imports must not recursively initialize TransnnMIL."""
    from src.models.transnnmil.adaptive_pruning import (
        AdaptivePruning,
        ImportanceScorer,
        PrunedTransMIL,
    )

    assert AdaptivePruning is not None
    assert ImportanceScorer is not None
    assert PrunedTransMIL is not None
