"""Property-based tests for failure analysis module."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hypothesis import assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as npst
from src.interpretability.failure_analysis import FailureAnalyzer


# Strategy for generating valid predictions and ground truth
@st.composite
def predictions_and_labels(draw, min_samples=1, max_samples=100):
    """Generate predictions, ground truth, and confidence scores."""
    n_samples = draw(st.integers(min_value=min_samples, max_value=max_samples))
    n_classes = draw(st.integers(min_value=2, max_value=10))

    predictions = draw(
        npst.arrays(
            dtype=np.int64,
            shape=n_samples,
            elements=st.integers(min_value=0, max_value=n_classes - 1),
        )
    )

    ground_truth = draw(
        npst.arrays(
            dtype=np.int64,
            shape=n_samples,
            elements=st.integers(min_value=0, max_value=n_classes - 1),
        )
    )

    confidence_scores = draw(
        npst.arrays(
            dtype=np.float32,
            shape=n_samples,
            elements=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        )
    )

    slide_ids = [f"slide_{i:04d}" for i in range(n_samples)]

    return predictions, ground_truth, confidence_scores, slide_ids


@st.composite
def failure_embeddings_and_metadata(draw, min_failures=1, max_failures=50):
    """Generate failure embeddings and metadata."""
    n_failures = draw(st.integers(min_value=min_failures, max_value=max_failures))
    embedding_dim = draw(st.integers(min_value=8, max_value=256))

    embeddings = draw(
        npst.arrays(
            dtype=np.float32,
            shape=(n_failures, embedding_dim),
            elements=st.floats(
                min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False
            ),
        )
    )

    # Create metadata DataFrame
    metadata = pd.DataFrame(
        {
            "slide_id": [f"slide_{i:04d}" for i in range(n_failures)],
            "prediction": np.random.randint(0, 2, n_failures),
            "ground_truth": np.random.randint(0, 2, n_failures),
            "confidence": np.random.uniform(0.0, 1.0, n_failures),
            "is_failure": [True] * n_failures,
        }
    )

    return embeddings, metadata


# Property 10: Failure Identification with Confidence
@given(data=predictions_and_labels())
@settings(max_examples=100, deadline=None)
def test_property_10_failure_identification_with_confidence(data):
    """
    Feature: model-interpretability, Property 10: Failure Identification with Confidence

    For any predictions and ground truth labels, all identified failure cases
    (where prediction ≠ ground truth) SHALL have associated confidence scores.

    Validates: Requirements 3.1, 3.2
    """
    predictions, ground_truth, confidence_scores, slide_ids = data

    analyzer = FailureAnalyzer()

    # Identify failures
    result = analyzer.identify_failures(
        predictions=predictions,
        ground_truth=ground_truth,
        confidence_scores=confidence_scores,
        slide_ids=slide_ids,
    )

    # Filter to only failures
    failures = result[result["is_failure"]]

    # Property: All failures must have confidence scores
    if len(failures) > 0:
        # Check that confidence column exists
        assert "confidence" in failures.columns, "Failures must have 'confidence' column"

        # Check that all confidence values are present (not NaN)
        assert not failures["confidence"].isna().any(), "All failures must have confidence scores"

        # Check that confidence values are in valid range [0, 1]
        assert (failures["confidence"] >= 0.0).all(), "Confidence scores must be >= 0"
        assert (failures["confidence"] <= 1.0).all(), "Confidence scores must be <= 1"

        # Verify failures are correctly identified (prediction != ground_truth)
        for _, row in failures.iterrows():
            assert (
                row["prediction"] != row["ground_truth"]
            ), f"Failure case must have prediction != ground_truth"


# Property 11: Failure Clustering Completeness
@given(data=failure_embeddings_and_metadata(min_failures=2, max_failures=50))
@settings(max_examples=100, deadline=None)
def test_property_11_failure_clustering_completeness(data):
    """
    Feature: model-interpretability, Property 11: Failure Clustering Completeness

    For any failure embeddings, clustering SHALL assign each failure to exactly
    one cluster, and both visualizations and statistics SHALL be generated for
    all clusters.

    Validates: Requirements 3.3, 3.4, 3.5
    """
    embeddings, metadata = data
    n_failures = len(embeddings)

    # Test with different clustering methods
    for method in ["kmeans", "dbscan", "hierarchical"]:
        analyzer = FailureAnalyzer(
            clustering_method=method,
            n_clusters=min(5, n_failures),  # Ensure n_clusters <= n_samples
        )

        # Cluster failures
        clustered = analyzer.cluster_failures(embeddings, metadata)

        # Property 1: Each failure assigned to exactly one cluster
        assert "cluster_id" in clustered.columns, "Result must have 'cluster_id' column"
        assert len(clustered) == n_failures, "All failures must be in result"
        assert not clustered["cluster_id"].isna().any(), "All failures must have cluster assignment"

        # Property 2: Cluster IDs are valid integers
        assert clustered["cluster_id"].dtype in [
            np.int32,
            np.int64,
            int,
        ], "Cluster IDs must be integers"

        # Property 3: Statistics generated for all clusters
        cluster_stats = analyzer.analyze_cluster_characteristics(clustered)

        unique_clusters = set(clustered["cluster_id"].unique())
        stats_clusters = set(cluster_stats.keys())

        assert unique_clusters == stats_clusters, (
            f"Statistics must be generated for all clusters. "
            f"Found clusters: {unique_clusters}, Stats for: {stats_clusters}"
        )

        # Property 4: Each cluster has required statistics
        for cluster_id, stats in cluster_stats.items():
            assert "count" in stats, f"Cluster {cluster_id} must have 'count'"
            assert "avg_confidence" in stats, f"Cluster {cluster_id} must have 'avg_confidence'"
            assert (
                "representative_samples" in stats
            ), f"Cluster {cluster_id} must have 'representative_samples'"

            # Verify count matches actual cluster size
            actual_count = (clustered["cluster_id"] == cluster_id).sum()
            assert (
                stats["count"] == actual_count
            ), f"Cluster {cluster_id} count mismatch: stats={stats['count']}, actual={actual_count}"


# Property 13: Systematic Bias Analysis Completeness
@given(
    data=predictions_and_labels(min_samples=10, max_samples=100),
    n_subgroups=st.integers(min_value=2, max_value=5),
)
@settings(max_examples=100, deadline=None)
def test_property_13_systematic_bias_analysis_completeness(data, n_subgroups):
    """
    Feature: model-interpretability, Property 13: Systematic Bias Analysis Completeness

    For any failure cases and clinical subgroups, bias metrics SHALL be computed
    for all specified subgroups.

    Validates: Requirements 3.7
    """
    predictions, ground_truth, confidence_scores, slide_ids = data
    n_samples = len(slide_ids)

    # Ensure we have at least as many samples as subgroups
    assume(n_samples >= n_subgroups)

    analyzer = FailureAnalyzer()

    # Identify failures
    failures = analyzer.identify_failures(
        predictions=predictions,
        ground_truth=ground_truth,
        confidence_scores=confidence_scores,
        slide_ids=slide_ids,
    )

    # Create clinical subgroups (partition slide_ids)
    subgroup_size = n_samples // n_subgroups
    clinical_subgroups = {}

    for i in range(n_subgroups):
        start_idx = i * subgroup_size
        end_idx = (i + 1) * subgroup_size if i < n_subgroups - 1 else n_samples
        subgroup_name = f"subgroup_{i}"
        clinical_subgroups[subgroup_name] = slide_ids[start_idx:end_idx]

    # Analyze systematic biases
    bias_metrics = analyzer.identify_systematic_biases(failures, clinical_subgroups)

    # Property: Bias metrics computed for all subgroups
    assert (
        len(bias_metrics) == n_subgroups
    ), f"Bias metrics must be computed for all {n_subgroups} subgroups, got {len(bias_metrics)}"

    for subgroup_name in clinical_subgroups.keys():
        assert subgroup_name in bias_metrics, f"Bias metric missing for subgroup '{subgroup_name}'"

        # Verify failure rate is in valid range [0, 1]
        failure_rate = bias_metrics[subgroup_name]
        assert (
            0.0 <= failure_rate <= 1.0
        ), f"Failure rate for '{subgroup_name}' must be in [0, 1], got {failure_rate}"


# Property 12: Failure CSV Export Completeness
@given(data=predictions_and_labels(min_samples=1, max_samples=50))
@settings(max_examples=100, deadline=None)
def test_property_12_failure_csv_export_completeness(data):
    """
    Feature: model-interpretability, Property 12: Failure CSV Export Completeness

    For any failure analysis results, the exported CSV SHALL contain all required
    columns: slide_id, prediction, ground_truth, confidence, and cluster_assignment.

    Validates: Requirements 3.6
    """
    predictions, ground_truth, confidence_scores, slide_ids = data

    analyzer = FailureAnalyzer()

    # Identify failures
    failures = analyzer.identify_failures(
        predictions=predictions,
        ground_truth=ground_truth,
        confidence_scores=confidence_scores,
        slide_ids=slide_ids,
    )

    # Add dummy cluster assignments
    failures["cluster_id"] = 0

    # Create dummy cluster stats
    cluster_stats = {0: {"count": len(failures[failures["is_failure"]])}}

    # Export to temporary file
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "failure_report.csv"

        result_path = analyzer.export_failure_report(
            failures=failures, cluster_stats=cluster_stats, output_path=output_path
        )

        # Verify file was created
        assert result_path.exists(), "Export file must be created"

        # Read exported CSV
        exported_df = pd.read_csv(result_path)

        # Property: CSV contains all required columns
        required_columns = [
            "slide_id",
            "prediction",
            "ground_truth",
            "confidence",
            "cluster_assignment",
        ]

        for col in required_columns:
            assert (
                col in exported_df.columns
            ), f"Exported CSV must contain column '{col}'. Found columns: {list(exported_df.columns)}"

        # Verify exported data matches failures
        n_failures = failures["is_failure"].sum()
        assert (
            len(exported_df) == n_failures
        ), f"Exported CSV must contain all {n_failures} failures, got {len(exported_df)}"

        # Verify no NaN values in required columns
        for col in required_columns:
            assert not exported_df[col].isna().any(), f"Column '{col}' must not contain NaN values"


# Edge case: Zero failures
@given(n_samples=st.integers(min_value=1, max_value=50))
@settings(max_examples=50, deadline=None)
def test_zero_failures_edge_case(n_samples):
    """Test that analyzer handles zero failures correctly."""
    analyzer = FailureAnalyzer()

    # Create data with no failures (all predictions correct)
    predictions = np.arange(n_samples) % 2
    ground_truth = predictions.copy()  # All correct
    confidence_scores = np.random.uniform(0.5, 1.0, n_samples).astype(np.float32)
    slide_ids = [f"slide_{i:04d}" for i in range(n_samples)]

    # Identify failures
    result = analyzer.identify_failures(
        predictions=predictions,
        ground_truth=ground_truth,
        confidence_scores=confidence_scores,
        slide_ids=slide_ids,
    )

    # Verify no failures identified
    assert result["is_failure"].sum() == 0, "Should have zero failures"

    # Test export with zero failures
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "empty_report.csv"

        result_path = analyzer.export_failure_report(
            failures=result, cluster_stats={}, output_path=output_path
        )

        # Verify empty CSV was created
        assert result_path.exists(), "Empty report file must be created"

        exported_df = pd.read_csv(result_path)
        assert len(exported_df) == 0, "Empty report must have zero rows"

        # Verify required columns exist even in empty report
        required_columns = [
            "slide_id",
            "prediction",
            "ground_truth",
            "confidence",
            "cluster_assignment",
        ]
        for col in required_columns:
            assert col in exported_df.columns, f"Empty report must have column '{col}'"
