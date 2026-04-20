"""Unit tests for failure analysis module."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile

from src.interpretability.failure_analysis import FailureAnalyzer


class TestFailureAnalyzerInit:
    """Test FailureAnalyzer initialization."""

    def test_init_default_parameters(self):
        """Test initialization with default parameters."""
        analyzer = FailureAnalyzer()

        assert analyzer.clustering_method == "kmeans"
        assert analyzer.n_clusters == 5
        assert analyzer.embedding_dim == 256

    def test_init_custom_parameters(self):
        """Test initialization with custom parameters."""
        analyzer = FailureAnalyzer(clustering_method="dbscan", n_clusters=10, embedding_dim=128)

        assert analyzer.clustering_method == "dbscan"
        assert analyzer.n_clusters == 10
        assert analyzer.embedding_dim == 128

    def test_init_invalid_clustering_method(self):
        """Test initialization with invalid clustering method."""
        with pytest.raises(ValueError, match="Invalid clustering_method"):
            FailureAnalyzer(clustering_method="invalid_method")


class TestIdentifyFailures:
    """Test failure identification."""

    def test_identify_failures_basic(self):
        """Test basic failure identification."""
        analyzer = FailureAnalyzer()

        predictions = np.array([0, 1, 0, 1, 0])
        ground_truth = np.array([0, 0, 0, 1, 1])  # 2 failures
        confidence_scores = np.array([0.9, 0.8, 0.95, 0.85, 0.7])
        slide_ids = [f"slide_{i}" for i in range(5)]

        result = analyzer.identify_failures(
            predictions=predictions,
            ground_truth=ground_truth,
            confidence_scores=confidence_scores,
            slide_ids=slide_ids,
        )

        assert len(result) == 5
        assert result["is_failure"].sum() == 2
        assert list(result.columns) == [
            "slide_id",
            "prediction",
            "ground_truth",
            "confidence",
            "is_failure",
        ]

    def test_identify_failures_all_correct(self):
        """Test with all predictions correct."""
        analyzer = FailureAnalyzer()

        predictions = np.array([0, 1, 0, 1])
        ground_truth = np.array([0, 1, 0, 1])  # All correct
        confidence_scores = np.array([0.9, 0.8, 0.95, 0.85])
        slide_ids = [f"slide_{i}" for i in range(4)]

        result = analyzer.identify_failures(
            predictions=predictions,
            ground_truth=ground_truth,
            confidence_scores=confidence_scores,
            slide_ids=slide_ids,
        )

        assert result["is_failure"].sum() == 0

    def test_identify_failures_all_wrong(self):
        """Test with all predictions wrong."""
        analyzer = FailureAnalyzer()

        predictions = np.array([1, 0, 1, 0])
        ground_truth = np.array([0, 1, 0, 1])  # All wrong
        confidence_scores = np.array([0.6, 0.7, 0.65, 0.55])
        slide_ids = [f"slide_{i}" for i in range(4)]

        result = analyzer.identify_failures(
            predictions=predictions,
            ground_truth=ground_truth,
            confidence_scores=confidence_scores,
            slide_ids=slide_ids,
        )

        assert result["is_failure"].sum() == 4

    def test_identify_failures_mismatched_lengths(self):
        """Test with mismatched input lengths."""
        analyzer = FailureAnalyzer()

        with pytest.raises(ValueError, match="Input arrays must have same length"):
            analyzer.identify_failures(
                predictions=np.array([0, 1]),
                ground_truth=np.array([0, 1, 0]),  # Different length
                confidence_scores=np.array([0.9, 0.8]),
                slide_ids=["slide_0", "slide_1"],
            )


class TestClusterFailures:
    """Test failure clustering."""

    def test_cluster_failures_kmeans(self):
        """Test k-means clustering."""
        analyzer = FailureAnalyzer(clustering_method="kmeans", n_clusters=3)

        # Create synthetic embeddings
        n_failures = 20
        embeddings = np.random.randn(n_failures, 128).astype(np.float32)

        metadata = pd.DataFrame(
            {
                "slide_id": [f"slide_{i}" for i in range(n_failures)],
                "prediction": np.random.randint(0, 2, n_failures),
                "ground_truth": np.random.randint(0, 2, n_failures),
                "confidence": np.random.uniform(0.5, 1.0, n_failures),
                "is_failure": [True] * n_failures,
            }
        )

        result = analyzer.cluster_failures(embeddings, metadata)

        assert "cluster_id" in result.columns
        assert len(result) == n_failures
        assert result["cluster_id"].nunique() <= 3

    def test_cluster_failures_dbscan(self):
        """Test DBSCAN clustering."""
        analyzer = FailureAnalyzer(clustering_method="dbscan")

        n_failures = 30
        embeddings = np.random.randn(n_failures, 64).astype(np.float32)

        metadata = pd.DataFrame(
            {
                "slide_id": [f"slide_{i}" for i in range(n_failures)],
                "prediction": [0] * n_failures,
                "ground_truth": [1] * n_failures,
                "confidence": np.random.uniform(0.5, 1.0, n_failures),
                "is_failure": [True] * n_failures,
            }
        )

        result = analyzer.cluster_failures(embeddings, metadata)

        assert "cluster_id" in result.columns
        assert len(result) == n_failures

    def test_cluster_failures_hierarchical(self):
        """Test hierarchical clustering."""
        analyzer = FailureAnalyzer(clustering_method="hierarchical", n_clusters=4)

        n_failures = 25
        embeddings = np.random.randn(n_failures, 256).astype(np.float32)

        metadata = pd.DataFrame(
            {
                "slide_id": [f"slide_{i}" for i in range(n_failures)],
                "prediction": np.random.randint(0, 2, n_failures),
                "ground_truth": np.random.randint(0, 2, n_failures),
                "confidence": np.random.uniform(0.5, 1.0, n_failures),
                "is_failure": [True] * n_failures,
            }
        )

        result = analyzer.cluster_failures(embeddings, metadata)

        assert "cluster_id" in result.columns
        assert len(result) == n_failures
        assert result["cluster_id"].nunique() <= 4

    def test_cluster_failures_zero_failures(self):
        """Test clustering with zero failures."""
        analyzer = FailureAnalyzer()

        embeddings = np.array([]).reshape(0, 128)
        metadata = pd.DataFrame(
            columns=["slide_id", "prediction", "ground_truth", "confidence", "is_failure"]
        )

        result = analyzer.cluster_failures(embeddings, metadata)

        assert "cluster_id" in result.columns
        assert len(result) == 0

    def test_cluster_failures_fewer_samples_than_clusters(self):
        """Test clustering when n_samples < n_clusters."""
        analyzer = FailureAnalyzer(clustering_method="kmeans", n_clusters=10)

        n_failures = 3  # Fewer than n_clusters
        embeddings = np.random.randn(n_failures, 64).astype(np.float32)

        metadata = pd.DataFrame(
            {
                "slide_id": [f"slide_{i}" for i in range(n_failures)],
                "prediction": [0] * n_failures,
                "ground_truth": [1] * n_failures,
                "confidence": [0.8] * n_failures,
                "is_failure": [True] * n_failures,
            }
        )

        result = analyzer.cluster_failures(embeddings, metadata)

        # Should automatically reduce n_clusters to n_samples
        assert "cluster_id" in result.columns
        assert len(result) == n_failures
        assert result["cluster_id"].nunique() <= n_failures

    def test_cluster_failures_mismatched_lengths(self):
        """Test clustering with mismatched embeddings and metadata lengths."""
        analyzer = FailureAnalyzer()

        embeddings = np.random.randn(10, 64).astype(np.float32)
        metadata = pd.DataFrame(
            {
                "slide_id": [f"slide_{i}" for i in range(5)],  # Different length
                "prediction": [0] * 5,
                "ground_truth": [1] * 5,
                "confidence": [0.8] * 5,
                "is_failure": [True] * 5,
            }
        )

        with pytest.raises(ValueError, match="Embeddings and metadata must have same length"):
            analyzer.cluster_failures(embeddings, metadata)


class TestAnalyzeClusterCharacteristics:
    """Test cluster characteristics analysis."""

    def test_analyze_cluster_characteristics_basic(self):
        """Test basic cluster characteristics analysis."""
        analyzer = FailureAnalyzer()

        clustered_failures = pd.DataFrame(
            {
                "slide_id": [f"slide_{i}" for i in range(10)],
                "prediction": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                "ground_truth": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
                "confidence": [0.8, 0.7, 0.9, 0.6, 0.85, 0.75, 0.95, 0.65, 0.8, 0.7],
                "cluster_id": [0, 0, 0, 1, 1, 1, 2, 2, 2, 2],
            }
        )

        stats = analyzer.analyze_cluster_characteristics(clustered_failures)

        assert len(stats) == 3  # 3 clusters
        assert 0 in stats and 1 in stats and 2 in stats

        for cluster_id, cluster_stats in stats.items():
            assert "count" in cluster_stats
            assert "avg_confidence" in cluster_stats
            assert "representative_samples" in cluster_stats
            assert cluster_stats["count"] > 0
            assert 0.0 <= cluster_stats["avg_confidence"] <= 1.0

    def test_analyze_cluster_characteristics_with_clinical_features(self):
        """Test cluster analysis with clinical features."""
        analyzer = FailureAnalyzer()

        clustered_failures = pd.DataFrame(
            {
                "slide_id": ["slide_0", "slide_1", "slide_2", "slide_3"],
                "prediction": [0, 1, 0, 1],
                "ground_truth": [1, 0, 1, 0],
                "confidence": [0.8, 0.7, 0.9, 0.6],
                "cluster_id": [0, 0, 1, 1],
            }
        )

        clinical_features = pd.DataFrame(
            {
                "slide_id": ["slide_0", "slide_1", "slide_2", "slide_3"],
                "age": [45, 50, 45, 60],
                "tumor_size": ["small", "large", "small", "large"],
            }
        )

        stats = analyzer.analyze_cluster_characteristics(
            clustered_failures, clinical_features=clinical_features
        )

        assert len(stats) == 2
        for cluster_stats in stats.values():
            assert "common_characteristics" in cluster_stats
            assert isinstance(cluster_stats["common_characteristics"], dict)

    def test_analyze_cluster_characteristics_missing_cluster_id(self):
        """Test analysis with missing cluster_id column."""
        analyzer = FailureAnalyzer()

        clustered_failures = pd.DataFrame(
            {
                "slide_id": ["slide_0", "slide_1"],
                "prediction": [0, 1],
                "ground_truth": [1, 0],
                "confidence": [0.8, 0.7],
                # Missing cluster_id column
            }
        )

        with pytest.raises(ValueError, match="must contain 'cluster_id' column"):
            analyzer.analyze_cluster_characteristics(clustered_failures)


class TestIdentifySystematicBiases:
    """Test systematic bias identification."""

    def test_identify_systematic_biases_basic(self):
        """Test basic systematic bias identification."""
        analyzer = FailureAnalyzer()

        failures = pd.DataFrame(
            {
                "slide_id": [f"slide_{i}" for i in range(10)],
                "prediction": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                "ground_truth": [0, 0, 0, 1, 1, 0, 1, 1, 1, 1],
                "confidence": [0.8] * 10,
                "is_failure": [False, True, False, False, True, True, True, False, True, False],
            }
        )

        clinical_subgroups = {
            "group_A": ["slide_0", "slide_1", "slide_2", "slide_3", "slide_4"],
            "group_B": ["slide_5", "slide_6", "slide_7", "slide_8", "slide_9"],
        }

        bias_metrics = analyzer.identify_systematic_biases(failures, clinical_subgroups)

        assert len(bias_metrics) == 2
        assert "group_A" in bias_metrics
        assert "group_B" in bias_metrics

        for failure_rate in bias_metrics.values():
            assert 0.0 <= failure_rate <= 1.0

    def test_identify_systematic_biases_empty_subgroup(self):
        """Test bias identification with empty subgroup."""
        analyzer = FailureAnalyzer()

        failures = pd.DataFrame(
            {
                "slide_id": ["slide_0", "slide_1"],
                "prediction": [0, 1],
                "ground_truth": [1, 0],
                "confidence": [0.8, 0.7],
                "is_failure": [True, True],
            }
        )

        clinical_subgroups = {
            "group_A": ["slide_0", "slide_1"],
            "group_B": ["slide_999"],  # No matching slides
        }

        bias_metrics = analyzer.identify_systematic_biases(failures, clinical_subgroups)

        assert len(bias_metrics) == 2
        assert bias_metrics["group_B"] == 0.0

    def test_identify_systematic_biases_missing_is_failure(self):
        """Test bias identification with missing is_failure column."""
        analyzer = FailureAnalyzer()

        failures = pd.DataFrame(
            {
                "slide_id": ["slide_0", "slide_1"],
                "prediction": [0, 1],
                "ground_truth": [1, 0],
                "confidence": [0.8, 0.7],
                # Missing is_failure column
            }
        )

        clinical_subgroups = {"group_A": ["slide_0", "slide_1"]}

        with pytest.raises(ValueError, match="must contain 'is_failure' column"):
            analyzer.identify_systematic_biases(failures, clinical_subgroups)


class TestExportFailureReport:
    """Test failure report export."""

    def test_export_failure_report_basic(self):
        """Test basic failure report export."""
        analyzer = FailureAnalyzer()

        failures = pd.DataFrame(
            {
                "slide_id": ["slide_0", "slide_1", "slide_2"],
                "prediction": [0, 1, 0],
                "ground_truth": [1, 0, 1],
                "confidence": [0.8, 0.7, 0.9],
                "is_failure": [True, True, True],
                "cluster_id": [0, 1, 0],
            }
        )

        cluster_stats = {0: {"count": 2}, 1: {"count": 1}}

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.csv"

            result_path = analyzer.export_failure_report(
                failures=failures, cluster_stats=cluster_stats, output_path=output_path
            )

            assert result_path.exists()

            # Read and verify
            df = pd.read_csv(result_path)
            assert len(df) == 3
            assert list(df.columns) == [
                "slide_id",
                "prediction",
                "ground_truth",
                "confidence",
                "cluster_assignment",
            ]

    def test_export_failure_report_zero_failures(self):
        """Test export with zero failures."""
        analyzer = FailureAnalyzer()

        failures = pd.DataFrame(
            {
                "slide_id": ["slide_0", "slide_1"],
                "prediction": [0, 1],
                "ground_truth": [0, 1],
                "confidence": [0.9, 0.95],
                "is_failure": [False, False],
            }
        )

        cluster_stats = {}

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "empty_report.csv"

            result_path = analyzer.export_failure_report(
                failures=failures, cluster_stats=cluster_stats, output_path=output_path
            )

            assert result_path.exists()

            # Read and verify empty report
            df = pd.read_csv(result_path)
            assert len(df) == 0
            assert "slide_id" in df.columns
            assert "cluster_assignment" in df.columns

    def test_export_failure_report_creates_directory(self):
        """Test that export creates output directory if it doesn't exist."""
        analyzer = FailureAnalyzer()

        failures = pd.DataFrame(
            {
                "slide_id": ["slide_0"],
                "prediction": [0],
                "ground_truth": [1],
                "confidence": [0.8],
                "is_failure": [True],
                "cluster_id": [0],
            }
        )

        cluster_stats = {0: {"count": 1}}

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "subdir" / "nested" / "report.csv"

            result_path = analyzer.export_failure_report(
                failures=failures, cluster_stats=cluster_stats, output_path=output_path
            )

            assert result_path.exists()
            assert result_path.parent.exists()
