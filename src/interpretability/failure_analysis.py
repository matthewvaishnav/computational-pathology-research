"""Failure analysis for systematic error identification and clustering.

Identifies misclassified samples, clusters failures by feature embeddings,
and analyzes systematic biases across clinical subgroups.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)


@dataclass
class FailureCase:
    """Single failure case with metadata."""

    slide_id: str
    prediction: int
    ground_truth: int
    confidence: float
    embedding: np.ndarray
    cluster_id: Optional[int] = None
    metadata: Optional[Dict] = None


class FailureAnalyzer:
    """Analyze systematic errors in model predictions.

    Identifies misclassified samples, clusters failures by embeddings,
    and analyzes systematic biases across clinical subgroups.

    Attributes:
        clustering_method: Clustering algorithm ('kmeans', 'dbscan', 'hierarchical')
        n_clusters: Number of clusters (for kmeans/hierarchical)
        embedding_dim: Expected embedding dimensionality
        scaler: StandardScaler for embedding normalization
        failures: List of identified failure cases
        cluster_model: Fitted clustering model

    Examples:
        >>> analyzer = FailureAnalyzer(clustering_method='kmeans', n_clusters=5)
        >>> failures = analyzer.identify_failures(predictions, labels, confidences, embeddings, slide_ids)
        >>> clusters = analyzer.cluster_failures(failures)
        >>> stats = analyzer.analyze_cluster_characteristics(clusters)
    """

    def __init__(
        self,
        clustering_method: str = "kmeans",
        n_clusters: int = 5,
        embedding_dim: int = 512,
        random_state: int = 42,
    ):
        """Initialize failure analyzer.

        Args:
            clustering_method: Clustering algorithm ('kmeans', 'dbscan', 'hierarchical')
            n_clusters: Number of clusters (for kmeans/hierarchical)
            embedding_dim: Expected embedding dimensionality
            random_state: Random seed for reproducibility

        Raises:
            ValueError: If clustering_method is invalid
        """
        valid_methods = ["kmeans", "dbscan", "hierarchical"]
        if clustering_method not in valid_methods:
            raise ValueError(
                f"Invalid clustering_method '{clustering_method}'. "
                f"Must be one of {valid_methods}"
            )

        self.clustering_method = clustering_method
        self.n_clusters = n_clusters
        self.embedding_dim = embedding_dim
        self.random_state = random_state

        self.scaler = StandardScaler()
        self.failures: List[FailureCase] = []
        self.cluster_model = None

        logger.info(
            f"FailureAnalyzer initialized with {clustering_method} clustering, "
            f"{n_clusters} clusters, embedding_dim={embedding_dim}"
        )

    def identify_failures(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray,
        confidences: np.ndarray,
        embeddings: np.ndarray,
        slide_ids: List[str],
        metadata: Optional[List[Dict]] = None,
    ) -> List[FailureCase]:
        """Identify misclassified samples.

        Args:
            predictions: Predicted class labels [N]
            ground_truth: True class labels [N]
            confidences: Prediction confidences [N]
            embeddings: Feature embeddings [N, embedding_dim]
            slide_ids: Slide identifiers [N]
            metadata: Optional metadata for each sample [N]

        Returns:
            List of FailureCase objects for misclassified samples

        Examples:
            >>> failures = analyzer.identify_failures(
            ...     predictions=np.array([0, 1, 0, 1]),
            ...     ground_truth=np.array([0, 0, 0, 1]),
            ...     confidences=np.array([0.9, 0.7, 0.8, 0.95]),
            ...     embeddings=np.random.randn(4, 512),
            ...     slide_ids=['slide1', 'slide2', 'slide3', 'slide4']
            ... )
        """
        # Validate inputs
        n_samples = len(predictions)
        if not (
            len(ground_truth) == len(confidences) == len(embeddings) == len(slide_ids) == n_samples
        ):
            raise ValueError("All input arrays must have same length")

        if embeddings.shape[1] != self.embedding_dim:
            logger.warning(
                f"Embedding dimension {embeddings.shape[1]} != expected {self.embedding_dim}. "
                f"Updating embedding_dim."
            )
            self.embedding_dim = embeddings.shape[1]

        # Identify misclassifications
        misclassified_mask = predictions != ground_truth
        n_failures = misclassified_mask.sum()

        logger.info(
            f"Identified {n_failures}/{n_samples} failures ({100*n_failures/n_samples:.1f}%)"
        )

        # Create FailureCase objects
        failures = []
        for idx in np.where(misclassified_mask)[0]:
            failure = FailureCase(
                slide_id=slide_ids[idx],
                prediction=int(predictions[idx]),
                ground_truth=int(ground_truth[idx]),
                confidence=float(confidences[idx]),
                embedding=embeddings[idx],
                metadata=metadata[idx] if metadata else None,
            )
            failures.append(failure)

        self.failures = failures
        return failures

    def cluster_failures(
        self, failures: Optional[List[FailureCase]] = None
    ) -> Dict[int, List[FailureCase]]:
        """Cluster failures by feature embeddings.

        Args:
            failures: List of FailureCase objects (uses self.failures if None)

        Returns:
            Dictionary mapping cluster_id to list of FailureCase objects

        Examples:
            >>> clusters = analyzer.cluster_failures(failures)
            >>> print(f"Cluster 0 has {len(clusters[0])} failures")
        """
        if failures is None:
            failures = self.failures

        if not failures:
            logger.warning("No failures to cluster")
            return {}

        # Extract embeddings
        embeddings = np.array([f.embedding for f in failures])

        # Normalize embeddings
        embeddings_scaled = self.scaler.fit_transform(embeddings)

        # Fit clustering model
        if self.clustering_method == "kmeans":
            self.cluster_model = KMeans(
                n_clusters=min(self.n_clusters, len(failures)),
                random_state=self.random_state,
                n_init=10,
            )
        elif self.clustering_method == "dbscan":
            self.cluster_model = DBSCAN(eps=0.5, min_samples=2)
        elif self.clustering_method == "hierarchical":
            self.cluster_model = AgglomerativeClustering(
                n_clusters=min(self.n_clusters, len(failures)), linkage="ward"
            )

        # Predict cluster assignments
        cluster_labels = self.cluster_model.fit_predict(embeddings_scaled)

        # Assign cluster IDs to failures
        for failure, cluster_id in zip(failures, cluster_labels):
            failure.cluster_id = int(cluster_id)

        # Group failures by cluster
        clusters = {}
        for failure in failures:
            cluster_id = failure.cluster_id
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(failure)

        logger.info(f"Clustered {len(failures)} failures into {len(clusters)} clusters")

        return clusters

    def analyze_cluster_characteristics(
        self, clusters: Dict[int, List[FailureCase]]
    ) -> pd.DataFrame:
        """Compute statistics for each failure cluster.

        Args:
            clusters: Dictionary mapping cluster_id to list of FailureCase objects

        Returns:
            DataFrame with cluster statistics (size, avg_confidence, common_prediction, etc.)

        Examples:
            >>> stats = analyzer.analyze_cluster_characteristics(clusters)
            >>> print(stats[['cluster_id', 'size', 'avg_confidence']])
        """
        stats = []

        for cluster_id, cluster_failures in clusters.items():
            if not cluster_failures:
                continue

            # Compute statistics
            confidences = [f.confidence for f in cluster_failures]
            predictions = [f.prediction for f in cluster_failures]
            ground_truths = [f.ground_truth for f in cluster_failures]

            # Most common prediction and ground truth
            pred_counts = {}
            gt_counts = {}
            for p, gt in zip(predictions, ground_truths):
                pred_counts[p] = pred_counts.get(p, 0) + 1
                gt_counts[gt] = gt_counts.get(gt, 0) + 1

            most_common_pred = max(pred_counts, key=pred_counts.get)
            most_common_gt = max(gt_counts, key=gt_counts.get)

            cluster_stats = {
                "cluster_id": cluster_id,
                "size": len(cluster_failures),
                "avg_confidence": np.mean(confidences),
                "std_confidence": np.std(confidences),
                "min_confidence": np.min(confidences),
                "max_confidence": np.max(confidences),
                "most_common_prediction": most_common_pred,
                "most_common_ground_truth": most_common_gt,
                "prediction_entropy": self._compute_entropy(predictions),
            }

            stats.append(cluster_stats)

        df = pd.DataFrame(stats)
        df = df.sort_values("size", ascending=False).reset_index(drop=True)

        logger.info(f"Computed statistics for {len(df)} clusters")

        return df

    def identify_systematic_biases(
        self, failures: Optional[List[FailureCase]] = None, subgroup_key: str = "subgroup"
    ) -> pd.DataFrame:
        """Analyze failure distribution across clinical subgroups.

        Args:
            failures: List of FailureCase objects (uses self.failures if None)
            subgroup_key: Metadata key for subgroup identification

        Returns:
            DataFrame with bias metrics per subgroup

        Examples:
            >>> biases = analyzer.identify_systematic_biases(failures, subgroup_key='tissue_type')
            >>> print(biases[['subgroup', 'failure_rate', 'avg_confidence']])
        """
        if failures is None:
            failures = self.failures

        if not failures:
            logger.warning("No failures to analyze")
            return pd.DataFrame()

        # Extract subgroup information
        subgroup_stats = {}

        for failure in failures:
            if failure.metadata and subgroup_key in failure.metadata:
                subgroup = failure.metadata[subgroup_key]
            else:
                subgroup = "unknown"

            if subgroup not in subgroup_stats:
                subgroup_stats[subgroup] = {
                    "count": 0,
                    "confidences": [],
                    "predictions": [],
                    "ground_truths": [],
                }

            subgroup_stats[subgroup]["count"] += 1
            subgroup_stats[subgroup]["confidences"].append(failure.confidence)
            subgroup_stats[subgroup]["predictions"].append(failure.prediction)
            subgroup_stats[subgroup]["ground_truths"].append(failure.ground_truth)

        # Compute bias metrics
        bias_data = []
        for subgroup, stats in subgroup_stats.items():
            bias_metrics = {
                "subgroup": subgroup,
                "failure_count": stats["count"],
                "avg_confidence": np.mean(stats["confidences"]),
                "std_confidence": np.std(stats["confidences"]),
                "prediction_entropy": self._compute_entropy(stats["predictions"]),
            }
            bias_data.append(bias_metrics)

        df = pd.DataFrame(bias_data)
        df = df.sort_values("failure_count", ascending=False).reset_index(drop=True)

        logger.info(f"Analyzed systematic biases across {len(df)} subgroups")

        return df

    def export_failure_report(
        self,
        output_path: Path,
        failures: Optional[List[FailureCase]] = None,
        include_embeddings: bool = False,
    ) -> Path:
        """Export failure analysis report to CSV.

        Args:
            output_path: Output CSV file path
            failures: List of FailureCase objects (uses self.failures if None)
            include_embeddings: Whether to include embedding vectors in CSV

        Returns:
            Path to saved CSV file

        Examples:
            >>> report_path = analyzer.export_failure_report(Path('failures.csv'))
        """
        if failures is None:
            failures = self.failures

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not failures:
            logger.warning("No failures to export. Creating empty report.")
            df = pd.DataFrame(
                columns=["slide_id", "prediction", "ground_truth", "confidence", "cluster_id"]
            )
            df.to_csv(output_path, index=False)
            return output_path

        # Convert failures to DataFrame
        data = []
        for failure in failures:
            row = {
                "slide_id": failure.slide_id,
                "prediction": failure.prediction,
                "ground_truth": failure.ground_truth,
                "confidence": failure.confidence,
                "cluster_id": failure.cluster_id if failure.cluster_id is not None else -1,
            }

            # Add metadata fields
            if failure.metadata:
                for key, value in failure.metadata.items():
                    row[f"metadata_{key}"] = value

            # Add embedding if requested
            if include_embeddings:
                for i, val in enumerate(failure.embedding):
                    row[f"embedding_{i}"] = val

            data.append(row)

        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)

        logger.info(f"Exported failure report with {len(df)} failures to {output_path}")

        return output_path

    @staticmethod
    def _compute_entropy(values: List[int]) -> float:
        """Compute Shannon entropy of discrete distribution.

        Args:
            values: List of discrete values

        Returns:
            Entropy in bits
        """
        if not values:
            return 0.0

        # Count occurrences
        counts = {}
        for v in values:
            counts[v] = counts.get(v, 0) + 1

        # Compute probabilities
        total = len(values)
        probs = [count / total for count in counts.values()]

        # Compute entropy
        entropy = -sum(p * np.log2(p) for p in probs if p > 0)

        return entropy
