"""Failure analysis for systematic error identification and clustering.

Identifies misclassified samples, clusters failures by feature embeddings,
and analyzes systematic biases across clinical subgroups.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.preprocessing import StandardScaler

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
        embedding_dim: int = 256,
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
        confidence_scores: np.ndarray,
        slide_ids: List[str],
        embeddings: Optional[np.ndarray] = None,
        metadata: Optional[List[Dict]] = None,
    ) -> pd.DataFrame:
        """Identify misclassified samples.

        Args:
            predictions: Predicted class labels [N]
            ground_truth: True class labels [N]
            confidence_scores: Prediction confidences [N]
            slide_ids: Slide identifiers [N]
            embeddings: Feature embeddings [N, embedding_dim] (optional)
            metadata: Optional metadata for each sample [N]

        Returns:
            DataFrame with columns: slide_id, prediction, ground_truth, confidence, is_failure

        Examples:
            >>> failures_df = analyzer.identify_failures(
            ...     predictions=np.array([0, 1, 0, 1]),
            ...     ground_truth=np.array([0, 0, 0, 1]),
            ...     confidence_scores=np.array([0.9, 0.7, 0.8, 0.95]),
            ...     slide_ids=['slide1', 'slide2', 'slide3', 'slide4']
            ... )
        """
        # Validate inputs
        n_samples = len(predictions)
        if not (len(ground_truth) == len(confidence_scores) == len(slide_ids) == n_samples):
            raise ValueError("Input arrays must have same length")

        if embeddings is not None and len(embeddings) != n_samples:
            raise ValueError("Input arrays must have same length")

        if embeddings is not None and embeddings.shape[1] != self.embedding_dim:
            logger.warning(
                f"Embedding dimension {embeddings.shape[1]} != expected {self.embedding_dim}. "
                f"Updating embedding_dim."
            )
            self.embedding_dim = embeddings.shape[1]

        # Identify misclassifications
        is_failure = predictions != ground_truth
        n_failures = is_failure.sum()

        logger.info(
            f"Identified {n_failures}/{n_samples} failures ({100*n_failures/n_samples:.1f}%)"
        )

        # Create DataFrame
        result_df = pd.DataFrame(
            {
                "slide_id": slide_ids,
                "prediction": predictions,
                "ground_truth": ground_truth,
                "confidence": confidence_scores,
                "is_failure": is_failure,
            }
        )

        # Store failures as FailureCase objects for internal use
        failures = []
        for idx in np.where(is_failure)[0]:
            failure = FailureCase(
                slide_id=slide_ids[idx],
                prediction=int(predictions[idx]),
                ground_truth=int(ground_truth[idx]),
                confidence=float(confidence_scores[idx]),
                embedding=embeddings[idx] if embeddings is not None else np.array([]),
                metadata=metadata[idx] if metadata else None,
            )
            failures.append(failure)

        self.failures = failures
        return result_df

    def cluster_failures(self, embeddings: np.ndarray, metadata: pd.DataFrame) -> pd.DataFrame:
        """Cluster failures by feature embeddings.

        Args:
            embeddings: Feature embeddings [N, embedding_dim]
            metadata: DataFrame with failure metadata (must contain is_failure column)

        Returns:
            DataFrame with original metadata plus cluster_id column

        Examples:
            >>> clustered_df = analyzer.cluster_failures(embeddings, metadata)
            >>> print(clustered_df['cluster_id'].value_counts())
        """
        if len(embeddings) != len(metadata):
            raise ValueError("Embeddings and metadata must have same length")

        if len(embeddings) == 0:
            logger.warning("No failures to cluster")
            result = metadata.copy()
            result["cluster_id"] = []
            return result

        # Normalize embeddings (constant columns produce NaN; replace with 0)
        embeddings_scaled = self.scaler.fit_transform(embeddings)
        embeddings_scaled = np.nan_to_num(embeddings_scaled, nan=0.0, posinf=0.0, neginf=0.0)

        # Fit clustering model
        if self.clustering_method == "kmeans":
            self.cluster_model = KMeans(
                n_clusters=min(self.n_clusters, len(embeddings)),
                random_state=self.random_state,
                n_init=10,
            )
        elif self.clustering_method == "dbscan":
            self.cluster_model = DBSCAN(eps=0.5, min_samples=2)
        elif self.clustering_method == "hierarchical":
            self.cluster_model = AgglomerativeClustering(
                n_clusters=min(self.n_clusters, len(embeddings)), linkage="ward"
            )

        # Predict cluster assignments
        cluster_labels = self.cluster_model.fit_predict(embeddings_scaled)

        # Add cluster_id to metadata
        result = metadata.copy()
        result["cluster_id"] = cluster_labels

        logger.info(
            f"Clustered {len(embeddings)} failures into {len(set(cluster_labels))} clusters"
        )

        return result

    def analyze_cluster_characteristics(
        self, clustered_failures: pd.DataFrame, clinical_features: Optional[pd.DataFrame] = None
    ) -> Dict[int, Dict]:
        """Compute statistics for each failure cluster.

        Args:
            clustered_failures: DataFrame with cluster_id column
            clinical_features: Optional DataFrame with clinical features

        Returns:
            Dictionary mapping cluster_id to statistics dict

        Examples:
            >>> stats = analyzer.analyze_cluster_characteristics(clustered_df)
            >>> print(stats[0]['count'], stats[0]['avg_confidence'])
        """
        if "cluster_id" not in clustered_failures.columns:
            raise ValueError("clustered_failures must contain 'cluster_id' column")

        stats = {}

        for cluster_id in clustered_failures["cluster_id"].unique():
            cluster_data = clustered_failures[clustered_failures["cluster_id"] == cluster_id]

            if len(cluster_data) == 0:
                continue

            # Basic statistics
            cluster_stats = {
                "count": len(cluster_data),
                "avg_confidence": cluster_data["confidence"].mean(),
                "representative_samples": cluster_data["slide_id"].head(3).tolist(),
            }

            # Add clinical characteristics if provided
            if clinical_features is not None:
                merged = cluster_data.merge(clinical_features, on="slide_id", how="left")
                common_chars = {}
                for col in clinical_features.columns:
                    if col != "slide_id":
                        mode_val = merged[col].mode()
                        if len(mode_val) > 0:
                            common_chars[col] = mode_val.iloc[0]
                cluster_stats["common_characteristics"] = common_chars

            stats[cluster_id] = cluster_stats

        logger.info(f"Computed statistics for {len(stats)} clusters")

        return stats

    def identify_systematic_biases(
        self, failures: pd.DataFrame, clinical_subgroups: Dict[str, List[str]]
    ) -> Dict[str, float]:
        """Analyze failure distribution across clinical subgroups.

        Args:
            failures: DataFrame with is_failure column
            clinical_subgroups: Dict mapping subgroup names to slide_id lists

        Returns:
            Dictionary mapping subgroup names to failure rates

        Examples:
            >>> subgroups = {'group_A': ['slide1', 'slide2'], 'group_B': ['slide3', 'slide4']}
            >>> bias_metrics = analyzer.identify_systematic_biases(failures_df, subgroups)
            >>> print(bias_metrics['group_A'])
        """
        if "is_failure" not in failures.columns:
            raise ValueError("failures must contain 'is_failure' column")

        bias_metrics = {}

        for subgroup_name, slide_ids in clinical_subgroups.items():
            subgroup_data = failures[failures["slide_id"].isin(slide_ids)]

            if len(subgroup_data) == 0:
                bias_metrics[subgroup_name] = 0.0
            else:
                failure_rate = subgroup_data["is_failure"].mean()
                bias_metrics[subgroup_name] = failure_rate

        logger.info(f"Analyzed systematic biases across {len(bias_metrics)} subgroups")

        return bias_metrics

    def export_failure_report(
        self,
        failures: pd.DataFrame,
        cluster_stats: Dict[int, Dict],
        output_path: Path,
    ) -> Path:
        """Export failure analysis report to CSV.

        Args:
            failures: DataFrame with failure data
            cluster_stats: Dictionary with cluster statistics
            output_path: Output CSV file path

        Returns:
            Path to saved CSV file

        Examples:
            >>> report_path = analyzer.export_failure_report(failures_df, stats, Path('failures.csv'))
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if len(failures) == 0:
            logger.warning("No failures to export. Creating empty report.")
            df = pd.DataFrame(
                columns=[
                    "slide_id",
                    "prediction",
                    "ground_truth",
                    "confidence",
                    "cluster_assignment",
                ]
            )
            df.to_csv(output_path, index=False)
            return output_path

        # Filter to only failures and add cluster assignment
        failure_data = failures[failures["is_failure"] == True].copy()

        if "cluster_id" in failure_data.columns:
            failure_data["cluster_assignment"] = failure_data["cluster_id"]
        else:
            failure_data["cluster_assignment"] = -1

        # Select columns for export
        export_cols = ["slide_id", "prediction", "ground_truth", "confidence", "cluster_assignment"]
        export_df = failure_data[export_cols]

        export_df.to_csv(output_path, index=False)

        logger.info(f"Exported failure report with {len(export_df)} failures to {output_path}")

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
