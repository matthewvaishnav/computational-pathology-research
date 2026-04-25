"""
Byzantine-robust aggregation algorithms for federated learning.

Implements robust aggregation methods that can handle malicious clients
sending adversarial updates to poison the global model.
"""

import logging
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN

from ..common.data_models import ClientUpdate
from .base import BaseAggregator

logger = logging.getLogger(__name__)


class KrumAggregator(BaseAggregator):
    """
    Krum aggregation algorithm for Byzantine robustness.

    Krum selects the update that is closest to its k nearest neighbors,
    effectively filtering out outlier (potentially malicious) updates.

    Reference: "Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent" (Blanchard et al., 2017)
    """

    def __init__(self, num_byzantine: int = 1, multi_krum: bool = False):
        """
        Initialize Krum aggregator.

        Args:
            num_byzantine: Maximum number of Byzantine clients (f)
            multi_krum: Use Multi-Krum (average of top-k) instead of single Krum
        """
        super().__init__()
        self.num_byzantine = num_byzantine
        self.multi_krum = multi_krum
        self.algorithm_name = "Multi-Krum" if multi_krum else "Krum"

        logger.info(f"{self.algorithm_name} aggregator initialized: f={num_byzantine}")

    def aggregate(
        self, client_updates: List[ClientUpdate], global_model: Optional[nn.Module] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate client updates using Krum algorithm.

        Args:
            client_updates: List of client updates
            global_model: Current global model (unused)

        Returns:
            Aggregated model parameters
        """
        if not client_updates:
            raise ValueError("No client updates provided")

        n = len(client_updates)
        f = self.num_byzantine

        if n <= 2 * f:
            raise ValueError(f"Need n > 2f clients: got n={n}, f={f}")

        # Validate updates
        self._validate_updates(client_updates)

        # Flatten all gradients for distance computation
        flattened_gradients = self._flatten_gradients(client_updates)

        # Compute Krum scores
        krum_scores = self._compute_krum_scores(flattened_gradients, f)

        if self.multi_krum:
            # Multi-Krum: average top (n-f) updates
            top_k = n - f
            selected_indices = torch.topk(krum_scores, top_k, largest=False).indices

            logger.info(f"Multi-Krum selected {top_k} updates: {selected_indices.tolist()}")

            # Average selected updates
            return self._average_selected_updates(client_updates, selected_indices)
        else:
            # Single Krum: select best update
            best_idx = torch.argmin(krum_scores).item()

            logger.info(
                f"Krum selected client {client_updates[best_idx].client_id} (index {best_idx})"
            )

            return client_updates[best_idx].gradients

    def _flatten_gradients(self, client_updates: List[ClientUpdate]) -> torch.Tensor:
        """
        Flatten all client gradients into vectors.

        Args:
            client_updates: List of client updates

        Returns:
            Tensor of shape [num_clients, total_params]
        """
        flattened_list = []

        for update in client_updates:
            # Concatenate all parameters into single vector
            param_vectors = []
            for param_name in sorted(update.gradients.keys()):
                param_vectors.append(update.gradients[param_name].flatten())

            flattened_gradient = torch.cat(param_vectors)
            flattened_list.append(flattened_gradient)

        return torch.stack(flattened_list)

    def _compute_krum_scores(self, gradients: torch.Tensor, f: int) -> torch.Tensor:
        """
        Compute Krum scores for all gradients.

        Args:
            gradients: Flattened gradients [num_clients, total_params]
            f: Number of Byzantine clients

        Returns:
            Krum scores for each client
        """
        n = gradients.shape[0]
        k = n - f - 2  # Number of nearest neighbors to consider

        if k <= 0:
            raise ValueError(f"Invalid k={k} for n={n}, f={f}")

        # Compute pairwise squared distances
        distances = torch.cdist(gradients, gradients, p=2) ** 2

        # For each client, find k nearest neighbors and sum distances
        krum_scores = torch.zeros(n)

        for i in range(n):
            # Get distances from client i to all others
            client_distances = distances[i]

            # Exclude self (distance 0)
            client_distances[i] = float("inf")

            # Find k smallest distances
            k_nearest_distances = torch.topk(client_distances, k, largest=False).values

            # Krum score is sum of distances to k nearest neighbors
            krum_scores[i] = k_nearest_distances.sum()

        return krum_scores

    def _average_selected_updates(
        self, client_updates: List[ClientUpdate], selected_indices: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Average selected client updates.

        Args:
            client_updates: All client updates
            selected_indices: Indices of selected updates

        Returns:
            Averaged gradients
        """
        if len(selected_indices) == 0:
            raise ValueError("No updates selected")

        # Initialize averaged gradients
        param_names = list(client_updates[0].gradients.keys())
        averaged_gradients = {}

        for param_name in param_names:
            averaged_gradients[param_name] = torch.zeros_like(
                client_updates[0].gradients[param_name]
            )

        # Sum selected gradients
        for idx in selected_indices:
            update = client_updates[idx.item()]
            for param_name in param_names:
                averaged_gradients[param_name] += update.gradients[param_name]

        # Average
        num_selected = len(selected_indices)
        for param_name in param_names:
            averaged_gradients[param_name] /= num_selected

        return averaged_gradients

    def _validate_updates(self, client_updates: List[ClientUpdate]):
        """Validate client updates for consistency."""
        if not client_updates:
            raise ValueError("Empty client updates list")

        # Check parameter consistency
        first_params = set(client_updates[0].gradients.keys())
        for i, update in enumerate(client_updates[1:], 1):
            update_params = set(update.gradients.keys())
            if update_params != first_params:
                raise ValueError(f"Parameter mismatch in update {i}")

        # Check tensor shapes
        for param_name in first_params:
            first_shape = client_updates[0].gradients[param_name].shape
            for i, update in enumerate(client_updates[1:], 1):
                if update.gradients[param_name].shape != first_shape:
                    raise ValueError(f"Shape mismatch for {param_name} in update {i}")


class TrimmedMeanAggregator(BaseAggregator):
    """
    Trimmed Mean aggregation for Byzantine robustness.

    Removes the largest and smallest values for each parameter coordinate
    and averages the remaining values.
    """

    def __init__(self, trim_ratio: float = 0.1):
        """
        Initialize Trimmed Mean aggregator.

        Args:
            trim_ratio: Fraction of extreme values to trim (0.0 to 0.5)
        """
        super().__init__()
        self.trim_ratio = trim_ratio
        self.algorithm_name = "TrimmedMean"

        if not 0.0 <= trim_ratio <= 0.5:
            raise ValueError("trim_ratio must be between 0.0 and 0.5")

        logger.info(f"TrimmedMean aggregator initialized: trim_ratio={trim_ratio}")

    def aggregate(
        self, client_updates: List[ClientUpdate], global_model: Optional[nn.Module] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate client updates using trimmed mean.

        Args:
            client_updates: List of client updates
            global_model: Current global model (unused)

        Returns:
            Aggregated model parameters
        """
        if not client_updates:
            raise ValueError("No client updates provided")

        n = len(client_updates)
        num_trim = int(n * self.trim_ratio)

        if num_trim * 2 >= n:
            logger.warning(f"Trimming {num_trim*2}/{n} values, using simple mean instead")
            num_trim = 0

        # Validate updates
        self._validate_updates(client_updates)

        # Stack gradients for coordinate-wise operations
        stacked_gradients = self._stack_gradients(client_updates)

        # Apply trimmed mean to each parameter
        trimmed_gradients = {}

        for param_name, param_stack in stacked_gradients.items():
            # param_stack shape: [num_clients, param_shape...]

            if num_trim > 0:
                # Sort along client dimension and trim extremes
                sorted_params, _ = torch.sort(param_stack, dim=0)
                trimmed_params = sorted_params[num_trim : n - num_trim]

                # Average remaining values
                trimmed_gradients[param_name] = trimmed_params.mean(dim=0)
            else:
                # Simple mean if no trimming
                trimmed_gradients[param_name] = param_stack.mean(dim=0)

        logger.info(f"TrimmedMean aggregated {n} updates (trimmed {num_trim*2})")

        return trimmed_gradients

    def _stack_gradients(self, client_updates: List[ClientUpdate]) -> Dict[str, torch.Tensor]:
        """
        Stack client gradients for coordinate-wise operations.

        Args:
            client_updates: List of client updates

        Returns:
            Dictionary mapping param_name to stacked tensor
        """
        param_names = list(client_updates[0].gradients.keys())
        stacked_gradients = {}

        for param_name in param_names:
            # Stack gradients from all clients
            param_list = [update.gradients[param_name] for update in client_updates]
            stacked_gradients[param_name] = torch.stack(param_list, dim=0)

        return stacked_gradients

    def _validate_updates(self, client_updates: List[ClientUpdate]):
        """Validate client updates for consistency."""
        if not client_updates:
            raise ValueError("Empty client updates list")

        # Check parameter consistency
        first_params = set(client_updates[0].gradients.keys())
        for i, update in enumerate(client_updates[1:], 1):
            update_params = set(update.gradients.keys())
            if update_params != first_params:
                raise ValueError(f"Parameter mismatch in update {i}")


class MedianAggregator(BaseAggregator):
    """
    Coordinate-wise median aggregation for Byzantine robustness.

    Computes the median value for each parameter coordinate across all clients.
    """

    def __init__(self):
        """Initialize Median aggregator."""
        super().__init__()
        self.algorithm_name = "Median"

        logger.info("Median aggregator initialized")

    def aggregate(
        self, client_updates: List[ClientUpdate], global_model: Optional[nn.Module] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate client updates using coordinate-wise median.

        Args:
            client_updates: List of client updates
            global_model: Current global model (unused)

        Returns:
            Aggregated model parameters
        """
        if not client_updates:
            raise ValueError("No client updates provided")

        # Validate updates
        self._validate_updates(client_updates)

        # Stack gradients for coordinate-wise operations
        stacked_gradients = self._stack_gradients(client_updates)

        # Apply median to each parameter
        median_gradients = {}

        for param_name, param_stack in stacked_gradients.items():
            # Compute coordinate-wise median
            median_gradients[param_name] = torch.median(param_stack, dim=0).values

        logger.info(f"Median aggregated {len(client_updates)} updates")

        return median_gradients

    def _stack_gradients(self, client_updates: List[ClientUpdate]) -> Dict[str, torch.Tensor]:
        """Stack client gradients for coordinate-wise operations."""
        param_names = list(client_updates[0].gradients.keys())
        stacked_gradients = {}

        for param_name in param_names:
            param_list = [update.gradients[param_name] for update in client_updates]
            stacked_gradients[param_name] = torch.stack(param_list, dim=0)

        return stacked_gradients

    def _validate_updates(self, client_updates: List[ClientUpdate]):
        """Validate client updates for consistency."""
        if not client_updates:
            raise ValueError("Empty client updates list")

        first_params = set(client_updates[0].gradients.keys())
        for i, update in enumerate(client_updates[1:], 1):
            update_params = set(update.gradients.keys())
            if update_params != first_params:
                raise ValueError(f"Parameter mismatch in update {i}")


class ByzantineDetector:
    """
    Byzantine client detection using statistical methods.

    Identifies potentially malicious clients based on gradient patterns
    and distances from the majority.
    """

    def __init__(
        self,
        detection_method: str = "distance",
        threshold_factor: float = 2.0,
        min_cluster_size: int = 2,
    ):
        """
        Initialize Byzantine detector.

        Args:
            detection_method: "distance", "clustering", or "statistical"
            threshold_factor: Threshold multiplier for outlier detection
            min_cluster_size: Minimum cluster size for clustering method
        """
        self.detection_method = detection_method
        self.threshold_factor = threshold_factor
        self.min_cluster_size = min_cluster_size

        logger.info(f"Byzantine detector initialized: method={detection_method}")

    def detect_byzantine_clients(
        self, client_updates: List[ClientUpdate]
    ) -> Tuple[List[int], List[int]]:
        """
        Detect Byzantine clients in the update list.

        Args:
            client_updates: List of client updates

        Returns:
            Tuple of (honest_indices, byzantine_indices)
        """
        if len(client_updates) < 3:
            # Not enough clients for meaningful detection
            return list(range(len(client_updates))), []

        if self.detection_method == "distance":
            return self._detect_by_distance(client_updates)
        elif self.detection_method == "clustering":
            return self._detect_by_clustering(client_updates)
        elif self.detection_method == "statistical":
            return self._detect_by_statistics(client_updates)
        else:
            raise ValueError(f"Unknown detection method: {self.detection_method}")

    def _detect_by_distance(
        self, client_updates: List[ClientUpdate]
    ) -> Tuple[List[int], List[int]]:
        """
        Detect Byzantine clients using distance-based method.

        Args:
            client_updates: List of client updates

        Returns:
            Tuple of (honest_indices, byzantine_indices)
        """
        # Flatten gradients
        flattened_gradients = []
        for update in client_updates:
            param_vectors = []
            for param_name in sorted(update.gradients.keys()):
                param_vectors.append(update.gradients[param_name].flatten())
            flattened_gradients.append(torch.cat(param_vectors))

        gradients_tensor = torch.stack(flattened_gradients)

        # Compute pairwise distances
        distances = torch.cdist(gradients_tensor, gradients_tensor, p=2)

        # For each client, compute average distance to all others
        avg_distances = distances.mean(dim=1)

        # Detect outliers using threshold
        median_distance = torch.median(avg_distances)
        mad = torch.median(torch.abs(avg_distances - median_distance))  # Median Absolute Deviation

        threshold = median_distance + self.threshold_factor * mad

        byzantine_indices = []
        honest_indices = []

        for i, dist in enumerate(avg_distances):
            if dist > threshold:
                byzantine_indices.append(i)
            else:
                honest_indices.append(i)

        logger.info(
            f"Distance-based detection: {len(byzantine_indices)} Byzantine, {len(honest_indices)} honest"
        )

        return honest_indices, byzantine_indices

    def _detect_by_clustering(
        self, client_updates: List[ClientUpdate]
    ) -> Tuple[List[int], List[int]]:
        """
        Detect Byzantine clients using clustering method.

        Args:
            client_updates: List of client updates

        Returns:
            Tuple of (honest_indices, byzantine_indices)
        """
        # Flatten gradients
        flattened_gradients = []
        for update in client_updates:
            param_vectors = []
            for param_name in sorted(update.gradients.keys()):
                param_vectors.append(update.gradients[param_name].flatten())
            flattened_gradients.append(torch.cat(param_vectors).numpy())

        gradients_array = np.array(flattened_gradients)

        # Apply DBSCAN clustering
        clustering = DBSCAN(eps=0.5, min_samples=self.min_cluster_size)
        cluster_labels = clustering.fit_predict(gradients_array)

        # Find the largest cluster (assumed to be honest clients)
        unique_labels, counts = np.unique(cluster_labels[cluster_labels != -1], return_counts=True)

        if len(unique_labels) == 0:
            # No clusters found, assume all honest
            return list(range(len(client_updates))), []

        largest_cluster_label = unique_labels[np.argmax(counts)]

        honest_indices = []
        byzantine_indices = []

        for i, label in enumerate(cluster_labels):
            if label == largest_cluster_label:
                honest_indices.append(i)
            else:
                byzantine_indices.append(i)

        logger.info(
            f"Clustering-based detection: {len(byzantine_indices)} Byzantine, {len(honest_indices)} honest"
        )

        return honest_indices, byzantine_indices

    def _detect_by_statistics(
        self, client_updates: List[ClientUpdate]
    ) -> Tuple[List[int], List[int]]:
        """
        Detect Byzantine clients using statistical method.

        Args:
            client_updates: List of client updates

        Returns:
            Tuple of (honest_indices, byzantine_indices)
        """
        # Compute gradient norms
        gradient_norms = []
        for update in client_updates:
            total_norm = 0.0
            for param_name, gradient in update.gradients.items():
                total_norm += torch.norm(gradient).item() ** 2
            gradient_norms.append(np.sqrt(total_norm))

        gradient_norms = np.array(gradient_norms)

        # Detect outliers using z-score
        mean_norm = np.mean(gradient_norms)
        std_norm = np.std(gradient_norms)

        if std_norm == 0:
            # All gradients have same norm, assume all honest
            return list(range(len(client_updates))), []

        z_scores = np.abs((gradient_norms - mean_norm) / std_norm)

        byzantine_indices = []
        honest_indices = []

        for i, z_score in enumerate(z_scores):
            if z_score > self.threshold_factor:
                byzantine_indices.append(i)
            else:
                honest_indices.append(i)

        logger.info(
            f"Statistical detection: {len(byzantine_indices)} Byzantine, {len(honest_indices)} honest"
        )

        return honest_indices, byzantine_indices


def simulate_byzantine_attack(
    honest_updates: List[ClientUpdate], attack_type: str = "sign_flip", attack_strength: float = 1.0
) -> List[ClientUpdate]:
    """
    Simulate Byzantine attacks for testing robustness.

    Args:
        honest_updates: List of honest client updates
        attack_type: Type of attack ("sign_flip", "gaussian_noise", "zero")
        attack_strength: Strength of the attack

    Returns:
        List of updates with some Byzantine attacks
    """
    attacked_updates = []

    for i, update in enumerate(honest_updates):
        if i == 0:  # Make first client Byzantine for demo
            attacked_gradients = {}

            for param_name, gradient in update.gradients.items():
                if attack_type == "sign_flip":
                    # Flip the sign and scale
                    attacked_gradients[param_name] = -gradient * attack_strength
                elif attack_type == "gaussian_noise":
                    # Add large Gaussian noise
                    noise = torch.randn_like(gradient) * attack_strength
                    attacked_gradients[param_name] = gradient + noise
                elif attack_type == "zero":
                    # Send zero gradients
                    attacked_gradients[param_name] = torch.zeros_like(gradient)
                else:
                    attacked_gradients[param_name] = gradient

            # Create Byzantine update
            byzantine_update = ClientUpdate(
                client_id=f"byzantine_{update.client_id}",
                round_id=update.round_id,
                model_version=update.model_version,
                gradients=attacked_gradients,
                dataset_size=update.dataset_size,
                training_time_seconds=update.training_time_seconds,
                privacy_epsilon=update.privacy_epsilon,
            )

            attacked_updates.append(byzantine_update)
        else:
            attacked_updates.append(update)

    return attacked_updates


if __name__ == "__main__":
    # Demo: Byzantine-robust aggregation
    from datetime import datetime

    print("=== Byzantine-Robust Aggregation Demo ===\n")

    # Create honest client updates
    honest_updates = []
    for i in range(5):
        gradients = {"layer1": torch.randn(3, 2) * 0.1, "layer2": torch.randn(2, 1) * 0.1}

        update = ClientUpdate(
            client_id=f"honest_client_{i}",
            round_id=1,
            model_version=0,
            gradients=gradients,
            dataset_size=100,
            training_time_seconds=10.0,
            privacy_epsilon=0.0,
        )
        honest_updates.append(update)

    # Simulate Byzantine attack
    attacked_updates = simulate_byzantine_attack(
        honest_updates, attack_type="sign_flip", attack_strength=5.0
    )

    print(f"Created {len(honest_updates)} honest updates + 1 Byzantine attack")

    # Test different aggregation methods
    aggregators = [
        ("FedAvg (vulnerable)", None),  # Will implement simple average for comparison
        ("Krum", KrumAggregator(num_byzantine=1)),
        ("Multi-Krum", KrumAggregator(num_byzantine=1, multi_krum=True)),
        ("TrimmedMean", TrimmedMeanAggregator(trim_ratio=0.2)),
        ("Median", MedianAggregator()),
    ]

    print("\nAggregation Results:")

    for name, aggregator in aggregators:
        if aggregator is None:
            # Simple average (FedAvg equivalent)
            avg_gradients = {}
            param_names = list(attacked_updates[0].gradients.keys())

            for param_name in param_names:
                avg_gradients[param_name] = torch.zeros_like(
                    attacked_updates[0].gradients[param_name]
                )
                for update in attacked_updates:
                    avg_gradients[param_name] += update.gradients[param_name]
                avg_gradients[param_name] /= len(attacked_updates)

            result_norm = sum(torch.norm(grad).item() for grad in avg_gradients.values())
        else:
            try:
                aggregated = aggregator.aggregate(attacked_updates)
                result_norm = sum(torch.norm(grad).item() for grad in aggregated.values())
            except Exception as e:
                result_norm = float("inf")
                print(f"   {name}: Error - {e}")
                continue

        print(f"   {name}: Total gradient norm = {result_norm:.4f}")

    # Test Byzantine detection
    print("\nByzantine Detection:")

    detector = ByzantineDetector(detection_method="distance", threshold_factor=2.0)
    honest_indices, byzantine_indices = detector.detect_byzantine_clients(attacked_updates)

    print(f"   Detected honest clients: {honest_indices}")
    print(f"   Detected Byzantine clients: {byzantine_indices}")

    # Verify detection accuracy
    actual_byzantine = [0]  # We made client 0 Byzantine
    detected_correctly = set(byzantine_indices) == set(actual_byzantine)
    print(f"   Detection accuracy: {'Correct' if detected_correctly else 'Incorrect'}")

    print("\n=== Demo Complete ===")
