"""
Secure aggregation using homomorphic encryption for federated learning.

Implements secure multi-party computation for gradient aggregation
using TenSEAL (SEAL homomorphic encryption library).
"""

import base64
import logging
import pickle
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# Optional imports - gracefully handle missing dependencies
try:
    import tenseal as ts

    TENSEAL_AVAILABLE = True
except ImportError:
    TENSEAL_AVAILABLE = False

    # Create dummy classes for type hints
    class ts:
        class Context:
            pass

        class CKKSVector:
            pass

        SCHEME_TYPE = type("SCHEME_TYPE", (), {"CKKS": "CKKS"})()

        @staticmethod
        def context(*args, **kwargs):
            return None

        @staticmethod
        def context_from(*args, **kwargs):
            return None

        @staticmethod
        def ckks_vector(*args, **kwargs):
            return None


logger = logging.getLogger(__name__)


class HomomorphicEncryptionManager:
    """Manages homomorphic encryption context and operations."""

    def __init__(
        self,
        poly_modulus_degree: int = 8192,
        coeff_mod_bit_sizes: List[int] = None,
        scale: float = 2**40,
        global_scale: float = 2**40,
    ):
        """
        Initialize homomorphic encryption manager.

        Args:
            poly_modulus_degree: Polynomial modulus degree (power of 2)
            coeff_mod_bit_sizes: Coefficient modulus bit sizes
            scale: Scale for CKKS encoding
            global_scale: Global scale for operations
        """
        self.poly_modulus_degree = poly_modulus_degree
        self.coeff_mod_bit_sizes = coeff_mod_bit_sizes or [60, 40, 40, 60]
        self.scale = scale
        self.global_scale = global_scale

        # Initialize TenSEAL context
        self.context = None
        self.public_key = None
        self.secret_key = None

        self._setup_context()

        logger.info(f"HE manager initialized: poly_degree={poly_modulus_degree}, scale={scale}")

    def _setup_context(self):
        """Set up TenSEAL context with CKKS scheme."""
        if not TENSEAL_AVAILABLE:
            logger.warning("TenSEAL not available - secure aggregation disabled")
            self.context = None
            self.public_key = None
            self.secret_key = None
            return

        try:
            # Create TenSEAL context
            self.context = ts.context(
                ts.SCHEME_TYPE.CKKS,
                poly_modulus_degree=self.poly_modulus_degree,
                coeff_mod_bit_sizes=self.coeff_mod_bit_sizes,
            )

            # Set global scale
            self.context.global_scale = self.global_scale

            # Generate keys
            self.context.generate_galois_keys()

            # Extract keys for distribution
            self.public_key = self.context.public_key()
            self.secret_key = self.context.secret_key()

            logger.info("TenSEAL context initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize TenSEAL context: {e}")
            raise

    def get_public_context(self) -> ts.Context:
        """
        Get public context (without secret key) for clients.

        Returns:
            Public TenSEAL context
        """
        if not TENSEAL_AVAILABLE or self.context is None:
            raise RuntimeError("TenSEAL not available - cannot create public context")

        public_context = self.context.copy()
        public_context.make_context_public()
        return public_context

    def serialize_public_context(self) -> bytes:
        """
        Serialize public context for transmission.

        Returns:
            Serialized public context
        """
        public_context = self.get_public_context()
        return public_context.serialize()

    def deserialize_context(self, serialized_context: bytes) -> ts.Context:
        """
        Deserialize context from bytes.

        Args:
            serialized_context: Serialized context

        Returns:
            Deserialized TenSEAL context
        """
        return ts.context_from(serialized_context)

    def encrypt_tensor(
        self, tensor: torch.Tensor, context: Optional[ts.Context] = None
    ) -> ts.CKKSVector:
        """
        Encrypt a tensor using CKKS.

        Args:
            tensor: Tensor to encrypt
            context: TenSEAL context (uses self.context if None)

        Returns:
            Encrypted CKKS vector
        """
        ctx = context if context is not None else self.context

        # Flatten tensor and convert to list
        flat_tensor = tensor.flatten().tolist()

        # Encrypt using CKKS
        encrypted_vector = ts.ckks_vector(ctx, flat_tensor, scale=self.scale)

        return encrypted_vector

    def decrypt_tensor(
        self, encrypted_vector: ts.CKKSVector, original_shape: Tuple[int, ...]
    ) -> torch.Tensor:
        """
        Decrypt CKKS vector back to tensor.

        Args:
            encrypted_vector: Encrypted CKKS vector
            original_shape: Original tensor shape

        Returns:
            Decrypted tensor
        """
        # Decrypt to list
        decrypted_list = encrypted_vector.decrypt()

        # Convert to tensor and reshape
        tensor = torch.tensor(decrypted_list[: np.prod(original_shape)], dtype=torch.float32)
        return tensor.reshape(original_shape)

    def add_encrypted_vectors(self, encrypted_vectors: List[ts.CKKSVector]) -> ts.CKKSVector:
        """
        Add multiple encrypted vectors homomorphically.

        Args:
            encrypted_vectors: List of encrypted vectors

        Returns:
            Sum of encrypted vectors
        """
        if not encrypted_vectors:
            raise ValueError("No encrypted vectors provided")

        # Start with first vector
        result = encrypted_vectors[0]

        # Add remaining vectors
        for encrypted_vector in encrypted_vectors[1:]:
            result = result + encrypted_vector

        return result

    def multiply_encrypted_by_scalar(
        self, encrypted_vector: ts.CKKSVector, scalar: float
    ) -> ts.CKKSVector:
        """
        Multiply encrypted vector by plaintext scalar.

        Args:
            encrypted_vector: Encrypted vector
            scalar: Plaintext scalar

        Returns:
            Scaled encrypted vector
        """
        return encrypted_vector * scalar


class SecureAggregator:
    """Secure aggregator using homomorphic encryption."""

    def __init__(self, he_manager: HomomorphicEncryptionManager, max_workers: int = 4):
        """
        Initialize secure aggregator.

        Args:
            he_manager: Homomorphic encryption manager
            max_workers: Maximum worker threads for parallel operations
        """
        self.he_manager = he_manager
        self.max_workers = max_workers

        # Aggregation state
        self.encrypted_gradients: Dict[str, List[ts.CKKSVector]] = {}
        self.gradient_shapes: Dict[str, Tuple[int, ...]] = {}
        self.client_weights: Dict[str, float] = {}

        logger.info(f"Secure aggregator initialized with {max_workers} workers")

    def add_client_update(
        self,
        client_id: str,
        gradients: Dict[str, torch.Tensor],
        weight: float = 1.0,
        context: Optional[ts.Context] = None,
    ):
        """
        Add encrypted client update to aggregation.

        Args:
            client_id: Client identifier
            gradients: Client gradients (plaintext)
            weight: Client weight for aggregation
            context: TenSEAL context for encryption
        """
        ctx = context if context is not None else self.he_manager.context

        # Store client weight
        self.client_weights[client_id] = weight

        # Encrypt and store gradients
        for param_name, gradient in gradients.items():
            # Store shape for later decryption
            self.gradient_shapes[param_name] = gradient.shape

            # Encrypt gradient
            encrypted_gradient = self.he_manager.encrypt_tensor(gradient, ctx)

            # Scale by client weight
            if weight != 1.0:
                encrypted_gradient = self.he_manager.multiply_encrypted_by_scalar(
                    encrypted_gradient, weight
                )

            # Add to aggregation
            if param_name not in self.encrypted_gradients:
                self.encrypted_gradients[param_name] = []

            self.encrypted_gradients[param_name].append(encrypted_gradient)

        logger.info(f"Added encrypted update from {client_id} (weight={weight})")

    def aggregate_encrypted_gradients(self) -> Dict[str, ts.CKKSVector]:
        """
        Aggregate all encrypted gradients homomorphically.

        Returns:
            Dictionary of aggregated encrypted gradients
        """
        if not self.encrypted_gradients:
            raise ValueError("No encrypted gradients to aggregate")

        aggregated_encrypted = {}

        # Aggregate each parameter
        for param_name, encrypted_list in self.encrypted_gradients.items():
            if not encrypted_list:
                continue

            logger.debug(f"Aggregating {len(encrypted_list)} encrypted gradients for {param_name}")

            # Sum encrypted gradients homomorphically
            aggregated_encrypted[param_name] = self.he_manager.add_encrypted_vectors(encrypted_list)

        # Normalize by total weight if needed
        total_weight = sum(self.client_weights.values())
        if total_weight != len(self.client_weights):  # Not uniform weighting
            for param_name in aggregated_encrypted:
                aggregated_encrypted[param_name] = self.he_manager.multiply_encrypted_by_scalar(
                    aggregated_encrypted[param_name], 1.0 / total_weight
                )

        logger.info(f"Aggregated {len(aggregated_encrypted)} encrypted parameters")

        return aggregated_encrypted

    def decrypt_aggregated_gradients(
        self, aggregated_encrypted: Dict[str, ts.CKKSVector]
    ) -> Dict[str, torch.Tensor]:
        """
        Decrypt aggregated gradients.

        Args:
            aggregated_encrypted: Encrypted aggregated gradients

        Returns:
            Decrypted aggregated gradients
        """
        decrypted_gradients = {}

        for param_name, encrypted_gradient in aggregated_encrypted.items():
            if param_name not in self.gradient_shapes:
                logger.warning(f"No shape information for parameter {param_name}")
                continue

            # Decrypt gradient
            decrypted_gradient = self.he_manager.decrypt_tensor(
                encrypted_gradient, self.gradient_shapes[param_name]
            )

            decrypted_gradients[param_name] = decrypted_gradient

        logger.info(f"Decrypted {len(decrypted_gradients)} aggregated parameters")

        return decrypted_gradients

    def secure_aggregate(
        self, client_updates: Dict[str, Tuple[Dict[str, torch.Tensor], float]]
    ) -> Dict[str, torch.Tensor]:
        """
        Perform complete secure aggregation.

        Args:
            client_updates: Dict mapping client_id to (gradients, weight)

        Returns:
            Aggregated gradients (decrypted)
        """
        start_time = time.time()

        # Clear previous state
        self.encrypted_gradients.clear()
        self.gradient_shapes.clear()
        self.client_weights.clear()

        # Add all client updates
        for client_id, (gradients, weight) in client_updates.items():
            self.add_client_update(client_id, gradients, weight)

        # Aggregate encrypted gradients
        aggregated_encrypted = self.aggregate_encrypted_gradients()

        # Decrypt result
        aggregated_gradients = self.decrypt_aggregated_gradients(aggregated_encrypted)

        elapsed_time = time.time() - start_time
        logger.info(
            f"Secure aggregation completed in {elapsed_time:.2f}s for {len(client_updates)} clients"
        )

        return aggregated_gradients

    def clear_state(self):
        """Clear aggregation state."""
        self.encrypted_gradients.clear()
        self.gradient_shapes.clear()
        self.client_weights.clear()


class SecureAggregationProtocol:
    """High-level secure aggregation protocol for federated learning."""

    def __init__(
        self,
        coordinator_id: str = "coordinator",
        poly_modulus_degree: int = 8192,
        max_workers: int = 4,
    ):
        """
        Initialize secure aggregation protocol.

        Args:
            coordinator_id: Coordinator identifier
            poly_modulus_degree: HE polynomial modulus degree
            max_workers: Maximum worker threads
        """
        self.coordinator_id = coordinator_id

        # Initialize HE manager (coordinator only)
        self.he_manager = HomomorphicEncryptionManager(poly_modulus_degree=poly_modulus_degree)

        # Initialize secure aggregator
        self.secure_aggregator = SecureAggregator(self.he_manager, max_workers=max_workers)

        # Protocol state
        self.public_context_serialized = None
        self.round_id = 0

        logger.info(f"Secure aggregation protocol initialized for {coordinator_id}")

    def setup_round(self) -> bytes:
        """
        Set up new aggregation round and return public context.

        Returns:
            Serialized public context for clients
        """
        self.round_id += 1

        # Clear previous round state
        self.secure_aggregator.clear_state()

        # Serialize public context for clients
        self.public_context_serialized = self.he_manager.serialize_public_context()

        logger.info(f"Set up secure aggregation round {self.round_id}")

        return self.public_context_serialized

    def aggregate_client_updates(
        self, client_updates: Dict[str, Tuple[Dict[str, torch.Tensor], float]]
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate client updates securely.

        Args:
            client_updates: Client updates with weights

        Returns:
            Aggregated gradients
        """
        if not client_updates:
            raise ValueError("No client updates provided")

        logger.info(f"Starting secure aggregation for {len(client_updates)} clients")

        # Perform secure aggregation
        aggregated_gradients = self.secure_aggregator.secure_aggregate(client_updates)

        logger.info(f"Secure aggregation round {self.round_id} completed")

        return aggregated_gradients

    def get_public_context(self) -> bytes:
        """Get serialized public context for clients."""
        if self.public_context_serialized is None:
            self.public_context_serialized = self.he_manager.serialize_public_context()

        return self.public_context_serialized


class SecureAggregationClient:
    """Client-side secure aggregation functionality."""

    def __init__(self, client_id: str):
        """
        Initialize secure aggregation client.

        Args:
            client_id: Client identifier
        """
        self.client_id = client_id
        self.public_context = None

        logger.info(f"Secure aggregation client initialized: {client_id}")

    def setup_context(self, serialized_public_context: bytes):
        """
        Set up encryption context from coordinator.

        Args:
            serialized_public_context: Serialized public context
        """
        self.public_context = ts.context_from(serialized_public_context)
        logger.info(f"Client {self.client_id}: Received public context")

    def encrypt_gradients(self, gradients: Dict[str, torch.Tensor]) -> Dict[str, bytes]:
        """
        Encrypt gradients for secure transmission.

        Args:
            gradients: Client gradients

        Returns:
            Dictionary of serialized encrypted gradients
        """
        if self.public_context is None:
            raise ValueError("Public context not set up")

        encrypted_gradients = {}

        for param_name, gradient in gradients.items():
            # Flatten and encrypt
            flat_gradient = gradient.flatten().tolist()
            encrypted_vector = ts.ckks_vector(self.public_context, flat_gradient)

            # Serialize for transmission
            encrypted_gradients[param_name] = encrypted_vector.serialize()

        logger.info(f"Client {self.client_id}: Encrypted {len(gradients)} parameters")

        return encrypted_gradients


def benchmark_secure_aggregation(
    num_clients: int = 5, param_sizes: List[Tuple[int, ...]] = None, poly_modulus_degree: int = 8192
):
    """
    Benchmark secure aggregation performance.

    Args:
        num_clients: Number of simulated clients
        param_sizes: List of parameter tensor shapes
        poly_modulus_degree: HE polynomial modulus degree
    """
    if param_sizes is None:
        param_sizes = [(100, 50), (50, 10), (10,)]  # Small example

    print(f"=== Secure Aggregation Benchmark ===")
    print(f"Clients: {num_clients}")
    print(f"Parameters: {param_sizes}")
    print(f"HE poly degree: {poly_modulus_degree}")
    print()

    # Initialize protocol
    protocol = SecureAggregationProtocol(poly_modulus_degree=poly_modulus_degree)

    # Generate synthetic client updates
    client_updates = {}
    for i in range(num_clients):
        gradients = {}
        for j, shape in enumerate(param_sizes):
            gradients[f"param_{j}"] = torch.randn(shape) * 0.1

        client_updates[f"client_{i}"] = (gradients, 1.0)  # Equal weights

    # Benchmark aggregation
    start_time = time.time()

    # Setup round
    public_context = protocol.setup_round()
    setup_time = time.time() - start_time

    # Aggregate
    aggregation_start = time.time()
    aggregated = protocol.aggregate_client_updates(client_updates)
    aggregation_time = time.time() - aggregation_start

    total_time = time.time() - start_time

    # Results
    print(f"Setup time: {setup_time:.3f}s")
    print(f"Aggregation time: {aggregation_time:.3f}s")
    print(f"Total time: {total_time:.3f}s")
    print(f"Time per client: {aggregation_time/num_clients:.3f}s")

    # Verify correctness (compare with plaintext aggregation)
    plaintext_sum = {}
    for param_name in aggregated.keys():
        plaintext_sum[param_name] = torch.zeros_like(aggregated[param_name])
        for gradients, weight in client_updates.values():
            plaintext_sum[param_name] += gradients[param_name] * weight
        plaintext_sum[param_name] /= num_clients

    # Check accuracy
    max_error = 0.0
    for param_name in aggregated.keys():
        error = torch.max(torch.abs(aggregated[param_name] - plaintext_sum[param_name])).item()
        max_error = max(max_error, error)

    print(f"Maximum error vs plaintext: {max_error:.2e}")
    print(f"Public context size: {len(public_context)} bytes")

    print("=== Benchmark Complete ===")


if __name__ == "__main__":
    # Demo: Secure aggregation

    print("=== Secure Aggregation Demo ===\n")

    # Initialize HE manager
    he_manager = HomomorphicEncryptionManager(poly_modulus_degree=4096)  # Smaller for demo

    # Test basic encryption/decryption
    print("1. Basic Encryption Test:")
    test_tensor = torch.randn(3, 4)
    print(f"   Original: {test_tensor.flatten()[:5].tolist()}")

    encrypted = he_manager.encrypt_tensor(test_tensor)
    decrypted = he_manager.decrypt_tensor(encrypted, test_tensor.shape)
    print(f"   Decrypted: {decrypted.flatten()[:5].tolist()}")

    error = torch.max(torch.abs(test_tensor - decrypted)).item()
    print(f"   Max error: {error:.2e}")

    # Test secure aggregation
    print("\n2. Secure Aggregation Test:")

    # Create synthetic client updates
    client_updates = {}
    for i in range(3):
        gradients = {"layer1": torch.randn(5, 3) * 0.1, "layer2": torch.randn(3, 2) * 0.1}
        client_updates[f"client_{i}"] = (gradients, 1.0)

    # Initialize secure aggregator
    secure_agg = SecureAggregator(he_manager)

    # Perform secure aggregation
    start_time = time.time()
    aggregated = secure_agg.secure_aggregate(client_updates)
    elapsed = time.time() - start_time

    print(f"   Aggregated {len(client_updates)} clients in {elapsed:.3f}s")
    print(f"   Result shapes: {[(k, v.shape) for k, v in aggregated.items()]}")

    # Verify against plaintext
    plaintext_agg = {}
    for param_name in aggregated.keys():
        plaintext_agg[param_name] = torch.zeros_like(aggregated[param_name])
        for gradients, weight in client_updates.values():
            plaintext_agg[param_name] += gradients[param_name]
        plaintext_agg[param_name] /= len(client_updates)

    # Check accuracy
    for param_name in aggregated.keys():
        error = torch.max(torch.abs(aggregated[param_name] - plaintext_agg[param_name])).item()
        print(f"   {param_name} error: {error:.2e}")

    # Run benchmark
    print("\n3. Performance Benchmark:")
    benchmark_secure_aggregation(
        num_clients=3, param_sizes=[(10, 5), (5, 2)], poly_modulus_degree=4096
    )

    print("\n=== Demo Complete ===")
