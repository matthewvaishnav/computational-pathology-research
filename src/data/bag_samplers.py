"""
Fixed-length bag sampling for Multiple Instance Learning.

This module implements the FixedLengthBagSampler from nnMIL (Stanford/NIH 2024),
which converts variable-length bags (100-10,000 patches) into uniform sub-bags
for efficient batching during training and inference.

Key Design Decisions:
- Bag length M = median_patches / 2 (rule-based from dataset fingerprint)
- Training: Random sampling without replacement if N > M
- Inference: Sliding window with stride for N > M
- Padding: Zero vectors if N < M
"""

import logging
from typing import Optional, Tuple

import torch

logger = logging.getLogger(__name__)


class FixedLengthBagSampler:
    """
    Converts variable-length bags into fixed-length sub-bags.
    
    This sampler enables efficient batching of multiple slides by ensuring
    all bags have the same length M. It handles three scenarios:
    
    1. N < M (fewer patches than bag length): Pad with zero vectors
    2. N > M (more patches than bag length):
       - Training mode: Random sampling without replacement
       - Inference mode: Sliding window with configurable stride
    3. N == M: Return features as-is
    
    The sampler also creates attention masks that mark padded positions
    as False (invalid) and valid positions as True.
    
    Args:
        bag_length: Fixed length M for all bags (100-10000)
        mode: Sampling mode - 'train' (random sample) or 'inference' (sliding window)
        stride: Stride for sliding window in inference mode.
                Default: bag_length (non-overlapping windows)
    
    Raises:
        ValueError: If bag_length is not in range [100, 10000]
        ValueError: If mode is not 'train' or 'inference'
        ValueError: If stride is not positive
    
    Example:
        >>> sampler = FixedLengthBagSampler(bag_length=512, mode='train')
        >>> features = torch.randn(1000, 1024)  # 1000 patches, 1024-dim features
        >>> sampled_features, mask = sampler.sample(features, num_patches=1000)
        >>> sampled_features.shape
        torch.Size([512, 1024])
        >>> mask.shape
        torch.Size([512])
        >>> mask.all()  # All True since no padding needed
        tensor(True)
    """
    
    def __init__(
        self,
        bag_length: int,
        mode: str = 'train',
        stride: Optional[int] = None
    ):
        """
        Initialize FixedLengthBagSampler.
        
        Args:
            bag_length: Fixed length M for all bags (100-10000)
            mode: Sampling mode - 'train' or 'inference'
            stride: Stride for sliding window (default: bag_length)
        """
        # Validate bag_length
        if not (100 <= bag_length <= 10000):
            raise ValueError(
                f"bag_length must be in range [100, 10000], got {bag_length}"
            )
        
        # Validate mode
        if mode not in ['train', 'inference']:
            raise ValueError(
                f"mode must be 'train' or 'inference', got '{mode}'"
            )
        
        # Validate stride
        if stride is not None and stride <= 0:
            raise ValueError(
                f"stride must be positive, got {stride}"
            )
        
        self.bag_length = bag_length
        self.mode = mode
        self.stride = stride if stride is not None else bag_length
        
        logger.debug(
            f"Initialized FixedLengthBagSampler: "
            f"bag_length={bag_length}, mode={mode}, stride={self.stride}"
        )
    
    def sample(
        self,
        features: torch.Tensor,
        num_patches: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample fixed-length bag from variable-length input.
        
        This method handles three cases:
        1. N < M: Pad with zeros
        2. N == M: Return as-is
        3. N > M: Random sample (train) or sliding window (inference)
        
        Args:
            features: Input features with shape [N, D] where N is number of patches
                     and D is feature dimension
            num_patches: Actual number of valid patches (N)
        
        Returns:
            sampled_features: Fixed-length features [M, D]
            mask: Boolean mask [M] where True indicates valid patches,
                  False indicates padded positions
        
        Raises:
            ValueError: If features is not 2D tensor
            ValueError: If num_patches is negative or exceeds features.shape[0]
        
        Example:
            >>> sampler = FixedLengthBagSampler(bag_length=100, mode='train')
            >>> # Case 1: Padding (N < M)
            >>> features = torch.randn(50, 512)
            >>> sampled, mask = sampler.sample(features, num_patches=50)
            >>> sampled.shape, mask.shape
            (torch.Size([100, 512]), torch.Size([100]))
            >>> mask.sum()  # 50 valid patches
            tensor(50)
            >>> 
            >>> # Case 2: Sampling (N > M)
            >>> features = torch.randn(200, 512)
            >>> sampled, mask = sampler.sample(features, num_patches=200)
            >>> sampled.shape, mask.shape
            (torch.Size([100, 512]), torch.Size([100]))
            >>> mask.all()  # All valid, no padding
            tensor(True)
        """
        # Validate inputs
        if features.dim() != 2:
            raise ValueError(
                f"Expected 2D tensor [num_patches, feature_dim], got {features.dim()}D tensor"
            )
        
        if num_patches < 0 or num_patches > features.shape[0]:
            raise ValueError(
                f"num_patches must be in range [0, {features.shape[0]}], got {num_patches}"
            )
        
        N = num_patches
        M = self.bag_length
        D = features.shape[1]
        
        # Case 1: N < M - Padding required
        if N < M:
            return self._pad_bag(features, N, M, D)
        
        # Case 2: N == M - Return as-is
        elif N == M:
            return features[:N], torch.ones(M, dtype=torch.bool, device=features.device)
        
        # Case 3: N > M - Sampling required
        else:
            if self.mode == 'train':
                return self._random_sample(features, N, M)
            else:  # inference mode
                return self._sliding_window_sample(features, N, M)
    
    def _pad_bag(
        self,
        features: torch.Tensor,
        N: int,
        M: int,
        D: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pad bag with zero vectors when N < M.
        
        Args:
            features: Input features [N_total, D]
            N: Actual number of valid patches
            M: Target bag length
            D: Feature dimension
        
        Returns:
            padded_features: [M, D] with N valid patches and (M-N) zero vectors
            mask: [M] with True for first N positions, False for padding
        """
        # Create output tensor filled with zeros
        padded_features = torch.zeros(M, D, dtype=features.dtype, device=features.device)
        
        # Copy valid patches to the beginning
        padded_features[:N] = features[:N]
        
        # Create mask: True for valid patches, False for padding
        mask = torch.zeros(M, dtype=torch.bool, device=features.device)
        mask[:N] = True
        
        logger.debug(f"Padded bag: {N} patches -> {M} (added {M-N} zero vectors)")
        
        return padded_features, mask
    
    def _random_sample(
        self,
        features: torch.Tensor,
        N: int,
        M: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Randomly sample M patches from N patches without replacement (training mode).
        
        Args:
            features: Input features [N_total, D]
            N: Actual number of valid patches
            M: Target bag length
        
        Returns:
            sampled_features: [M, D] randomly sampled patches
            mask: [M] all True (no padding)
        """
        # Random sampling without replacement
        indices = torch.randperm(N, device=features.device)[:M]
        sampled_features = features[indices]
        
        # All patches are valid (no padding)
        mask = torch.ones(M, dtype=torch.bool, device=features.device)
        
        logger.debug(f"Random sampled: {M} patches from {N} (train mode)")
        
        return sampled_features, mask
    
    def _sliding_window_sample(
        self,
        features: torch.Tensor,
        N: int,
        M: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample using sliding window (inference mode).
        
        For inference, we use the first window only. The full sliding window
        ensemble is handled by the SlidingWindowInference class.
        
        Args:
            features: Input features [N_total, D]
            N: Actual number of valid patches
            M: Target bag length
        
        Returns:
            sampled_features: [M, D] first window of patches
            mask: [M] all True (no padding)
        """
        # For single-window sampling, just take the first M patches
        # Full sliding window ensemble is handled by SlidingWindowInference
        sampled_features = features[:M]
        
        # All patches are valid (no padding)
        mask = torch.ones(M, dtype=torch.bool, device=features.device)
        
        logger.debug(
            f"Sliding window sample: first {M} patches from {N} (inference mode)"
        )
        
        return sampled_features, mask
    
    def get_num_windows(self, num_patches: int) -> int:
        """
        Calculate number of windows for sliding window inference.
        
        This is useful for pre-allocating memory for ensemble predictions.
        
        Args:
            num_patches: Total number of patches in the bag
        
        Returns:
            Number of windows that will be generated
        
        Example:
            >>> sampler = FixedLengthBagSampler(bag_length=100, stride=50)
            >>> sampler.get_num_windows(250)
            4  # Windows at positions: 0, 50, 100, 150
        """
        if num_patches <= self.bag_length:
            return 1
        
        # Number of windows = (N - M) / stride + 1
        num_windows = (num_patches - self.bag_length) // self.stride + 1
        return num_windows
    
    def __repr__(self) -> str:
        """String representation of the sampler."""
        return (
            f"FixedLengthBagSampler("
            f"bag_length={self.bag_length}, "
            f"mode='{self.mode}', "
            f"stride={self.stride})"
        )
