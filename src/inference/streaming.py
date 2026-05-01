"""
Streaming inference for large WSIs.

Memory-efficient tile-by-tile processing for gigapixel whole-slide images.
Processes WSI tiles incrementally without loading entire slide into memory.
"""

import logging
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


class TileDataset(Dataset):
    """Dataset for streaming WSI tiles."""

    def __init__(
        self,
        tiles: List[np.ndarray],
        coordinates: List[Tuple[int, int]],
        transform: Optional[callable] = None,
    ):
        """Initialize tile dataset.

        Args:
            tiles: List of tile images [H, W, C]
            coordinates: List of (x, y) coordinates
            transform: Optional transform to apply to tiles
        """
        self.tiles = tiles
        self.coordinates = coordinates
        self.transform = transform

    def __len__(self) -> int:
        return len(self.tiles)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        tile = self.tiles[idx]
        coord = self.coordinates[idx]

        if self.transform:
            tile = self.transform(tile)
        else:
            # Default: convert to tensor and normalize
            tile = torch.from_numpy(tile).float()
            if tile.ndim == 3:
                tile = tile.permute(2, 0, 1)  # HWC -> CHW
            tile = tile / 255.0

        return {
            "image": tile,
            "coordinates": torch.tensor(coord, dtype=torch.long),
        }


class StreamingInference:
    """Memory-efficient streaming inference for large WSIs.

    Processes WSI tiles incrementally without loading entire slide into memory.
    Supports:
    - Tile-by-tile processing with configurable batch size
    - Memory-efficient feature extraction
    - Progressive result aggregation
    - GPU memory optimization

    Args:
        model: Feature extraction model (e.g., ResNet, EfficientNet)
        device: Device to run inference on
        batch_size: Batch size for tile processing
        num_workers: Number of data loading workers
        memory_limit_gb: GPU memory limit in GB

    Example:
        >>> model = resnet50(pretrained=True)
        >>> streaming = StreamingInference(model, device='cuda', batch_size=32)
        >>> features, coords = streaming.process_tiles(tiles, coordinates)
        >>> print(features.shape)  # [num_tiles, feature_dim]
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        batch_size: int = 32,
        num_workers: int = 4,
        memory_limit_gb: float = 2.0,
        use_amp: bool = True,
    ):
        """Initialize streaming inference.

        Args:
            model: Feature extraction model
            device: Device to run inference on
            batch_size: Batch size for tile processing
            num_workers: Number of data loading workers
            memory_limit_gb: GPU memory limit in GB
            use_amp: Use automatic mixed precision
        """
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.memory_limit_gb = memory_limit_gb
        self.use_amp = use_amp and device == "cuda"

        logger.info(
            f"Initialized StreamingInference: device={device}, "
            f"batch_size={batch_size}, memory_limit={memory_limit_gb}GB, "
            f"amp={self.use_amp}"
        )

    def _check_memory_usage(self) -> float:
        """Check current GPU memory usage.

        Returns:
            Memory usage in GB
        """
        if self.device.type == "cuda":
            return torch.cuda.memory_allocated(self.device) / (1024**3)
        return 0.0

    def _optimize_batch_size(self, current_memory_gb: float) -> int:
        """Dynamically adjust batch size based on memory usage.

        Args:
            current_memory_gb: Current memory usage in GB

        Returns:
            Optimized batch size
        """
        if current_memory_gb > self.memory_limit_gb * 0.8:
            # Reduce batch size if approaching limit
            new_batch_size = max(1, self.batch_size // 2)
            logger.warning(
                f"Memory usage high ({current_memory_gb:.2f}GB), "
                f"reducing batch size: {self.batch_size} → {new_batch_size}"
            )
            return new_batch_size

        return self.batch_size

    def process_tiles(
        self,
        tiles: List[np.ndarray],
        coordinates: List[Tuple[int, int]],
        transform: Optional[callable] = None,
        return_numpy: bool = True,
    ) -> Tuple[Union[np.ndarray, torch.Tensor], np.ndarray]:
        """Process tiles with streaming inference.

        Args:
            tiles: List of tile images [H, W, C]
            coordinates: List of (x, y) coordinates
            transform: Optional transform to apply to tiles
            return_numpy: Return numpy arrays instead of tensors

        Returns:
            Tuple of (features, coordinates):
                - features: [num_tiles, feature_dim]
                - coordinates: [num_tiles, 2]

        Raises:
            RuntimeError: If GPU out of memory
        """
        if len(tiles) != len(coordinates):
            raise ValueError(
                f"Mismatched lengths: tiles={len(tiles)}, coordinates={len(coordinates)}"
            )

        # Create dataset and dataloader
        dataset = TileDataset(tiles, coordinates, transform)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True if self.device.type == "cuda" else False,
        )

        # Process tiles in batches
        all_features = []
        all_coords = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                # Check memory usage
                memory_usage = self._check_memory_usage()

                # Move to device
                images = batch["image"].to(self.device)
                coords = batch["coordinates"]

                try:
                    # Forward pass with AMP
                    if self.use_amp:
                        with torch.cuda.amp.autocast():
                            features = self.model(images)
                    else:
                        features = self.model(images)

                    # Move to CPU to free GPU memory
                    all_features.append(features.cpu())
                    all_coords.append(coords)

                    # Clear GPU cache periodically
                    if batch_idx % 10 == 0 and self.device.type == "cuda":
                        torch.cuda.empty_cache()

                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        logger.error(f"GPU OOM at batch {batch_idx}, memory={memory_usage:.2f}GB")

                        # Emergency cleanup
                        torch.cuda.empty_cache()

                        # Reduce batch size and retry
                        self.batch_size = self._optimize_batch_size(memory_usage)

                        # Recreate dataloader with smaller batch size
                        dataloader = DataLoader(
                            dataset,
                            batch_size=self.batch_size,
                            shuffle=False,
                            num_workers=self.num_workers,
                            pin_memory=True if self.device.type == "cuda" else False,
                        )

                        raise RuntimeError(
                            f"GPU out of memory. Reduced batch size to {self.batch_size}. "
                            "Please retry with smaller batch size."
                        )
                    else:
                        raise e

        # Concatenate results
        features = torch.cat(all_features, dim=0)
        coordinates = torch.cat(all_coords, dim=0)

        if return_numpy:
            features = features.numpy()
            coordinates = coordinates.numpy()

        logger.info(
            f"Processed {len(tiles)} tiles: features shape={features.shape}, "
            f"final memory={self._check_memory_usage():.2f}GB"
        )

        return features, coordinates

    def process_tiles_generator(
        self,
        tiles: List[np.ndarray],
        coordinates: List[Tuple[int, int]],
        transform: Optional[callable] = None,
    ) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """Process tiles with generator for memory efficiency.

        Yields batches of features and coordinates without accumulating in memory.

        Args:
            tiles: List of tile images [H, W, C]
            coordinates: List of (x, y) coordinates
            transform: Optional transform to apply to tiles

        Yields:
            Tuple of (features, coordinates) for each batch:
                - features: [batch_size, feature_dim]
                - coordinates: [batch_size, 2]
        """
        if len(tiles) != len(coordinates):
            raise ValueError(
                f"Mismatched lengths: tiles={len(tiles)}, coordinates={len(coordinates)}"
            )

        # Create dataset and dataloader
        dataset = TileDataset(tiles, coordinates, transform)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True if self.device.type == "cuda" else False,
        )

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                # Move to device
                images = batch["image"].to(self.device)
                coords = batch["coordinates"]

                # Forward pass with AMP
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        features = self.model(images)
                else:
                    features = self.model(images)

                # Yield batch results (on CPU)
                yield features.cpu(), coords

                # Clear GPU cache periodically
                if batch_idx % 10 == 0 and self.device.type == "cuda":
                    torch.cuda.empty_cache()

    def estimate_memory_usage(
        self,
        num_tiles: int,
        tile_size: int = 224,
        feature_dim: int = 2048,
    ) -> Dict[str, float]:
        """Estimate memory usage for processing.

        Args:
            num_tiles: Number of tiles to process
            tile_size: Tile size (assumes square tiles)
            feature_dim: Feature dimension

        Returns:
            Dictionary with memory estimates in GB:
                - input_memory: Memory for input tiles
                - feature_memory: Memory for output features
                - model_memory: Memory for model parameters
                - total_memory: Total estimated memory
        """
        # Input memory (batch_size tiles)
        input_bytes = self.batch_size * 3 * tile_size * tile_size * 4  # float32
        input_memory_gb = input_bytes / (1024**3)

        # Feature memory (batch_size features)
        feature_bytes = self.batch_size * feature_dim * 4  # float32
        feature_memory_gb = feature_bytes / (1024**3)

        # Model memory (parameters)
        model_params = sum(p.numel() for p in self.model.parameters())
        model_memory_gb = model_params * 4 / (1024**3)  # float32

        # Total memory (with overhead)
        total_memory_gb = (input_memory_gb + feature_memory_gb + model_memory_gb) * 1.2

        return {
            "input_memory_gb": input_memory_gb,
            "feature_memory_gb": feature_memory_gb,
            "model_memory_gb": model_memory_gb,
            "total_memory_gb": total_memory_gb,
            "num_batches": (num_tiles + self.batch_size - 1) // self.batch_size,
        }


def create_streaming_inference(
    model_name: str = "resnet50",
    pretrained: bool = True,
    device: str = "cuda",
    batch_size: int = 32,
    memory_limit_gb: float = 2.0,
) -> StreamingInference:
    """Create streaming inference with pretrained model.

    Args:
        model_name: Model architecture name
        pretrained: Use pretrained weights
        device: Device to run inference on
        batch_size: Batch size for tile processing
        memory_limit_gb: GPU memory limit in GB

    Returns:
        StreamingInference instance

    Example:
        >>> streaming = create_streaming_inference('resnet50', device='cuda')
        >>> features, coords = streaming.process_tiles(tiles, coordinates)
    """
    import torchvision.models as models

    # Load model
    if model_name == "resnet50":
        model = models.resnet50(pretrained=pretrained)
        # Remove classification head
        model = nn.Sequential(*list(model.children())[:-1])
    elif model_name == "resnet18":
        model = models.resnet18(pretrained=pretrained)
        model = nn.Sequential(*list(model.children())[:-1])
    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(pretrained=pretrained)
        model.classifier = nn.Identity()
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return StreamingInference(
        model=model,
        device=device,
        batch_size=batch_size,
        memory_limit_gb=memory_limit_gb,
    )
