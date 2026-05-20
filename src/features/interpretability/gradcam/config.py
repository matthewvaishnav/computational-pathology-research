"""Configuration data models, parsers, and pretty printers for interpretability.

Provides dataclasses for Grad-CAM and attention configurations,
parsers for loading from dict/HDF5, and pretty printers for serialization.
"""

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
import torch
import torch.nn as nn
import yaml

logger = logging.getLogger(__name__)


@dataclass
class GradCAMConfig:
    """Configuration for Grad-CAM visualization.

    Attributes:
        target_layers: List of layer names to generate CAMs for
        colormap: Matplotlib colormap name ('jet', 'viridis', 'plasma')
        alpha: Transparency for overlay (0=transparent, 1=opaque)
        dpi: Resolution for saved figures (default 300)
        device: Device for computation ('cuda' or 'cpu')
    """

    target_layers: List[str]
    colormap: str = "jet"
    alpha: float = 0.5
    dpi: int = 300
    device: str = "cpu"

    def validate(self, model: Optional[nn.Module] = None) -> Tuple[bool, Optional[str]]:
        """Validate configuration.

        Args:
            model: Optional model to validate target_layers against

        Returns:
            (is_valid, error_message)
        """
        # Validate target_layers
        if not self.target_layers:
            return False, "target_layers cannot be empty"

        if model is not None:
            # Check if layers exist in model
            available_layers = {name for name, _ in model.named_modules() if name}
            for layer_name in self.target_layers:
                if layer_name not in available_layers:
                    return False, f"Layer '{layer_name}' not found in model"

        # Validate alpha
        if not (0.0 <= self.alpha <= 1.0):
            return False, f"alpha must be in [0, 1], got {self.alpha}"

        # Validate colormap
        valid_colormaps = ["jet", "viridis", "plasma", "inferno", "magma", "cividis"]
        if self.colormap not in valid_colormaps:
            return False, f"colormap must be one of {valid_colormaps}, got '{self.colormap}'"

        # Validate dpi
        if self.dpi <= 0:
            return False, f"dpi must be positive, got {self.dpi}"

        # Validate device
        if self.device not in ["cpu", "cuda"]:
            return False, f"device must be 'cpu' or 'cuda', got '{self.device}'"

        return True, None


@dataclass
class AttentionConfig:
    """Configuration for attention visualization.

    Attributes:
        architecture: MIL architecture ('AttentionMIL', 'CLAM', 'TransMIL')
        num_heads: Number of attention heads (for multi-head attention)
        top_k: Number of top patches to highlight
        colormap: Matplotlib colormap name
        dpi: Resolution for saved figures
    """

    architecture: str
    num_heads: int = 1
    top_k: int = 10
    colormap: str = "viridis"
    dpi: int = 300

    def validate(self) -> Tuple[bool, Optional[str]]:
        """Validate configuration.

        Returns:
            (is_valid, error_message)
        """
        # Validate architecture
        valid_architectures = ["AttentionMIL", "CLAM", "TransMIL"]
        if self.architecture not in valid_architectures:
            return (
                False,
                f"architecture must be one of {valid_architectures}, got '{self.architecture}'",
            )

        # Validate num_heads
        if self.num_heads <= 0:
            return False, f"num_heads must be positive, got {self.num_heads}"

        # Validate top_k
        if self.top_k <= 0:
            return False, f"top_k must be positive, got {self.top_k}"

        # Validate dpi
        if self.dpi <= 0:
            return False, f"dpi must be positive, got {self.dpi}"

        return True, None


@dataclass
class AttentionData:
    """Attention data for HDF5 serialization.

    Attributes:
        slide_id: Slide identifier
        architecture: MIL architecture name
        attention_weights: Attention weights [num_patches] or [num_heads, num_patches]
        coordinates: Patch coordinates [num_patches, 2] (x, y)
        slide_dimensions: Slide dimensions (width, height)
    """

    slide_id: str
    architecture: str
    attention_weights: np.ndarray
    coordinates: np.ndarray
    slide_dimensions: Tuple[int, int]

    def validate(self) -> Tuple[bool, Optional[str]]:
        """Validate attention data.

        Returns:
            (is_valid, error_message)
        """
        # Validate attention_weights
        if self.attention_weights.ndim not in [1, 2]:
            return False, f"attention_weights must be 1D or 2D, got {self.attention_weights.ndim}D"

        if np.any(self.attention_weights < 0):
            return False, "attention_weights must be non-negative"

        # Validate coordinates
        if self.coordinates.ndim != 2 or self.coordinates.shape[1] != 2:
            return False, f"coordinates must be [N, 2], got shape {self.coordinates.shape}"

        # Check number of patches matches
        num_patches_weights = self.attention_weights.shape[-1]
        num_patches_coords = self.coordinates.shape[0]
        if num_patches_weights != num_patches_coords:
            return False, (
                f"Number of patches mismatch: "
                f"attention_weights has {num_patches_weights}, "
                f"coordinates has {num_patches_coords}"
            )

        # Validate coordinates are within slide dimensions
        max_x = self.coordinates[:, 0].max()
        max_y = self.coordinates[:, 1].max()
        slide_width, slide_height = self.slide_dimensions

        if max_x >= slide_width or max_y >= slide_height:
            return False, (
                f"Coordinates exceed slide dimensions: "
                f"max_x={max_x}, max_y={max_y}, "
                f"slide_dimensions=({slide_width}, {slide_height})"
            )

        return True, None


class GradCAMParser:
    """Parser for Grad-CAM configuration."""

    @staticmethod
    def parse(config_dict: Dict[str, Any], model: Optional[nn.Module] = None) -> GradCAMConfig:
        """Parse Grad-CAM configuration from dictionary.

        Args:
            config_dict: Configuration dictionary
            model: Optional model to validate target_layers against

        Returns:
            GradCAMConfig object

        Raises:
            ValueError: If configuration is invalid
        """
        try:
            config = GradCAMConfig(**config_dict)
        except TypeError as e:
            raise ValueError(f"Invalid configuration fields: {e}")

        # Validate
        is_valid, error_msg = config.validate(model)
        if not is_valid:
            raise ValueError(f"Invalid Grad-CAM configuration: {error_msg}")

        logger.info("Parsed Grad-CAM configuration")
        return config

    @staticmethod
    def parse_from_yaml(yaml_path: Path, model: Optional[nn.Module] = None) -> GradCAMConfig:
        """Parse Grad-CAM configuration from YAML file.

        Args:
            yaml_path: Path to YAML configuration file
            model: Optional model to validate target_layers against

        Returns:
            GradCAMConfig object
        """
        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)

        return GradCAMParser.parse(config_dict, model)


class GradCAMPrettyPrinter:
    """Pretty printer for Grad-CAM configuration."""

    @staticmethod
    def format(config: GradCAMConfig) -> Dict[str, Any]:
        """Format Grad-CAM configuration as dictionary.

        Args:
            config: GradCAMConfig object

        Returns:
            Configuration dictionary
        """
        return asdict(config)

    @staticmethod
    def save_to_yaml(config: GradCAMConfig, yaml_path: Path):
        """Save Grad-CAM configuration to YAML file.

        Args:
            config: GradCAMConfig object
            yaml_path: Output YAML file path
        """
        yaml_path = Path(yaml_path)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)

        config_dict = GradCAMPrettyPrinter.format(config)

        with open(yaml_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Saved Grad-CAM configuration to {yaml_path}")


class AttentionParser:
    """Parser for attention data from HDF5 files."""

    @staticmethod
    def parse(hdf5_path: Path) -> AttentionData:
        """Parse attention data from HDF5 file.

        Args:
            hdf5_path: Path to HDF5 file

        Returns:
            AttentionData object

        Raises:
            ValueError: If data is invalid
        """
        hdf5_path = Path(hdf5_path)

        if not hdf5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")

        with h5py.File(hdf5_path, "r") as f:
            # Read required fields
            try:
                slide_id = f.attrs["slide_id"]
                architecture = f.attrs["architecture"]
                attention_weights = f["attention_weights"][:]
                coordinates = f["coordinates"][:]
                slide_dimensions = tuple(f.attrs["slide_dimensions"])
            except KeyError as e:
                raise ValueError(f"Missing required field in HDF5: {e}")

        # Create AttentionData object
        data = AttentionData(
            slide_id=slide_id,
            architecture=architecture,
            attention_weights=attention_weights,
            coordinates=coordinates,
            slide_dimensions=slide_dimensions,
        )

        # Validate
        is_valid, error_msg = data.validate()
        if not is_valid:
            raise ValueError(f"Invalid attention data: {error_msg}")

        logger.info(f"Parsed attention data from {hdf5_path}")
        return data


class AttentionPrettyPrinter:
    """Pretty printer for attention data to HDF5 files."""

    @staticmethod
    def format(data: AttentionData, hdf5_path: Path, compression_level: int = 4):
        """Format attention data and save to HDF5 file.

        Args:
            data: AttentionData object
            hdf5_path: Output HDF5 file path
            compression_level: Gzip compression level (0-9, default 4)
        """
        hdf5_path = Path(hdf5_path)
        hdf5_path.parent.mkdir(parents=True, exist_ok=True)

        # Validate data
        is_valid, error_msg = data.validate()
        if not is_valid:
            raise ValueError(f"Invalid attention data: {error_msg}")

        with h5py.File(hdf5_path, "w") as f:
            # Save attributes
            f.attrs["slide_id"] = data.slide_id
            f.attrs["architecture"] = data.architecture
            f.attrs["slide_dimensions"] = data.slide_dimensions

            # Save datasets with compression
            f.create_dataset(
                "attention_weights",
                data=data.attention_weights,
                compression="gzip",
                compression_opts=compression_level,
            )
            f.create_dataset(
                "coordinates",
                data=data.coordinates,
                compression="gzip",
                compression_opts=compression_level,
            )

        logger.info(
            f"Saved attention data to {hdf5_path} with compression level {compression_level}"
        )


# Convenience functions


def load_gradcam_config(yaml_path: Path, model: Optional[nn.Module] = None) -> GradCAMConfig:
    """Load Grad-CAM configuration from YAML file.

    Args:
        yaml_path: Path to YAML configuration file
        model: Optional model to validate target_layers against

    Returns:
        GradCAMConfig object
    """
    return GradCAMParser.parse_from_yaml(yaml_path, model)


def save_gradcam_config(config: GradCAMConfig, yaml_path: Path):
    """Save Grad-CAM configuration to YAML file.

    Args:
        config: GradCAMConfig object
        yaml_path: Output YAML file path
    """
    GradCAMPrettyPrinter.save_to_yaml(config, yaml_path)


def load_attention_data(hdf5_path: Path) -> AttentionData:
    """Load attention data from HDF5 file.

    Args:
        hdf5_path: Path to HDF5 file

    Returns:
        AttentionData object
    """
    return AttentionParser.parse(hdf5_path)


def save_attention_data(data: AttentionData, hdf5_path: Path, compression_level: int = 4):
    """Save attention data to HDF5 file.

    Args:
        data: AttentionData object
        hdf5_path: Output HDF5 file path
        compression_level: Gzip compression level (0-9, default 4)
    """
    AttentionPrettyPrinter.format(data, hdf5_path, compression_level)
