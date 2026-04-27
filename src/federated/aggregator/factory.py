"""
Aggregation algorithm factory for federated learning.

Provides a unified interface for creating and configuring different
aggregation algorithms (FedAvg, FedProx, FedAdam, etc.).
"""

import logging
from typing import Any, Dict, Optional, Type

from .base import BaseAggregator
from .byzantine_robust import KrumAggregator, MedianAggregator, TrimmedMeanAggregator
from .fedadam import FedAdamAggregator
from .fedavg import FedAvgAggregator
from .fedprox import FedProxAggregator
from .secure import SecureAggregator

logger = logging.getLogger(__name__)


class AggregatorFactory:
    """Factory for creating federated learning aggregators."""

    # Registry of available aggregators
    _aggregators: Dict[str, Type[BaseAggregator]] = {
        "fedavg": FedAvgAggregator,
        "fedprox": FedProxAggregator,
        "fedadam": FedAdamAggregator,
        "krum": KrumAggregator,
        "trimmed_mean": TrimmedMeanAggregator,
        "median": MedianAggregator,
        "secure": SecureAggregator,
    }

    # Default configurations for each aggregator
    _default_configs: Dict[str, Dict[str, Any]] = {
        "fedavg": {},
        "fedprox": {"mu": 0.01},
        "fedadam": {
            "learning_rate": 0.01,
            "beta1": 0.9,
            "beta2": 0.999,
            "epsilon": 1e-8,
            "weight_decay": 0.0,
        },
        "krum": {"num_byzantine": 1, "multi_krum": False},
        "trimmed_mean": {"trim_ratio": 0.1},
        "median": {},
        "secure": {"poly_modulus_degree": 8192, "max_workers": 4, "dropout_threshold": 0.5},
    }

    @classmethod
    def create_aggregator(
        self, algorithm: str, config: Optional[Dict[str, Any]] = None
    ) -> BaseAggregator:
        """
        Create aggregator instance.

        Args:
            algorithm: Algorithm name ("fedavg", "fedprox", "fedadam")
            config: Algorithm-specific configuration

        Returns:
            Aggregator instance

        Raises:
            ValueError: If algorithm is not supported
        """
        algorithm = algorithm.lower()

        if algorithm not in self._aggregators:
            available = list(self._aggregators.keys())
            raise ValueError(f"Unknown algorithm '{algorithm}'. Available: {available}")

        # Merge default config with provided config
        default_config = self._default_configs.get(algorithm, {})
        final_config = default_config.copy()
        if config:
            final_config.update(config)

        # Create aggregator instance
        aggregator_class = self._aggregators[algorithm]
        aggregator = aggregator_class(**final_config)

        logger.info(f"Created {algorithm} aggregator with config: {final_config}")

        return aggregator

    @classmethod
    def register_aggregator(
        cls,
        name: str,
        aggregator_class: Type[BaseAggregator],
        default_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Register a new aggregator algorithm.

        Args:
            name: Algorithm name
            aggregator_class: Aggregator class
            default_config: Default configuration
        """
        name = name.lower()
        cls._aggregators[name] = aggregator_class
        cls._default_configs[name] = default_config or {}

        logger.info(f"Registered aggregator '{name}': {aggregator_class.__name__}")

    @classmethod
    def get_available_algorithms(cls) -> list:
        """Get list of available aggregation algorithms."""
        return list(cls._aggregators.keys())

    @classmethod
    def get_algorithm_info(cls, algorithm: str) -> Dict[str, Any]:
        """
        Get information about an aggregation algorithm.

        Args:
            algorithm: Algorithm name

        Returns:
            Dictionary with algorithm information
        """
        algorithm = algorithm.lower()

        if algorithm not in cls._aggregators:
            raise ValueError(f"Unknown algorithm '{algorithm}'")

        aggregator_class = cls._aggregators[algorithm]
        default_config = cls._default_configs[algorithm]

        return {
            "name": algorithm,
            "class": aggregator_class.__name__,
            "module": aggregator_class.__module__,
            "default_config": default_config,
            "docstring": aggregator_class.__doc__,
        }

    @classmethod
    def create_from_config(cls, config: Dict[str, Any]) -> BaseAggregator:
        """
        Create aggregator from configuration dictionary.

        Args:
            config: Configuration with 'algorithm' key and optional parameters

        Returns:
            Aggregator instance

        Example:
            config = {
                "algorithm": "fedprox",
                "mu": 0.05
            }
        """
        if "algorithm" not in config:
            raise ValueError("Configuration must contain 'algorithm' key")

        algorithm = config["algorithm"]
        algorithm_config = {k: v for k, v in config.items() if k != "algorithm"}

        return cls.create_aggregator(algorithm, algorithm_config)


def create_aggregator(algorithm: str, **kwargs) -> BaseAggregator:
    """
    Convenience function to create aggregator.

    Args:
        algorithm: Algorithm name
        **kwargs: Algorithm-specific parameters

    Returns:
        Aggregator instance
    """
    return AggregatorFactory.create_aggregator(algorithm, kwargs)


def get_aggregator_recommendations(
    scenario: str,
    data_heterogeneity: str = "medium",
    system_heterogeneity: str = "low",
    privacy_requirements: str = "none",
) -> Dict[str, str]:
    """
    Get aggregator recommendations based on scenario.

    Args:
        scenario: Use case scenario ("research", "production", "demo")
        data_heterogeneity: Level of data heterogeneity ("low", "medium", "high")
        system_heterogeneity: Level of system heterogeneity ("low", "medium", "high")
        privacy_requirements: Privacy requirements ("none", "basic", "strict")

    Returns:
        Dictionary with recommendations and rationale
    """
    recommendations = {}

    # Base recommendation logic
    if data_heterogeneity == "low" and system_heterogeneity == "low":
        # Homogeneous setting - FedAvg works well
        recommendations["primary"] = "fedavg"
        recommendations["rationale"] = "FedAvg is optimal for homogeneous data and systems"

    elif data_heterogeneity in ["medium", "high"] or system_heterogeneity in ["medium", "high"]:
        # Heterogeneous setting - FedProx handles non-IID better
        recommendations["primary"] = "fedprox"
        recommendations["rationale"] = (
            "FedProx handles heterogeneous data/systems better with proximal term"
        )

        # Suggest mu value based on heterogeneity
        if data_heterogeneity == "high" or system_heterogeneity == "high":
            recommendations["config"] = {"mu": 0.1}
        else:
            recommendations["config"] = {"mu": 0.01}

    # Secondary recommendations
    if scenario == "research":
        recommendations["secondary"] = "fedadam"
        recommendations["secondary_rationale"] = (
            "FedAdam provides adaptive optimization for research experiments"
        )

    elif scenario == "production":
        if recommendations["primary"] == "fedavg":
            recommendations["secondary"] = "fedprox"
        else:
            recommendations["secondary"] = "fedadam"
        recommendations["secondary_rationale"] = (
            "Consider adaptive methods for production robustness"
        )

    # Privacy considerations
    if privacy_requirements in ["basic", "strict"]:
        recommendations["privacy_note"] = "All aggregators support differential privacy integration"

    return recommendations


# Pre-configured aggregator instances for common scenarios
AGGREGATOR_PRESETS = {
    "demo": {"algorithm": "fedavg", "description": "Simple FedAvg for demonstrations"},
    "research_iid": {"algorithm": "fedavg", "description": "FedAvg for IID research scenarios"},
    "research_noniid": {
        "algorithm": "fedprox",
        "mu": 0.01,
        "description": "FedProx for non-IID research scenarios",
    },
    "production_stable": {
        "algorithm": "fedprox",
        "mu": 0.001,
        "description": "Conservative FedProx for stable production",
    },
    "production_adaptive": {
        "algorithm": "fedadam",
        "learning_rate": 0.001,
        "beta1": 0.9,
        "beta2": 0.999,
        "description": "FedAdam for adaptive production systems",
    },
    "high_heterogeneity": {
        "algorithm": "fedprox",
        "mu": 0.1,
        "description": "Strong regularization for highly heterogeneous data",
    },
}


def create_preset_aggregator(preset: str) -> BaseAggregator:
    """
    Create aggregator from preset configuration.

    Args:
        preset: Preset name

    Returns:
        Aggregator instance
    """
    if preset not in AGGREGATOR_PRESETS:
        available = list(AGGREGATOR_PRESETS.keys())
        raise ValueError(f"Unknown preset '{preset}'. Available: {available}")

    config = AGGREGATOR_PRESETS[preset].copy()
    description = config.pop("description", "")

    aggregator = AggregatorFactory.create_from_config(config)

    logger.info(f"Created preset aggregator '{preset}': {description}")

    return aggregator


if __name__ == "__main__":
    # Demo: Aggregator factory usage

    print("=== Aggregator Factory Demo ===\n")

    # Show available algorithms
    algorithms = AggregatorFactory.get_available_algorithms()
    print(f"Available algorithms: {algorithms}\n")

    # Create different aggregators
    print("Creating aggregators:")

    # FedAvg (default)
    fedavg = AggregatorFactory.create_aggregator("fedavg")
    print(f"  FedAvg: {fedavg}")

    # FedProx with custom mu
    fedprox = AggregatorFactory.create_aggregator("fedprox", {"mu": 0.05})
    print(f"  FedProx: {fedprox}")

    # FedAdam with custom learning rate
    fedadam = AggregatorFactory.create_aggregator("fedadam", {"learning_rate": 0.001})
    print(f"  FedAdam: {fedadam}")

    # Create from config
    print("\nCreating from config:")
    config = {"algorithm": "fedprox", "mu": 0.02}
    aggregator = AggregatorFactory.create_from_config(config)
    print(f"  From config: {aggregator}")

    # Convenience function
    print("\nUsing convenience function:")
    aggregator = create_aggregator("fedadam", learning_rate=0.005, beta1=0.95)
    print(f"  Convenience: {aggregator}")

    # Algorithm info
    print("\nAlgorithm information:")
    for alg in algorithms:
        info = AggregatorFactory.get_algorithm_info(alg)
        print(f"  {alg}: {info['class']} - {info['default_config']}")

    # Recommendations
    print("\nRecommendations:")
    scenarios = [
        ("research", "low", "low", "none"),
        ("production", "high", "medium", "basic"),
        ("demo", "medium", "low", "none"),
    ]

    for scenario, data_het, sys_het, privacy in scenarios:
        rec = get_aggregator_recommendations(scenario, data_het, sys_het, privacy)
        print(
            f"  {scenario} (data:{data_het}, sys:{sys_het}): {rec['primary']} - {rec['rationale']}"
        )

    # Presets
    print("\nPreset aggregators:")
    for preset_name in AGGREGATOR_PRESETS:
        preset_info = AGGREGATOR_PRESETS[preset_name]
        print(f"  {preset_name}: {preset_info['description']}")

    # Create preset
    print("\nCreating preset aggregator:")
    preset_agg = create_preset_aggregator("research_noniid")
    print(f"  research_noniid: {preset_agg}")

    print("\n=== Demo Complete ===")
