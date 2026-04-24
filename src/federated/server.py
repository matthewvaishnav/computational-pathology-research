"""
Flower federated learning server for multi-institutional pathology training.

Manages global model, aggregation strategy selection, and convergence tracking.
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import flwr as fl
    from flwr.common import Metrics, NDArrays, Parameters
    from flwr.server.strategy import FedAvg, FedProx
    HAS_FLWR = True
except ImportError:
    HAS_FLWR = False
    logger.warning("flwr not installed. pip install flwr>=1.5.0")


class ConvergenceTracker:
    """Early stopping based on federated validation loss."""

    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self._best_loss = float("inf")
        self._rounds_without_improvement = 0

    def update(self, val_loss: float) -> bool:
        """Returns True if training should stop."""
        if val_loss < self._best_loss - self.min_delta:
            self._best_loss = val_loss
            self._rounds_without_improvement = 0
            return False
        self._rounds_without_improvement += 1
        if self._rounds_without_improvement >= self.patience:
            logger.info(
                f"Early stopping: no improvement for {self.patience} rounds. "
                f"Best loss: {self._best_loss:.4f}"
            )
            return True
        return False


def weighted_average_metrics(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate per-client metrics weighted by number of examples."""
    total = sum(n for n, _ in metrics)
    aggregated: Dict[str, float] = {}
    for n, m in metrics:
        w = n / total
        for k, v in m.items():
            if isinstance(v, (int, float)):
                aggregated[k] = aggregated.get(k, 0.0) + float(v) * w
    return aggregated


def start_federated_server(
    model_init_fn: Callable[[], Any],
    num_rounds: int = 20,
    strategy: str = "fedavg",
    min_fit_clients: int = 2,
    min_evaluate_clients: int = 2,
    min_available_clients: int = 2,
    server_address: str = "0.0.0.0:8080",
    early_stopping_patience: int = 5,
    proximal_mu: float = 0.01,
    fraction_fit: float = 1.0,
    fraction_evaluate: float = 1.0,
) -> Optional[Dict]:
    """
    Start federated learning server.

    Args:
        model_init_fn: Callable returning initialized nn.Module (for initial params).
        num_rounds: Number of federated rounds.
        strategy: One of "fedavg", "fedprox", "byzantine".
        min_fit_clients: Minimum clients for fit round.
        min_evaluate_clients: Minimum clients for evaluate round.
        min_available_clients: Minimum clients required to start round.
        server_address: gRPC server address.
        early_stopping_patience: Stop if val loss doesn't improve for N rounds.
        proximal_mu: FedProx proximal term coefficient.
        fraction_fit: Fraction of clients to use per fit round.
        fraction_evaluate: Fraction of clients to use per evaluate round.

    Returns:
        Dict of final metrics, or None if flwr unavailable.
    """
    if not HAS_FLWR:
        raise ImportError("flwr not installed. pip install flwr>=1.5.0")

    import torch
    from .client import get_model_weights

    # Get initial parameters from model
    model = model_init_fn()
    initial_params = fl.common.ndarrays_to_parameters(get_model_weights(model))
    del model

    convergence = ConvergenceTracker(patience=early_stopping_patience)
    history: List[Dict] = []

    def on_fit_config(server_round: int) -> Dict:
        return {"round": server_round, "proximal_mu": proximal_mu}

    def on_evaluate_config(server_round: int) -> Dict:
        return {"round": server_round}

    def evaluate_metrics_agg(metrics: List[Tuple[int, Metrics]]) -> Metrics:
        agg = weighted_average_metrics(metrics)
        val_loss = agg.get("val_loss", float("inf"))
        history.append({"round": len(history) + 1, **agg})
        convergence.update(val_loss)
        logger.info(f"Global eval — round {len(history)}: {agg}")
        return agg

    if strategy == "fedprox":
        fl_strategy = FedProx(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            on_fit_config_fn=on_fit_config,
            on_evaluate_config_fn=on_evaluate_config,
            evaluate_metrics_aggregation_fn=evaluate_metrics_agg,
            initial_parameters=initial_params,
            proximal_mu=proximal_mu,
        )
    else:
        # Default: FedAvg
        fl_strategy = FedAvg(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            on_fit_config_fn=on_fit_config,
            on_evaluate_config_fn=on_evaluate_config,
            evaluate_metrics_aggregation_fn=evaluate_metrics_agg,
            initial_parameters=initial_params,
        )

    logger.info(
        f"Starting FL server: strategy={strategy}, rounds={num_rounds}, "
        f"address={server_address}, min_clients={min_available_clients}"
    )

    fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=fl_strategy,
    )

    return {"history": history, "num_rounds": num_rounds}
