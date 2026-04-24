"""
Federated aggregation strategies for pathology model training.

FedAvg: weighted average by dataset size (McMahan et al. 2017)
FedProx: FedAvg + proximal regularization (Li et al. 2020)
ByzantineRobust: coordinate-wise median aggregation
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Flower types — imported lazily to keep package optional at import time
NDArrays = List[np.ndarray]


def _weighted_average(
    results: List[Tuple[NDArrays, int]],
) -> NDArrays:
    """Weighted average of parameter arrays by number of examples."""
    total_examples = sum(n for _, n in results)
    aggregated: Optional[NDArrays] = None
    for params, n_examples in results:
        weight = n_examples / total_examples
        if aggregated is None:
            aggregated = [w * weight for w in params]
        else:
            for i, w in enumerate(params):
                aggregated[i] += w * weight
    return aggregated or []


class FedAvgPathology:
    """
    FedAvg with slide-count weighting and per-round metric logging.

    Weights each client by number of WSI slides, not raw examples,
    to handle institutions with different slide counts fairly.
    """

    def __init__(
        self,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn=None,
    ):
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.evaluate_fn = evaluate_fn
        self._round = 0

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[NDArrays, Dict]],
        failures: List,
    ) -> Tuple[Optional[NDArrays], Dict]:
        self._round = server_round
        if not results:
            logger.warning(f"Round {server_round}: no fit results received")
            return None, {}
        if failures:
            logger.warning(f"Round {server_round}: {len(failures)} client failures")

        weighted_results = [
            (params, metrics.get("num_examples", 1))
            for params, metrics in results
        ]
        aggregated = _weighted_average(weighted_results)
        client_losses = [m.get("train_loss", float("nan")) for _, m in results]
        metrics = {
            "round": server_round,
            "num_clients": len(results),
            "mean_train_loss": float(np.nanmean(client_losses)),
        }
        logger.info(
            f"Round {server_round}: aggregated {len(results)} clients, "
            f"mean_loss={metrics['mean_train_loss']:.4f}"
        )
        return aggregated, metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[float, Dict]],
        failures: List,
    ) -> Tuple[Optional[float], Dict]:
        if not results:
            return None, {}
        total_examples = sum(m.get("num_examples", 1) for _, m in results)
        weighted_loss = sum(
            loss * m.get("num_examples", 1) / total_examples for loss, m in results
        )
        aucs = [m.get("auc", float("nan")) for _, m in results]
        metrics = {
            "round": server_round,
            "weighted_val_loss": weighted_loss,
            "mean_auc": float(np.nanmean(aucs)),
        }
        logger.info(
            f"Round {server_round} eval: loss={weighted_loss:.4f}, mean_AUC={metrics['mean_auc']:.4f}"
        )
        return weighted_loss, metrics


class FedProxPathology(FedAvgPathology):
    """
    FedProx: FedAvg with proximal term μ||w - w_global||² added to local loss.

    The proximal penalty is injected at the client side; the server
    simply stores global weights for clients to reference each round.
    """

    def __init__(self, proximal_mu: float = 0.01, **kwargs):
        super().__init__(**kwargs)
        self.proximal_mu = proximal_mu
        self._global_params: Optional[NDArrays] = None
        logger.info(f"FedProx: μ={proximal_mu}")

    def get_proximal_mu(self) -> float:
        return self.proximal_mu

    def set_global_params(self, params: NDArrays) -> None:
        self._global_params = params

    def get_global_params(self) -> Optional[NDArrays]:
        return self._global_params

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[NDArrays, Dict]],
        failures: List,
    ) -> Tuple[Optional[NDArrays], Dict]:
        aggregated, metrics = super().aggregate_fit(server_round, results, failures)
        if aggregated is not None:
            self._global_params = aggregated
        return aggregated, metrics


class ByzantineRobustAggregation:
    """
    Coordinate-wise median aggregation for Byzantine fault tolerance.

    Median is robust to up to (n-1)/2 malicious clients corrupting
    arbitrary gradient updates (Yin et al. 2018).
    """

    def __init__(self, min_clients: int = 3):
        self.min_clients = min_clients

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[NDArrays, Dict]],
        failures: List,
    ) -> Tuple[Optional[NDArrays], Dict]:
        if len(results) < self.min_clients:
            logger.warning(
                f"Round {server_round}: only {len(results)} clients, "
                f"need {self.min_clients} for Byzantine robustness"
            )
            return None, {}

        # Stack each layer across clients and take coordinate-wise median
        num_layers = len(results[0][0])
        aggregated = []
        for layer_idx in range(num_layers):
            layer_stack = np.stack(
                [params[layer_idx] for params, _ in results], axis=0
            )
            aggregated.append(np.median(layer_stack, axis=0))

        metrics = {
            "round": server_round,
            "num_clients": len(results),
            "aggregation": "byzantine_median",
        }
        logger.info(
            f"Round {server_round}: Byzantine-robust median over {len(results)} clients"
        )
        return aggregated, metrics
