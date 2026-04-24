"""
Flower federated learning client for pathology models.

Wraps local AttentionMIL/CLAM/TransMIL training with optional DP-SGD.
Each participating institution runs one PathologyFLClient.
"""

import logging
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .privacy import DifferentialPrivacyEngine

logger = logging.getLogger(__name__)

try:
    import flwr as fl
    from flwr.common import (
        NDArrays,
        Scalar,
        ndarrays_to_parameters,
        parameters_to_ndarrays,
    )
    HAS_FLWR = True
except ImportError:
    HAS_FLWR = False
    logger.warning("flwr not installed — federated client unavailable. pip install flwr>=1.5.0")


def get_model_weights(model: nn.Module) -> List[np.ndarray]:
    return [val.cpu().numpy() for val in model.state_dict().values()]


def set_model_weights(model: nn.Module, weights: List[np.ndarray]) -> None:
    state_dict = OrderedDict(
        {k: torch.tensor(v) for k, v in zip(model.state_dict().keys(), weights)}
    )
    model.load_state_dict(state_dict, strict=True)


class LocalTrainer:
    """Handles local training loop for one federated round."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        dp_engine: Optional[DifferentialPrivacyEngine] = None,
        local_epochs: int = 1,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.dp_engine = dp_engine
        self.local_epochs = local_epochs

    def train_round(self) -> Dict[str, float]:
        self.model.train()
        self.model.to(self.device)
        total_loss = 0.0
        n_batches = 0

        for _ in range(self.local_epochs):
            for batch in self.train_loader:
                features = batch["wsi_features"]
                if features is None:
                    continue
                features = features.to(self.device)
                labels = batch["label"].to(self.device)
                num_patches = batch.get("num_patches", None)

                self.optimizer.zero_grad()
                logits, _ = self.model(features, num_patches=num_patches)
                loss = self.criterion(logits, labels)
                loss.backward()

                if self.dp_engine is not None:
                    self.dp_engine.clip_and_noise()
                else:
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                self.optimizer.step()
                total_loss += loss.item()
                n_batches += 1

        return {"train_loss": total_loss / max(n_batches, 1)}

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        self.model.eval()
        self.model.to(self.device)
        total_loss = 0.0
        all_probs: List[float] = []
        all_labels: List[int] = []

        for batch in self.val_loader:
            features = batch["wsi_features"]
            if features is None:
                continue
            features = features.to(self.device)
            labels = batch["label"].to(self.device)
            num_patches = batch.get("num_patches", None)

            logits, _ = self.model(features, num_patches=num_patches)
            loss = self.criterion(logits, labels)
            total_loss += loss.item()

            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().tolist()
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().tolist())

        metrics: Dict[str, float] = {"val_loss": total_loss / max(len(self.val_loader), 1)}

        try:
            from sklearn.metrics import roc_auc_score
            if len(set(all_labels)) > 1:
                metrics["auc"] = roc_auc_score(all_labels, all_probs)
        except ImportError:
            pass

        return metrics


if HAS_FLWR:

    class PathologyFLClient(fl.client.NumPyClient):
        """
        Flower NumPy client wrapping local pathology model training.

        Each hospital instantiates this client with its local data.
        The server never sees raw patient data — only model weights.
        """

        def __init__(
            self,
            model: nn.Module,
            train_loader: DataLoader,
            val_loader: DataLoader,
            device: Optional[torch.device] = None,
            local_epochs: int = 1,
            learning_rate: float = 1e-4,
            use_dp: bool = False,
            dp_noise_multiplier: float = 1.1,
            dp_max_grad_norm: float = 1.0,
            dp_delta: float = 1e-5,
            dataset_size: Optional[int] = None,
            proximal_mu: float = 0.0,
        ):
            self.model = model
            self.train_loader = train_loader
            self.val_loader = val_loader
            self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.local_epochs = local_epochs
            self.proximal_mu = proximal_mu
            self._num_examples = dataset_size or len(train_loader.dataset)

            self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            self.criterion = nn.CrossEntropyLoss()

            dp_engine = None
            if use_dp:
                dp_engine = DifferentialPrivacyEngine(
                    model=model,
                    noise_multiplier=dp_noise_multiplier,
                    max_grad_norm=dp_max_grad_norm,
                    delta=dp_delta,
                    batch_size=train_loader.batch_size or 32,
                    dataset_size=self._num_examples,
                )

            self._trainer = LocalTrainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=self.optimizer,
                criterion=self.criterion,
                device=self.device,
                dp_engine=dp_engine,
                local_epochs=local_epochs,
            )
            self._dp_engine = dp_engine

        def get_parameters(self, config: Dict) -> NDArrays:
            return get_model_weights(self.model)

        def set_parameters(self, parameters: NDArrays) -> None:
            set_model_weights(self.model, parameters)

        def fit(
            self, parameters: NDArrays, config: Dict
        ) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
            self.set_parameters(parameters)

            # FedProx: store global params for proximal penalty
            if self.proximal_mu > 0:
                global_params = [p.clone().detach() for p in self.model.parameters()]
                self._global_params = global_params

            train_metrics = self._trainer.train_round()
            updated_params = get_model_weights(self.model)

            metrics: Dict[str, Scalar] = {
                "num_examples": self._num_examples,
                **{k: float(v) for k, v in train_metrics.items()},
            }
            if self._dp_engine is not None:
                eps, delta = self._dp_engine.get_privacy_spent()
                metrics["epsilon"] = float(eps)
                metrics["delta"] = float(delta)
                logger.info(self._dp_engine.privacy_summary())

            return updated_params, self._num_examples, metrics

        def evaluate(
            self, parameters: NDArrays, config: Dict
        ) -> Tuple[float, int, Dict[str, Scalar]]:
            self.set_parameters(parameters)
            eval_metrics = self._trainer.evaluate()
            loss = eval_metrics.pop("val_loss", 0.0)
            return float(loss), self._num_examples, {
                "num_examples": self._num_examples,
                **{k: float(v) for k, v in eval_metrics.items()},
            }

else:

    class PathologyFLClient:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError("flwr not installed. pip install flwr>=1.5.0")
