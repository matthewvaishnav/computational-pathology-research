"""
Privacy budget tracking for federated learning clients.

Tracks per-client privacy expenditure and enforces budget limits
to ensure formal differential privacy guarantees.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import torch

from ..common.data_models import PrivacyBudget
from .dp_sgd import PrivacyAccountant

logger = logging.getLogger(__name__)


@dataclass
class PrivacyBudgetEntry:
    """Single privacy budget entry for a training round."""

    round_id: int
    timestamp: datetime
    epsilon_spent: float
    delta_spent: float
    noise_multiplier: float
    batch_size: int
    dataset_size: int
    clipping_norm: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "round_id": self.round_id,
            "timestamp": self.timestamp.isoformat(),
            "epsilon_spent": self.epsilon_spent,
            "delta_spent": self.delta_spent,
            "noise_multiplier": self.noise_multiplier,
            "batch_size": self.batch_size,
            "dataset_size": self.dataset_size,
            "clipping_norm": self.clipping_norm,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PrivacyBudgetEntry":
        """Create from dictionary."""
        return cls(
            round_id=data["round_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            epsilon_spent=data["epsilon_spent"],
            delta_spent=data["delta_spent"],
            noise_multiplier=data["noise_multiplier"],
            batch_size=data["batch_size"],
            dataset_size=data["dataset_size"],
            clipping_norm=data["clipping_norm"],
        )


class ClientPrivacyTracker:
    """Privacy budget tracker for a single client."""

    def __init__(
        self,
        client_id: str,
        epsilon_limit: float = 1.0,
        delta_limit: float = 1e-5,
        time_window: Optional[timedelta] = None,
        storage_path: Optional[str] = None,
    ):
        """
        Initialize client privacy tracker.

        Args:
            client_id: Client identifier
            epsilon_limit: Total epsilon budget limit
            delta_limit: Total delta budget limit
            time_window: Time window for budget reset (None = no reset)
            storage_path: Path to store budget history
        """
        self.client_id = client_id
        self.epsilon_limit = epsilon_limit
        self.delta_limit = delta_limit
        self.time_window = time_window
        self.storage_path = storage_path

        # Budget tracking
        self.budget_entries: List[PrivacyBudgetEntry] = []
        self.total_epsilon = 0.0
        self.total_delta = 0.0
        self.is_exhausted = False

        # Load existing budget if storage path provided
        if storage_path:
            self._load_budget_history()

        logger.info(
            f"Privacy tracker initialized for {client_id}: ε_limit={epsilon_limit}, δ_limit={delta_limit}"
        )

    def record_training_round(
        self,
        round_id: int,
        epsilon_spent: float,
        delta_spent: float,
        noise_multiplier: float,
        batch_size: int,
        dataset_size: int,
        clipping_norm: float,
    ) -> bool:
        """
        Record privacy expenditure for a training round.

        Args:
            round_id: Training round ID
            epsilon_spent: Epsilon spent in this round
            delta_spent: Delta spent in this round
            noise_multiplier: Noise multiplier used
            batch_size: Batch size used
            dataset_size: Local dataset size
            clipping_norm: Gradient clipping norm

        Returns:
            True if budget allows this expenditure
        """
        # Check if this would exceed budget
        new_epsilon = self.total_epsilon + epsilon_spent
        new_delta = self.total_delta + delta_spent

        if new_epsilon > self.epsilon_limit or new_delta > self.delta_limit:
            logger.warning(
                f"Client {self.client_id}: Budget exceeded! ε={new_epsilon:.6f}/{self.epsilon_limit}, δ={new_delta:.2e}/{self.delta_limit}"
            )
            self.is_exhausted = True
            return False

        # Record the expenditure
        entry = PrivacyBudgetEntry(
            round_id=round_id,
            timestamp=datetime.now(),
            epsilon_spent=epsilon_spent,
            delta_spent=delta_spent,
            noise_multiplier=noise_multiplier,
            batch_size=batch_size,
            dataset_size=dataset_size,
            clipping_norm=clipping_norm,
        )

        self.budget_entries.append(entry)
        self.total_epsilon = new_epsilon
        self.total_delta = new_delta

        # Check if budget is now exhausted
        if self.total_epsilon >= self.epsilon_limit * 0.95:  # 95% threshold
            logger.warning(
                f"Client {self.client_id}: Budget nearly exhausted ({self.total_epsilon:.6f}/{self.epsilon_limit})"
            )

        # Save to storage if configured
        if self.storage_path:
            self._save_budget_history()

        logger.info(
            f"Client {self.client_id} round {round_id}: ε={epsilon_spent:.6f} (total: {self.total_epsilon:.6f}/{self.epsilon_limit})"
        )

        return True

    def get_remaining_budget(self) -> Tuple[float, float]:
        """
        Get remaining privacy budget.

        Returns:
            Tuple of (remaining_epsilon, remaining_delta)
        """
        remaining_epsilon = max(0.0, self.epsilon_limit - self.total_epsilon)
        remaining_delta = max(0.0, self.delta_limit - self.total_delta)
        return remaining_epsilon, remaining_delta

    def can_participate(self, estimated_epsilon: float, estimated_delta: float) -> bool:
        """
        Check if client can participate in next round.

        Args:
            estimated_epsilon: Estimated epsilon for next round
            estimated_delta: Estimated delta for next round

        Returns:
            True if client can participate
        """
        remaining_epsilon, remaining_delta = self.get_remaining_budget()
        return (
            estimated_epsilon <= remaining_epsilon
            and estimated_delta <= remaining_delta
            and not self.is_exhausted
        )

    def reset_budget(
        self, new_epsilon_limit: Optional[float] = None, new_delta_limit: Optional[float] = None
    ):
        """
        Reset privacy budget (e.g., for new time window).

        Args:
            new_epsilon_limit: New epsilon limit (optional)
            new_delta_limit: New delta limit (optional)
        """
        if new_epsilon_limit is not None:
            self.epsilon_limit = new_epsilon_limit
        if new_delta_limit is not None:
            self.delta_limit = new_delta_limit

        # Archive old entries if storage configured
        if self.storage_path and self.budget_entries:
            self._archive_budget_history()

        # Reset tracking
        self.budget_entries.clear()
        self.total_epsilon = 0.0
        self.total_delta = 0.0
        self.is_exhausted = False

        logger.info(
            f"Client {self.client_id}: Budget reset to ε={self.epsilon_limit}, δ={self.delta_limit}"
        )

    def get_budget_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive budget summary.

        Returns:
            Dictionary with budget information
        """
        remaining_epsilon, remaining_delta = self.get_remaining_budget()

        return {
            "client_id": self.client_id,
            "epsilon_limit": self.epsilon_limit,
            "delta_limit": self.delta_limit,
            "total_epsilon": self.total_epsilon,
            "total_delta": self.total_delta,
            "remaining_epsilon": remaining_epsilon,
            "remaining_delta": remaining_delta,
            "is_exhausted": self.is_exhausted,
            "rounds_participated": len(self.budget_entries),
            "avg_epsilon_per_round": self.total_epsilon / max(1, len(self.budget_entries)),
            "budget_utilization": (
                self.total_epsilon / self.epsilon_limit if self.epsilon_limit > 0 else 0.0
            ),
        }

    def _load_budget_history(self):
        """Load budget history from storage."""
        if not self.storage_path or not os.path.exists(self.storage_path):
            return

        try:
            with open(self.storage_path, "r") as f:
                data = json.load(f)

            # Load entries
            self.budget_entries = [
                PrivacyBudgetEntry.from_dict(entry_data) for entry_data in data.get("entries", [])
            ]

            # Recompute totals
            self.total_epsilon = sum(entry.epsilon_spent for entry in self.budget_entries)
            self.total_delta = sum(entry.delta_spent for entry in self.budget_entries)

            # Check if budget exhausted
            self.is_exhausted = (
                self.total_epsilon >= self.epsilon_limit or self.total_delta >= self.delta_limit
            )

            logger.info(
                f"Loaded budget history for {self.client_id}: {len(self.budget_entries)} entries"
            )

        except Exception as e:
            logger.error(f"Failed to load budget history for {self.client_id}: {e}")

    def _save_budget_history(self):
        """Save budget history to storage."""
        if not self.storage_path:
            return

        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)

            data = {
                "client_id": self.client_id,
                "epsilon_limit": self.epsilon_limit,
                "delta_limit": self.delta_limit,
                "total_epsilon": self.total_epsilon,
                "total_delta": self.total_delta,
                "is_exhausted": self.is_exhausted,
                "last_updated": datetime.now().isoformat(),
                "entries": [entry.to_dict() for entry in self.budget_entries],
            }

            with open(self.storage_path, "w") as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save budget history for {self.client_id}: {e}")

    def _archive_budget_history(self):
        """Archive current budget history before reset."""
        if not self.storage_path:
            return

        archive_path = self.storage_path.replace(
            ".json", f'_archive_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        )

        try:
            data = {
                "client_id": self.client_id,
                "archived_at": datetime.now().isoformat(),
                "epsilon_limit": self.epsilon_limit,
                "delta_limit": self.delta_limit,
                "total_epsilon": self.total_epsilon,
                "total_delta": self.total_delta,
                "entries": [entry.to_dict() for entry in self.budget_entries],
            }

            with open(archive_path, "w") as f:
                json.dump(data, f, indent=2)

            logger.info(f"Archived budget history for {self.client_id}: {archive_path}")

        except Exception as e:
            logger.error(f"Failed to archive budget history for {self.client_id}: {e}")


class FederatedPrivacyManager:
    """Manages privacy budgets for all clients in federated learning."""

    def __init__(
        self,
        default_epsilon_limit: float = 1.0,
        default_delta_limit: float = 1e-5,
        storage_dir: str = "./privacy_budgets",
    ):
        """
        Initialize federated privacy manager.

        Args:
            default_epsilon_limit: Default epsilon limit for new clients
            default_delta_limit: Default delta limit for new clients
            storage_dir: Directory to store client budget files
        """
        self.default_epsilon_limit = default_epsilon_limit
        self.default_delta_limit = default_delta_limit
        self.storage_dir = storage_dir

        # Client trackers
        self.client_trackers: Dict[str, ClientPrivacyTracker] = {}

        # Ensure storage directory exists
        os.makedirs(storage_dir, exist_ok=True)

        logger.info(
            f"Federated privacy manager initialized: ε={default_epsilon_limit}, δ={default_delta_limit}"
        )

    def register_client(
        self,
        client_id: str,
        epsilon_limit: Optional[float] = None,
        delta_limit: Optional[float] = None,
    ) -> ClientPrivacyTracker:
        """
        Register a new client for privacy tracking.

        Args:
            client_id: Client identifier
            epsilon_limit: Client-specific epsilon limit
            delta_limit: Client-specific delta limit

        Returns:
            Client privacy tracker
        """
        if client_id in self.client_trackers:
            logger.warning(f"Client {client_id} already registered")
            return self.client_trackers[client_id]

        # Use provided limits or defaults
        eps_limit = epsilon_limit if epsilon_limit is not None else self.default_epsilon_limit
        delta_limit = delta_limit if delta_limit is not None else self.default_delta_limit

        # Create storage path
        storage_path = os.path.join(self.storage_dir, f"{client_id}_budget.json")

        # Create tracker
        tracker = ClientPrivacyTracker(
            client_id=client_id,
            epsilon_limit=eps_limit,
            delta_limit=delta_limit,
            storage_path=storage_path,
        )

        self.client_trackers[client_id] = tracker

        logger.info(f"Registered client {client_id} for privacy tracking")

        return tracker

    def record_client_round(
        self,
        client_id: str,
        round_id: int,
        epsilon_spent: float,
        delta_spent: float,
        noise_multiplier: float,
        batch_size: int,
        dataset_size: int,
        clipping_norm: float,
    ) -> bool:
        """
        Record privacy expenditure for a client's training round.

        Args:
            client_id: Client identifier
            round_id: Training round ID
            epsilon_spent: Epsilon spent
            delta_spent: Delta spent
            noise_multiplier: Noise multiplier used
            batch_size: Batch size
            dataset_size: Dataset size
            clipping_norm: Clipping norm

        Returns:
            True if expenditure was recorded (budget allows)
        """
        if client_id not in self.client_trackers:
            logger.error(f"Client {client_id} not registered for privacy tracking")
            return False

        tracker = self.client_trackers[client_id]
        return tracker.record_training_round(
            round_id,
            epsilon_spent,
            delta_spent,
            noise_multiplier,
            batch_size,
            dataset_size,
            clipping_norm,
        )

    def get_eligible_clients(self, estimated_epsilon: float, estimated_delta: float) -> List[str]:
        """
        Get list of clients eligible for next round.

        Args:
            estimated_epsilon: Estimated epsilon for next round
            estimated_delta: Estimated delta for next round

        Returns:
            List of eligible client IDs
        """
        eligible = []

        for client_id, tracker in self.client_trackers.items():
            if tracker.can_participate(estimated_epsilon, estimated_delta):
                eligible.append(client_id)

        logger.info(f"Eligible clients for next round: {len(eligible)}/{len(self.client_trackers)}")

        return eligible

    def get_global_budget_summary(self) -> Dict[str, Any]:
        """
        Get summary of all client budgets.

        Returns:
            Dictionary with global budget information
        """
        if not self.client_trackers:
            return {"total_clients": 0}

        summaries = [tracker.get_budget_summary() for tracker in self.client_trackers.values()]

        total_epsilon = sum(s["total_epsilon"] for s in summaries)
        total_rounds = sum(s["rounds_participated"] for s in summaries)
        exhausted_clients = sum(1 for s in summaries if s["is_exhausted"])

        return {
            "total_clients": len(self.client_trackers),
            "exhausted_clients": exhausted_clients,
            "active_clients": len(self.client_trackers) - exhausted_clients,
            "total_epsilon_spent": total_epsilon,
            "total_rounds": total_rounds,
            "avg_epsilon_per_client": total_epsilon / len(self.client_trackers),
            "avg_rounds_per_client": total_rounds / len(self.client_trackers),
            "client_summaries": {s["client_id"]: s for s in summaries},
        }

    def reset_all_budgets(self):
        """Reset privacy budgets for all clients."""
        for tracker in self.client_trackers.values():
            tracker.reset_budget()

        logger.info("Reset privacy budgets for all clients")

    def export_budget_report(self, output_path: str):
        """
        Export comprehensive budget report.

        Args:
            output_path: Path to save report
        """
        report = {
            "generated_at": datetime.now().isoformat(),
            "global_summary": self.get_global_budget_summary(),
            "client_details": {},
        }

        for client_id, tracker in self.client_trackers.items():
            report["client_details"][client_id] = {
                "summary": tracker.get_budget_summary(),
                "entries": [entry.to_dict() for entry in tracker.budget_entries],
            }

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Exported budget report to {output_path}")


if __name__ == "__main__":
    # Demo: Privacy budget tracking

    print("=== Privacy Budget Tracking Demo ===\n")

    # Initialize manager
    manager = FederatedPrivacyManager(
        default_epsilon_limit=1.0, default_delta_limit=1e-5, storage_dir="./demo_budgets"
    )

    # Register clients
    clients = ["hospital_A", "hospital_B", "hospital_C"]
    for client_id in clients:
        manager.register_client(client_id)

    print(f"Registered {len(clients)} clients")

    # Simulate training rounds
    print("\nSimulating federated training rounds:")

    for round_id in range(1, 11):
        print(f"\nRound {round_id}:")

        # Check eligible clients
        eligible = manager.get_eligible_clients(0.1, 1e-6)
        print(f"  Eligible clients: {eligible}")

        # Record training for eligible clients
        for client_id in eligible:
            success = manager.record_client_round(
                client_id=client_id,
                round_id=round_id,
                epsilon_spent=0.08,  # Spend 0.08 epsilon per round
                delta_spent=1e-6,
                noise_multiplier=1.0,
                batch_size=32,
                dataset_size=1000,
                clipping_norm=1.0,
            )

            if not success:
                print(f"    {client_id}: Budget exhausted!")

    # Final summary
    print("\nFinal Budget Summary:")
    summary = manager.get_global_budget_summary()
    print(f"  Total clients: {summary['total_clients']}")
    print(f"  Active clients: {summary['active_clients']}")
    print(f"  Exhausted clients: {summary['exhausted_clients']}")
    print(f"  Total epsilon spent: {summary['total_epsilon_spent']:.4f}")
    print(f"  Average rounds per client: {summary['avg_rounds_per_client']:.1f}")

    # Individual client details
    print("\nClient Details:")
    for client_id in clients:
        tracker = manager.client_trackers[client_id]
        client_summary = tracker.get_budget_summary()
        print(f"  {client_id}:")
        print(f"    Budget utilization: {client_summary['budget_utilization']:.1%}")
        print(f"    Remaining epsilon: {client_summary['remaining_epsilon']:.4f}")
        print(f"    Rounds participated: {client_summary['rounds_participated']}")

    # Export report
    manager.export_budget_report("./demo_budget_report.json")
    print("\nExported detailed budget report to ./demo_budget_report.json")

    print("\n=== Demo Complete ===")
