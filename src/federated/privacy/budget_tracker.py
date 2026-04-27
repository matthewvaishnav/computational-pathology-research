"""
Privacy budget tracking and monitoring for federated learning.

Provides comprehensive privacy budget management, composition analysis,
and real-time monitoring for medical AI federated learning systems.
"""

import json
import logging
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


@dataclass
class BudgetTransaction:
    """Represents a privacy budget transaction."""
    timestamp: float
    round_number: int
    epsilon_consumed: float
    delta_consumed: float
    remaining_epsilon: float
    remaining_delta: float
    client_count: int
    mechanism: str
    description: str = ""


@dataclass
class BudgetAlert:
    """Represents a privacy budget alert."""
    timestamp: float
    alert_type: str  # "warning", "critical", "exhausted"
    message: str
    remaining_epsilon: float
    remaining_delta: float
    threshold_triggered: float


class PrivacyBudgetTracker:
    """
    Comprehensive privacy budget tracking and monitoring.
    
    Features:
    - Real-time budget consumption tracking
    - Composition analysis and optimization
    - Alert system for budget thresholds
    - Historical analysis and reporting
    - Budget forecasting and planning
    """
    
    def __init__(
        self,
        total_epsilon: float,
        total_delta: float,
        warning_threshold: float = 0.8,
        critical_threshold: float = 0.95,
        save_path: Optional[str] = None
    ):
        """
        Initialize privacy budget tracker.
        
        Args:
            total_epsilon: Total epsilon budget
            total_delta: Total delta budget
            warning_threshold: Warning threshold (fraction of budget)
            critical_threshold: Critical threshold (fraction of budget)
            save_path: Path to save tracking data
        """
        self.total_epsilon = total_epsilon
        self.total_delta = total_delta
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.save_path = Path(save_path) if save_path else None
        
        # Current budget state
        self.consumed_epsilon = 0.0
        self.consumed_delta = 0.0
        self.round_count = 0
        
        # Transaction history
        self.transactions: List[BudgetTransaction] = []
        self.alerts: List[BudgetAlert] = []
        
        # Composition tracking
        self.composition_history: List[Dict] = []
        
        # Alert callbacks
        self.alert_callbacks = []
        
        logger.info(
            f"Initialized privacy budget tracker: "
            f"ε={total_epsilon}, δ={total_delta}, "
            f"warning={warning_threshold:.1%}, critical={critical_threshold:.1%}"
        )
    
    @property
    def remaining_epsilon(self) -> float:
        """Remaining epsilon budget."""
        return max(0.0, self.total_epsilon - self.consumed_epsilon)
    
    @property
    def remaining_delta(self) -> float:
        """Remaining delta budget."""
        return max(0.0, self.total_delta - self.consumed_delta)
    
    @property
    def epsilon_usage_rate(self) -> float:
        """Epsilon usage rate (0-1)."""
        return self.consumed_epsilon / self.total_epsilon if self.total_epsilon > 0 else 0.0
    
    @property
    def delta_usage_rate(self) -> float:
        """Delta usage rate (0-1)."""
        return self.consumed_delta / self.total_delta if self.total_delta > 0 else 0.0
    
    @property
    def is_exhausted(self) -> bool:
        """Check if budget is exhausted."""
        return (self.consumed_epsilon >= self.total_epsilon or 
                self.consumed_delta >= self.total_delta)
    
    def consume_budget(
        self,
        epsilon: float,
        delta: float,
        round_number: int,
        client_count: int,
        mechanism: str,
        description: str = ""
    ) -> bool:
        """
        Consume privacy budget and record transaction.
        
        Args:
            epsilon: Epsilon to consume
            delta: Delta to consume
            round_number: Round number
            client_count: Number of clients
            mechanism: Privacy mechanism used
            description: Optional description
            
        Returns:
            True if budget was consumed, False if insufficient
        """
        # Check if we have sufficient budget
        if (self.consumed_epsilon + epsilon > self.total_epsilon or
            self.consumed_delta + delta > self.total_delta):
            
            logger.warning(
                f"Insufficient privacy budget: "
                f"requested ε={epsilon:.6f}, δ={delta:.8f}, "
                f"remaining ε={self.remaining_epsilon:.6f}, δ={self.remaining_delta:.8f}"
            )
            return False
        
        # Consume budget
        self.consumed_epsilon += epsilon
        self.consumed_delta += delta
        self.round_count += 1
        
        # Record transaction
        transaction = BudgetTransaction(
            timestamp=time.time(),
            round_number=round_number,
            epsilon_consumed=epsilon,
            delta_consumed=delta,
            remaining_epsilon=self.remaining_epsilon,
            remaining_delta=self.remaining_delta,
            client_count=client_count,
            mechanism=mechanism,
            description=description
        )
        
        self.transactions.append(transaction)
        
        # Check for alerts
        self._check_budget_alerts()
        
        # Update composition history
        self._update_composition_history(epsilon, delta, mechanism)
        
        # Save if path specified
        if self.save_path:
            self._save_state()
        
        logger.info(
            f"Consumed privacy budget: ε={epsilon:.6f}, δ={delta:.8f}, "
            f"remaining ε={self.remaining_epsilon:.6f}, δ={self.remaining_delta:.8f}"
        )
        
        return True
    
    def _check_budget_alerts(self) -> None:
        """Check for budget threshold alerts."""
        epsilon_rate = self.epsilon_usage_rate
        delta_rate = self.delta_usage_rate
        max_rate = max(epsilon_rate, delta_rate)
        
        alert_type = None
        threshold = None
        
        if max_rate >= 1.0:
            alert_type = "exhausted"
            threshold = 1.0
        elif max_rate >= self.critical_threshold:
            alert_type = "critical"
            threshold = self.critical_threshold
        elif max_rate >= self.warning_threshold:
            alert_type = "warning"
            threshold = self.warning_threshold
        
        if alert_type:
            # Check if we already have a recent alert of this type
            recent_alerts = [
                a for a in self.alerts 
                if a.alert_type == alert_type and 
                time.time() - a.timestamp < 300  # 5 minutes
            ]
            
            if not recent_alerts:
                message = self._generate_alert_message(alert_type, max_rate, threshold)
                
                alert = BudgetAlert(
                    timestamp=time.time(),
                    alert_type=alert_type,
                    message=message,
                    remaining_epsilon=self.remaining_epsilon,
                    remaining_delta=self.remaining_delta,
                    threshold_triggered=threshold
                )
                
                self.alerts.append(alert)
                self._notify_alert(alert)
                
                logger.warning(f"Privacy budget alert: {message}")
    
    def _generate_alert_message(self, alert_type: str, usage_rate: float, threshold: float) -> str:
        """Generate alert message."""
        if alert_type == "exhausted":
            return f"Privacy budget exhausted! Usage: {usage_rate:.1%}"
        elif alert_type == "critical":
            return f"Critical privacy budget usage: {usage_rate:.1%} (threshold: {threshold:.1%})"
        elif alert_type == "warning":
            return f"Warning: High privacy budget usage: {usage_rate:.1%} (threshold: {threshold:.1%})"
        else:
            return f"Privacy budget alert: {usage_rate:.1%}"
    
    def _update_composition_history(self, epsilon: float, delta: float, mechanism: str) -> None:
        """Update composition analysis history."""
        composition_entry = {
            "timestamp": time.time(),
            "round": self.round_count,
            "epsilon": epsilon,
            "delta": delta,
            "mechanism": mechanism,
            "cumulative_epsilon": self.consumed_epsilon,
            "cumulative_delta": self.consumed_delta
        }
        
        self.composition_history.append(composition_entry)
    
    def _notify_alert(self, alert: BudgetAlert) -> None:
        """Notify registered callbacks of alert."""
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")
    
    def add_alert_callback(self, callback) -> None:
        """Add callback for budget alerts."""
        self.alert_callbacks.append(callback)
    
    def get_budget_status(self) -> Dict:
        """Get comprehensive budget status."""
        return {
            "total_epsilon": self.total_epsilon,
            "total_delta": self.total_delta,
            "consumed_epsilon": self.consumed_epsilon,
            "consumed_delta": self.consumed_delta,
            "remaining_epsilon": self.remaining_epsilon,
            "remaining_delta": self.remaining_delta,
            "epsilon_usage_rate": self.epsilon_usage_rate,
            "delta_usage_rate": self.delta_usage_rate,
            "rounds_completed": self.round_count,
            "is_exhausted": self.is_exhausted,
            "transactions_count": len(self.transactions),
            "alerts_count": len(self.alerts),
            "last_transaction": self.transactions[-1] if self.transactions else None
        }
    
    def get_usage_forecast(self, target_rounds: int) -> Dict:
        """
        Forecast budget usage for target number of rounds.
        
        Args:
            target_rounds: Target number of additional rounds
            
        Returns:
            Dictionary with forecast information
        """
        if not self.transactions:
            return {"error": "No transaction history for forecasting"}
        
        # Calculate average consumption per round
        recent_transactions = self.transactions[-10:]  # Last 10 rounds
        avg_epsilon_per_round = np.mean([t.epsilon_consumed for t in recent_transactions])
        avg_delta_per_round = np.mean([t.delta_consumed for t in recent_transactions])
        
        # Forecast consumption
        forecast_epsilon = avg_epsilon_per_round * target_rounds
        forecast_delta = avg_delta_per_round * target_rounds
        
        # Check feasibility
        epsilon_feasible = forecast_epsilon <= self.remaining_epsilon
        delta_feasible = forecast_delta <= self.remaining_delta
        feasible = epsilon_feasible and delta_feasible
        
        # Estimate maximum rounds possible
        max_rounds_epsilon = int(self.remaining_epsilon / avg_epsilon_per_round) if avg_epsilon_per_round > 0 else float('inf')
        max_rounds_delta = int(self.remaining_delta / avg_delta_per_round) if avg_delta_per_round > 0 else float('inf')
        max_rounds = min(max_rounds_epsilon, max_rounds_delta)
        
        return {
            "target_rounds": target_rounds,
            "avg_epsilon_per_round": avg_epsilon_per_round,
            "avg_delta_per_round": avg_delta_per_round,
            "forecast_epsilon_consumption": forecast_epsilon,
            "forecast_delta_consumption": forecast_delta,
            "epsilon_feasible": epsilon_feasible,
            "delta_feasible": delta_feasible,
            "feasible": feasible,
            "max_possible_rounds": max_rounds,
            "remaining_epsilon_after": self.remaining_epsilon - forecast_epsilon if feasible else None,
            "remaining_delta_after": self.remaining_delta - forecast_delta if feasible else None
        }
    
    def analyze_composition(self) -> Dict:
        """Analyze privacy composition across rounds."""
        if not self.composition_history:
            return {"error": "No composition history available"}
        
        # Group by mechanism
        mechanism_stats = {}
        for entry in self.composition_history:
            mechanism = entry["mechanism"]
            if mechanism not in mechanism_stats:
                mechanism_stats[mechanism] = {
                    "rounds": 0,
                    "total_epsilon": 0.0,
                    "total_delta": 0.0,
                    "avg_epsilon": 0.0,
                    "avg_delta": 0.0
                }
            
            stats = mechanism_stats[mechanism]
            stats["rounds"] += 1
            stats["total_epsilon"] += entry["epsilon"]
            stats["total_delta"] += entry["delta"]
        
        # Calculate averages
        for mechanism, stats in mechanism_stats.items():
            stats["avg_epsilon"] = stats["total_epsilon"] / stats["rounds"]
            stats["avg_delta"] = stats["total_delta"] / stats["rounds"]
        
        # Overall composition analysis
        total_rounds = len(self.composition_history)
        composition_efficiency = self.consumed_epsilon / (total_rounds * self.total_epsilon / 100) if total_rounds > 0 else 0
        
        return {
            "total_rounds": total_rounds,
            "mechanism_breakdown": mechanism_stats,
            "composition_efficiency": composition_efficiency,
            "average_epsilon_per_round": self.consumed_epsilon / total_rounds if total_rounds > 0 else 0,
            "average_delta_per_round": self.consumed_delta / total_rounds if total_rounds > 0 else 0
        }
    
    def get_recent_alerts(self, hours: int = 24) -> List[BudgetAlert]:
        """Get recent alerts within specified hours."""
        cutoff_time = time.time() - (hours * 3600)
        return [alert for alert in self.alerts if alert.timestamp > cutoff_time]
    
    def export_transaction_history(self, filepath: str) -> None:
        """Export transaction history to JSON file."""
        export_data = {
            "metadata": {
                "total_epsilon": self.total_epsilon,
                "total_delta": self.total_delta,
                "export_timestamp": time.time(),
                "export_date": datetime.now().isoformat()
            },
            "transactions": [asdict(t) for t in self.transactions],
            "alerts": [asdict(a) for a in self.alerts],
            "composition_history": self.composition_history
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Exported transaction history to {filepath}")
    
    def plot_budget_usage(self, save_path: Optional[str] = None) -> None:
        """Plot budget usage over time."""
        if not self.transactions:
            logger.warning("No transactions to plot")
            return
        
        # Extract data for plotting
        timestamps = [t.timestamp for t in self.transactions]
        epsilon_remaining = [t.remaining_epsilon for t in self.transactions]
        delta_remaining = [t.remaining_delta for t in self.transactions]
        
        # Convert timestamps to relative time (hours)
        start_time = timestamps[0]
        time_hours = [(t - start_time) / 3600 for t in timestamps]
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Epsilon plot
        ax1.plot(time_hours, epsilon_remaining, 'b-', linewidth=2, label='Remaining ε')
        ax1.axhline(y=self.total_epsilon * (1 - self.warning_threshold), 
                   color='orange', linestyle='--', alpha=0.7, label='Warning threshold')
        ax1.axhline(y=self.total_epsilon * (1 - self.critical_threshold), 
                   color='red', linestyle='--', alpha=0.7, label='Critical threshold')
        ax1.set_ylabel('Remaining Epsilon')
        ax1.set_title('Privacy Budget Usage Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Delta plot
        ax2.plot(time_hours, delta_remaining, 'g-', linewidth=2, label='Remaining δ')
        ax2.axhline(y=self.total_delta * (1 - self.warning_threshold), 
                   color='orange', linestyle='--', alpha=0.7, label='Warning threshold')
        ax2.axhline(y=self.total_delta * (1 - self.critical_threshold), 
                   color='red', linestyle='--', alpha=0.7, label='Critical threshold')
        ax2.set_xlabel('Time (hours)')
        ax2.set_ylabel('Remaining Delta')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved budget usage plot to {save_path}")
        else:
            plt.show()
    
    def _save_state(self) -> None:
        """Save tracker state to file."""
        if not self.save_path:
            return
        
        state_data = {
            "config": {
                "total_epsilon": self.total_epsilon,
                "total_delta": self.total_delta,
                "warning_threshold": self.warning_threshold,
                "critical_threshold": self.critical_threshold
            },
            "current_state": {
                "consumed_epsilon": self.consumed_epsilon,
                "consumed_delta": self.consumed_delta,
                "round_count": self.round_count,
                "last_updated": time.time()
            },
            "transactions": [asdict(t) for t in self.transactions[-100:]],  # Keep last 100
            "alerts": [asdict(a) for a in self.alerts[-50:]]  # Keep last 50
        }
        
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.save_path, 'w') as f:
            json.dump(state_data, f, indent=2)
    
    def load_state(self, filepath: str) -> None:
        """Load tracker state from file."""
        with open(filepath, 'r') as f:
            state_data = json.load(f)
        
        # Load current state
        current_state = state_data.get("current_state", {})
        self.consumed_epsilon = current_state.get("consumed_epsilon", 0.0)
        self.consumed_delta = current_state.get("consumed_delta", 0.0)
        self.round_count = current_state.get("round_count", 0)
        
        # Load transactions
        transaction_data = state_data.get("transactions", [])
        self.transactions = [BudgetTransaction(**t) for t in transaction_data]
        
        # Load alerts
        alert_data = state_data.get("alerts", [])
        self.alerts = [BudgetAlert(**a) for a in alert_data]
        
        logger.info(f"Loaded tracker state from {filepath}")
    
    def reset(self) -> None:
        """Reset tracker state."""
        self.consumed_epsilon = 0.0
        self.consumed_delta = 0.0
        self.round_count = 0
        self.transactions.clear()
        self.alerts.clear()
        self.composition_history.clear()
        
        logger.info("Reset privacy budget tracker")


if __name__ == "__main__":
    # Demo: Privacy budget tracking
    
    print("=== Privacy Budget Tracker Demo ===\n")
    
    # Create tracker
    tracker = PrivacyBudgetTracker(
        total_epsilon=1.0,
        total_delta=1e-5,
        warning_threshold=0.7,
        critical_threshold=0.9
    )
    
    # Add alert callback
    def alert_handler(alert: BudgetAlert):
        print(f"🚨 ALERT: {alert.message}")
    
    tracker.add_alert_callback(alert_handler)
    
    print("Budget tracker initialized")
    print(f"Initial status: {tracker.get_budget_status()}")
    
    # Simulate federated learning rounds
    mechanisms = ["gaussian", "laplace", "gaussian"]
    epsilons = [0.1, 0.15, 0.2, 0.25, 0.3]
    deltas = [1e-6, 2e-6, 1e-6, 3e-6, 2e-6]
    
    for i, (eps, delta, mech) in enumerate(zip(epsilons, deltas, mechanisms * 2)):
        round_num = i + 1
        client_count = np.random.randint(3, 8)
        
        print(f"\n--- Round {round_num} ---")
        success = tracker.consume_budget(
            epsilon=eps,
            delta=delta,
            round_number=round_num,
            client_count=client_count,
            mechanism=mech,
            description=f"Federated round with {client_count} clients"
        )
        
        if not success:
            print("Budget exhausted!")
            break
        
        # Show current status
        status = tracker.get_budget_status()
        print(f"Usage: ε={status['epsilon_usage_rate']:.1%}, δ={status['delta_usage_rate']:.1%}")
    
    # Show final analysis
    print(f"\n--- Final Analysis ---")
    
    # Budget status
    final_status = tracker.get_budget_status()
    print(f"Final budget status:")
    for key, value in final_status.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")
    
    # Composition analysis
    print(f"\nComposition analysis:")
    composition = tracker.analyze_composition()
    for key, value in composition.items():
        if key != "mechanism_breakdown":
            print(f"  {key}: {value}")
    
    print(f"\nMechanism breakdown:")
    for mechanism, stats in composition.get("mechanism_breakdown", {}).items():
        print(f"  {mechanism}: {stats['rounds']} rounds, "
              f"avg ε={stats['avg_epsilon']:.4f}, avg δ={stats['avg_delta']:.2e}")
    
    # Forecast
    print(f"\nForecast for 10 more rounds:")
    forecast = tracker.get_usage_forecast(10)
    for key, value in forecast.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")
    
    # Recent alerts
    recent_alerts = tracker.get_recent_alerts(24)
    print(f"\nRecent alerts ({len(recent_alerts)}):")
    for alert in recent_alerts:
        print(f"  {alert.alert_type}: {alert.message}")
    
    print("\n=== Demo Complete ===")