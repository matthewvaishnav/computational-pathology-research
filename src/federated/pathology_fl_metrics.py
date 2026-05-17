#!/usr/bin/env python3
"""
PathologyFL Performance Metrics - Track FL performance and efficiency
"""

import time
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class FLRoundMetrics:
    round_number: int
    participating_hospitals: int
    total_weight: float
    convergence_score: float
    communication_time: float
    aggregation_time: float


class PathologyFLMetrics:
    """Track PathologyFL performance metrics."""

    def __init__(self):
        self.round_metrics: List[FLRoundMetrics] = []
        self.hospital_contributions: Dict[str, List[float]] = {}

    def record_round(
        self,
        round_number: int,
        hospitals: Dict[str, float],
        convergence: float,
        comm_time: float,
        agg_time: float,
    ):
        """Record metrics for a federated learning round."""

        metrics = FLRoundMetrics(
            round_number=round_number,
            participating_hospitals=len(hospitals),
            total_weight=sum(hospitals.values()),
            convergence_score=convergence,
            communication_time=comm_time,
            aggregation_time=agg_time,
        )

        self.round_metrics.append(metrics)

        # Track individual hospital contributions
        for hospital_id, weight in hospitals.items():
            if hospital_id not in self.hospital_contributions:
                self.hospital_contributions[hospital_id] = []
            self.hospital_contributions[hospital_id].append(weight)

    def get_efficiency_report(self) -> Dict:
        """Generate efficiency report."""

        if not self.round_metrics:
            return {"error": "No metrics recorded"}

        avg_comm_time = sum(m.communication_time for m in self.round_metrics) / len(
            self.round_metrics
        )
        avg_agg_time = sum(m.aggregation_time for m in self.round_metrics) / len(self.round_metrics)
        avg_hospitals = sum(m.participating_hospitals for m in self.round_metrics) / len(
            self.round_metrics
        )

        return {
            "total_rounds": len(self.round_metrics),
            "avg_communication_time": f"{avg_comm_time:.3f}s",
            "avg_aggregation_time": f"{avg_agg_time:.3f}s",
            "avg_participating_hospitals": f"{avg_hospitals:.1f}",
            "pathology_fl_overhead": f"{avg_agg_time / avg_comm_time * 100:.1f}%",
        }

    def get_hospital_influence_report(self) -> Dict:
        """Generate hospital influence report."""

        influence_report = {}

        for hospital_id, weights in self.hospital_contributions.items():
            avg_weight = sum(weights) / len(weights)
            influence_report[hospital_id] = {
                "avg_weight": f"{avg_weight:.3f}",
                "rounds_participated": len(weights),
                "weight_stability": f"{(1 - (max(weights) - min(weights)) / avg_weight):.3f}",
            }

        return influence_report


# Example usage
def demo_metrics():
    """Demo PathologyFL metrics tracking."""

    metrics = PathologyFLMetrics()

    # Simulate 5 FL rounds
    for round_num in range(1, 6):
        hospitals = {
            "mayo_clinic": 2.5 + round_num * 0.1,
            "community_hospital": 1.0 + round_num * 0.05,
            "rural_clinic": 0.8 + round_num * 0.02,
        }

        convergence = 0.95 - round_num * 0.02
        comm_time = 2.0 + round_num * 0.1
        agg_time = 0.5 + round_num * 0.02

        metrics.record_round(round_num, hospitals, convergence, comm_time, agg_time)

    print("📊 PathologyFL Performance Report")
    print("=" * 40)

    efficiency = metrics.get_efficiency_report()
    for key, value in efficiency.items():
        print(f"{key}: {value}")

    print("\n🏥 Hospital Influence Report")
    print("-" * 40)

    influence = metrics.get_hospital_influence_report()
    for hospital, stats in influence.items():
        print(f"{hospital}:")
        for stat, value in stats.items():
            print(f"  {stat}: {value}")


if __name__ == "__main__":
    demo_metrics()
