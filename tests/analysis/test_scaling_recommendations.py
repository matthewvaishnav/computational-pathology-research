"""
Unit tests for scaling recommendations generation.

Tests the generate_scaling_recommendations method and related functionality.
Requirements: 8.7, 8.8
"""

import pytest
import tempfile
import shutil
from pathlib import Path

from src.analysis.scalability import ScalabilityAnalyzer
from src.analysis.models import ScalabilityAnalysis


class TestScalingRecommendations:
    """Test suite for scaling recommendations generation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.project_path = Path(self.temp_dir)
        self.analyzer = ScalabilityAnalyzer(str(self.project_path))

    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_generate_recommendations_no_ddp(self):
        """Test recommendations when DDP is not implemented."""
        recommendations = self.analyzer.generate_scaling_recommendations(
            ddp_correct=False, bottlenecks=[], comm_overhead=0.0, scaling_efficiency="unknown"
        )

        # Should recommend implementing DDP
        assert "optimization_strategies" in recommendations
        assert len(recommendations["optimization_strategies"]) > 0

        # Check for DDP recommendation
        strategies = recommendations["optimization_strategies"]
        ddp_strategy = next(
            (s for s in strategies if s["category"] == "distributed_training"), None
        )
        assert ddp_strategy is not None
        assert "DDP" in ddp_strategy["recommendation"]

        # Speedup estimates should indicate DDP is required
        assert "speedup_estimates" in recommendations
        estimates = recommendations["speedup_estimates"]
        assert "2_gpus" in estimates
        assert estimates["2_gpus"]["speedup"] == "N/A"

    def test_generate_recommendations_data_loading_bottlenecks(self):
        """Test recommendations for data loading bottlenecks."""
        bottlenecks = [
            "DataLoader in train.py has num_workers=0",
            "DataLoader in train.py has pin_memory=False",
        ]

        recommendations = self.analyzer.generate_scaling_recommendations(
            ddp_correct=True,
            bottlenecks=bottlenecks,
            comm_overhead=10.0,
            scaling_efficiency="sub-linear",
        )

        # Should recommend data loading optimizations
        strategies = recommendations["optimization_strategies"]
        data_strategy = next((s for s in strategies if s["category"] == "data_loading"), None)
        assert data_strategy is not None
        assert "num_workers" in data_strategy["implementation"]
        assert "pin_memory" in data_strategy["implementation"]

    def test_generate_recommendations_high_communication_overhead(self):
        """Test recommendations for high communication overhead."""
        recommendations = self.analyzer.generate_scaling_recommendations(
            ddp_correct=True,
            bottlenecks=[],
            comm_overhead=75.0,  # High overhead
            scaling_efficiency="sub-linear",
        )

        # Should recommend gradient accumulation
        strategies = recommendations["optimization_strategies"]
        comm_strategy = next((s for s in strategies if s["category"] == "communication"), None)
        assert comm_strategy is not None
        assert "gradient accumulation" in comm_strategy["recommendation"].lower()

    def test_generate_recommendations_large_dataset_issues(self):
        """Test recommendations for large dataset handling issues."""
        bottlenecks = [
            "No streaming dataset support detected",
            "No WSI-specific optimizations detected",
        ]

        recommendations = self.analyzer.generate_scaling_recommendations(
            ddp_correct=True,
            bottlenecks=bottlenecks,
            comm_overhead=20.0,
            scaling_efficiency="sub-linear",
        )

        # Should recommend streaming and WSI optimizations
        strategies = recommendations["optimization_strategies"]
        dataset_strategy = next((s for s in strategies if s["category"] == "large_datasets"), None)
        assert dataset_strategy is not None
        assert "streaming" in dataset_strategy["implementation"].lower()
        assert "wsi" in dataset_strategy["implementation"].lower()

    def test_generate_recommendations_memory_bottlenecks(self):
        """Test recommendations for memory bottlenecks."""
        bottlenecks = [
            "Large tensor concatenation (15 occurrences in model.py)",
            "Explicit GPU transfers (20 occurrences in train.py)",
        ]

        recommendations = self.analyzer.generate_scaling_recommendations(
            ddp_correct=True,
            bottlenecks=bottlenecks,
            comm_overhead=15.0,
            scaling_efficiency="sub-linear",
        )

        # Should recommend memory optimizations
        strategies = recommendations["optimization_strategies"]
        memory_strategy = next(
            (s for s in strategies if s["category"] == "memory_optimization"), None
        )
        assert memory_strategy is not None
        assert (
            "gradient checkpointing" in memory_strategy["implementation"].lower()
            or "amp" in memory_strategy["implementation"].lower()
        )

    def test_speedup_estimates_linear_scaling(self):
        """Test speedup estimates for linear scaling."""
        recommendations = self.analyzer.generate_scaling_recommendations(
            ddp_correct=True, bottlenecks=[], comm_overhead=10.0, scaling_efficiency="linear"
        )

        estimates = recommendations["speedup_estimates"]

        # Check 2 GPU estimate
        assert "2_gpus" in estimates
        assert "speedup" in estimates["2_gpus"]
        assert "efficiency" in estimates["2_gpus"]

        # Linear scaling should have high efficiency
        efficiency_str = estimates["2_gpus"]["efficiency"]
        efficiency_pct = float(efficiency_str.rstrip("%"))
        assert efficiency_pct >= 90.0

        # Check 4 GPU estimate
        assert "4_gpus" in estimates

        # Check 8 GPU estimate
        assert "8_gpus" in estimates

    def test_speedup_estimates_sub_linear_scaling(self):
        """Test speedup estimates for sub-linear scaling."""
        bottlenecks = ["bottleneck1", "bottleneck2"]

        recommendations = self.analyzer.generate_scaling_recommendations(
            ddp_correct=True,
            bottlenecks=bottlenecks,
            comm_overhead=30.0,
            scaling_efficiency="sub-linear",
        )

        estimates = recommendations["speedup_estimates"]

        # Sub-linear scaling should have lower efficiency
        efficiency_str = estimates["2_gpus"]["efficiency"]
        efficiency_pct = float(efficiency_str.rstrip("%"))
        assert efficiency_pct < 90.0

        # Efficiency should degrade with more GPUs
        eff_2 = float(estimates["2_gpus"]["efficiency"].rstrip("%"))
        eff_4 = float(estimates["4_gpus"]["efficiency"].rstrip("%"))
        eff_8 = float(estimates["8_gpus"]["efficiency"].rstrip("%"))

        assert eff_2 >= eff_4 >= eff_8

    def test_granular_efficiency_classification(self):
        """Test granular efficiency classification."""
        # Excellent linear
        classification = self.analyzer._classify_scaling_efficiency_granular(
            ddp_correct=True, bottlenecks=[], comm_overhead=10.0
        )
        assert classification == "excellent_linear"

        # Good linear
        classification = self.analyzer._classify_scaling_efficiency_granular(
            ddp_correct=True, bottlenecks=["minor issue"], comm_overhead=30.0
        )
        assert classification == "good_linear"

        # Sub-linear moderate
        classification = self.analyzer._classify_scaling_efficiency_granular(
            ddp_correct=True, bottlenecks=["issue1", "issue2"], comm_overhead=50.0
        )
        assert classification == "sub_linear_moderate"

        # Sub-linear poor
        classification = self.analyzer._classify_scaling_efficiency_granular(
            ddp_correct=True,
            bottlenecks=["issue1", "issue2", "issue3", "issue4"],
            comm_overhead=70.0,
        )
        assert classification == "sub_linear_poor"

        # Non-scalable
        classification = self.analyzer._classify_scaling_efficiency_granular(
            ddp_correct=True, bottlenecks=["issue" + str(i) for i in range(10)], comm_overhead=150.0
        )
        assert classification == "non_scalable"

        # Unknown (no DDP)
        classification = self.analyzer._classify_scaling_efficiency_granular(
            ddp_correct=False, bottlenecks=[], comm_overhead=0.0
        )
        assert classification == "unknown"

    def test_priority_actions_generated(self):
        """Test that priority actions are generated."""
        bottlenecks = [
            "DataLoader in train.py has num_workers=0",
            "No streaming dataset support detected",
        ]

        recommendations = self.analyzer.generate_scaling_recommendations(
            ddp_correct=False,
            bottlenecks=bottlenecks,
            comm_overhead=60.0,
            scaling_efficiency="unknown",
        )

        # Should have priority actions
        assert "priority_actions" in recommendations
        assert len(recommendations["priority_actions"]) > 0

        # Priority actions should be strings
        for action in recommendations["priority_actions"]:
            assert isinstance(action, str)
            assert len(action) > 0

    def test_recommendations_structure(self):
        """Test that recommendations have the expected structure."""
        recommendations = self.analyzer.generate_scaling_recommendations(
            ddp_correct=True,
            bottlenecks=["test bottleneck"],
            comm_overhead=25.0,
            scaling_efficiency="sub-linear",
        )

        # Check top-level keys
        assert "efficiency_classification" in recommendations
        assert "optimization_strategies" in recommendations
        assert "speedup_estimates" in recommendations
        assert "priority_actions" in recommendations

        # Check strategy structure
        if len(recommendations["optimization_strategies"]) > 0:
            strategy = recommendations["optimization_strategies"][0]
            assert "category" in strategy
            assert "issue" in strategy
            assert "recommendation" in strategy
            assert "implementation" in strategy
            assert "expected_benefit" in strategy
            assert "effort" in strategy

        # Check speedup estimate structure
        for gpu_count in ["2_gpus", "4_gpus", "8_gpus"]:
            assert gpu_count in recommendations["speedup_estimates"]
            estimate = recommendations["speedup_estimates"][gpu_count]
            assert "speedup" in estimate
            assert "efficiency" in estimate
            assert "note" in estimate

    def test_integration_with_analyze(self):
        """Test that recommendations are included in analyze() output."""
        from unittest.mock import patch

        with patch.object(self.analyzer, "_verify_ddp_implementation", return_value=True):
            with patch.object(self.analyzer, "_identify_memory_bottlenecks", return_value=[]):
                with patch.object(
                    self.analyzer, "_detect_data_loading_bottlenecks", return_value=[]
                ):
                    with patch.object(
                        self.analyzer, "_assess_large_dataset_handling", return_value=[]
                    ):
                        with patch.object(
                            self.analyzer, "_estimate_communication_overhead", return_value=15.0
                        ):
                            result = self.analyzer.analyze()

        # Check that result includes recommendations
        assert isinstance(result, ScalabilityAnalysis)
        assert hasattr(result, "recommendations")
        assert isinstance(result.recommendations, dict)
        assert "efficiency_classification" in result.recommendations
        assert "optimization_strategies" in result.recommendations
        assert "speedup_estimates" in result.recommendations


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
