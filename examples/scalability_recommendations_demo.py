"""
Demonstration of Scalability Recommendations Generation.

This script demonstrates the scaling recommendations feature of the
ScalabilityAnalyzer, showing how it generates actionable optimization
strategies and speedup estimates for multi-GPU configurations.
"""

import json
from pathlib import Path
from src.analysis.scalability import ScalabilityAnalyzer


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def print_recommendations(recommendations: dict):
    """Pretty print recommendations."""
    
    # Efficiency Classification
    print_section("Scaling Efficiency Classification")
    classification = recommendations['efficiency_classification']
    print(f"Classification: {classification.upper().replace('_', ' ')}")
    
    # Optimization Strategies
    if recommendations['optimization_strategies']:
        print_section("Optimization Strategies")
        for i, strategy in enumerate(recommendations['optimization_strategies'], 1):
            print(f"{i}. {strategy['category'].upper().replace('_', ' ')}")
            print(f"   Issue: {strategy['issue']}")
            print(f"   Recommendation: {strategy['recommendation']}")
            print(f"   Expected Benefit: {strategy['expected_benefit']}")
            print(f"   Effort: {strategy['effort']}")
            print(f"   Implementation:")
            for line in strategy['implementation'].split('. '):
                if line.strip():
                    print(f"     • {line.strip()}")
            print()
    
    # Priority Actions
    if recommendations['priority_actions']:
        print_section("Priority Actions")
        for i, action in enumerate(recommendations['priority_actions'], 1):
            print(f"{i}. {action}")
    
    # Speedup Estimates
    print_section("Multi-GPU Speedup Estimates")
    print(f"{'GPU Count':<12} {'Speedup':<12} {'Efficiency':<12} {'Note'}")
    print("-" * 80)
    for gpu_key in ['2_gpus', '4_gpus', '8_gpus']:
        estimate = recommendations['speedup_estimates'][gpu_key]
        gpu_count = gpu_key.split('_')[0]
        print(f"{gpu_count + ' GPUs':<12} {estimate['speedup']:<12} {estimate['efficiency']:<12} {estimate['note']}")


def demo_scenario_1():
    """Scenario 1: No DDP implementation."""
    print_section("SCENARIO 1: No DDP Implementation")
    print("Simulating a project without DistributedDataParallel...")
    
    analyzer = ScalabilityAnalyzer(".")
    recommendations = analyzer.generate_scaling_recommendations(
        ddp_correct=False,
        bottlenecks=[],
        comm_overhead=0.0,
        scaling_efficiency="unknown"
    )
    
    print_recommendations(recommendations)


def demo_scenario_2():
    """Scenario 2: DDP with data loading bottlenecks."""
    print_section("SCENARIO 2: DDP with Data Loading Bottlenecks")
    print("Simulating a project with DDP but poor DataLoader configuration...")
    
    analyzer = ScalabilityAnalyzer(".")
    bottlenecks = [
        "DataLoader in train.py has num_workers=0 (should be >0 for multi-GPU, recommended: 4-8)",
        "DataLoader in train.py has pin_memory=False (should be True for GPU training)"
    ]
    
    recommendations = analyzer.generate_scaling_recommendations(
        ddp_correct=True,
        bottlenecks=bottlenecks,
        comm_overhead=15.0,
        scaling_efficiency="sub-linear"
    )
    
    print_recommendations(recommendations)


def demo_scenario_3():
    """Scenario 3: High communication overhead."""
    print_section("SCENARIO 3: High Communication Overhead")
    print("Simulating a project with high gradient synchronization overhead...")
    
    analyzer = ScalabilityAnalyzer(".")
    recommendations = analyzer.generate_scaling_recommendations(
        ddp_correct=True,
        bottlenecks=[],
        comm_overhead=85.0,  # High overhead
        scaling_efficiency="sub-linear"
    )
    
    print_recommendations(recommendations)


def demo_scenario_4():
    """Scenario 4: Excellent scaling (optimal configuration)."""
    print_section("SCENARIO 4: Excellent Scaling (Optimal Configuration)")
    print("Simulating a well-optimized project with near-linear scaling...")
    
    analyzer = ScalabilityAnalyzer(".")
    recommendations = analyzer.generate_scaling_recommendations(
        ddp_correct=True,
        bottlenecks=[],
        comm_overhead=12.0,
        scaling_efficiency="linear"
    )
    
    print_recommendations(recommendations)


def demo_scenario_5():
    """Scenario 5: Multiple bottlenecks."""
    print_section("SCENARIO 5: Multiple Bottlenecks")
    print("Simulating a project with multiple scalability issues...")
    
    analyzer = ScalabilityAnalyzer(".")
    bottlenecks = [
        "DataLoader in train.py has num_workers=0",
        "No streaming dataset support detected (IterableDataset, StreamingDataset)",
        "No WSI-specific optimizations detected (OpenSlide, tile-based loading)",
        "Large tensor concatenation (15 occurrences in model.py)",
        "Explicit GPU transfers (20 occurrences in train.py)"
    ]
    
    recommendations = analyzer.generate_scaling_recommendations(
        ddp_correct=True,
        bottlenecks=bottlenecks,
        comm_overhead=65.0,
        scaling_efficiency="sub-linear"
    )
    
    print_recommendations(recommendations)


def main():
    """Run all demonstration scenarios."""
    print("\n" + "=" * 80)
    print("  SCALABILITY RECOMMENDATIONS DEMONSTRATION")
    print("  HistoCore Project Optimization Analysis System")
    print("=" * 80)
    
    # Run all scenarios
    demo_scenario_1()
    demo_scenario_2()
    demo_scenario_3()
    demo_scenario_4()
    demo_scenario_5()
    
    print("\n" + "=" * 80)
    print("  DEMONSTRATION COMPLETE")
    print("=" * 80 + "\n")
    
    print("Key Takeaways:")
    print("1. The analyzer provides granular efficiency classifications")
    print("2. Optimization strategies are specific and actionable")
    print("3. Speedup estimates help plan multi-GPU deployments")
    print("4. Priority actions guide implementation order")
    print("\nFor more information, see:")
    print("  - src/analysis/scalability.py")
    print("  - tests/analysis/test_scaling_recommendations.py")
    print("  - .kiro/specs/project-optimization-analysis/")


if __name__ == "__main__":
    main()
