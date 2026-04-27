#!/usr/bin/env python3
"""
Quick demo launcher for HistoCore Real-Time WSI Streaming

Usage:
    python scripts/run_demo.py --scenario speed
    python scripts/run_demo.py --scenario all
    python scripts/run_demo.py --interactive
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.streaming.demo_scenarios import DemoScenarioRunner, DemoScenario
from src.streaming.interactive_showcase import run_showcase


def main():
    parser = argparse.ArgumentParser(
        description="HistoCore Demo Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run speed demo
  python scripts/run_demo.py --scenario speed
  
  # Run all demos
  python scripts/run_demo.py --scenario all
  
  # Launch interactive showcase
  python scripts/run_demo.py --interactive
  
  # Multi-GPU demo
  python scripts/run_demo.py --scenario multi_gpu --gpus 0,1,2,3
        """
    )
    
    parser.add_argument(
        "--scenario",
        choices=["speed", "accuracy", "realtime", "pacs", "multi_gpu", "workflow", "all"],
        help="Demo scenario to run"
    )
    
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Launch interactive web showcase"
    )
    
    parser.add_argument(
        "--gpus",
        type=str,
        default="0",
        help="Comma-separated GPU IDs (default: 0)"
    )
    
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host for interactive showcase (default: 0.0.0.0)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for interactive showcase (default: 8000)"
    )
    
    args = parser.parse_args()
    
    # Parse GPU IDs
    gpu_ids = [int(x.strip()) for x in args.gpus.split(",")]
    
    if args.interactive:
        # Launch interactive showcase
        print("\n" + "="*60)
        print("LAUNCHING INTERACTIVE SHOWCASE")
        print("="*60)
        run_showcase(host=args.host, port=args.port, gpu_ids=gpu_ids)
    
    elif args.scenario:
        # Run specific scenario(s)
        runner = DemoScenarioRunner(gpu_ids=gpu_ids)
        
        if args.scenario == "all":
            # Run all scenarios
            asyncio.run(run_all_scenarios(runner))
        else:
            # Run single scenario
            scenario_map = {
                "speed": DemoScenario.SPEED_DEMO,
                "accuracy": DemoScenario.ACCURACY_DEMO,
                "realtime": DemoScenario.REALTIME_DEMO,
                "pacs": DemoScenario.PACS_DEMO,
                "multi_gpu": DemoScenario.MULTI_GPU_DEMO,
                "workflow": DemoScenario.CLINICAL_WORKFLOW
            }
            
            scenario = scenario_map[args.scenario]
            asyncio.run(runner.run_scenario(scenario))
    
    else:
        parser.print_help()


async def run_all_scenarios(runner: DemoScenarioRunner):
    """Run all demo scenarios"""
    scenarios = [
        DemoScenario.SPEED_DEMO,
        DemoScenario.ACCURACY_DEMO,
        DemoScenario.REALTIME_DEMO,
        DemoScenario.PACS_DEMO,
        DemoScenario.MULTI_GPU_DEMO,
        DemoScenario.CLINICAL_WORKFLOW
    ]
    
    print("\n" + "="*60)
    print("RUNNING ALL DEMO SCENARIOS")
    print("="*60)
    
    results = {}
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n[{i}/{len(scenarios)}] Running {scenario.value}...")
        result = await runner.run_scenario(scenario)
        results[scenario.value] = result
        
        if i < len(scenarios):
            print("\n" + "-"*60)
            await asyncio.sleep(2)
    
    print("\n" + "="*60)
    print("ALL DEMOS COMPLETE")
    print("="*60)
    print("\nSummary:")
    for scenario, result in results.items():
        print(f"  ✓ {scenario}")
    
    return results


if __name__ == "__main__":
    main()
