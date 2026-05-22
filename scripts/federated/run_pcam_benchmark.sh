#!/bin/bash
# PCam Federated Benchmark Execution Script
# Runs 4 strategies × 3 seeds × 30 rounds = 12 total runs
#
# Usage: bash scripts/federated/run_pcam_benchmark.sh
#
# Expected runtime: 4-6 hours (12 runs × ~20-30 min each)

set -e  # Exit on error

ROUNDS=30
NUM_SITES=5
SEEDS=(42 43 44)
STRATEGIES=("equal" "volume" "prestige" "fair_weights_h")

echo "=========================================="
echo "PCam Federated Benchmark"
echo "=========================================="
echo "Configuration:"
echo "  Rounds: $ROUNDS"
echo "  Sites: $NUM_SITES"
echo "  Seeds: ${SEEDS[@]}"
echo "  Strategies: ${STRATEGIES[@]}"
echo "  Total runs: $((${#STRATEGIES[@]} * ${#SEEDS[@]}))"
echo "=========================================="
echo ""

# Create results directory
mkdir -p results/pcam_federated_benchmark

# Run all combinations
for strategy in "${STRATEGIES[@]}"; do
    for seed in "${SEEDS[@]}"; do
        echo "----------------------------------------"
        echo "Running: $strategy (seed=$seed)"
        echo "----------------------------------------"
        
        python scripts/federated/run_pcam_federated_smoke.py \
            --weighting "$strategy" \
            --rounds "$ROUNDS" \
            --num-sites "$NUM_SITES" \
            --seed "$seed" \
            --output-dir "results/pcam_federated_benchmark/${strategy}_seed${seed}"
        
        echo "✓ Completed: $strategy (seed=$seed)"
        echo ""
    done
done

echo "=========================================="
echo "Benchmark Complete!"
echo "=========================================="
echo "Results saved to: results/pcam_federated_benchmark/"
echo ""
echo "Next step: Generate comparison report"
echo "  python scripts/federated/analyze_pcam_benchmark.py"
