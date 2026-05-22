# PCam Federated Benchmark Execution Script (PowerShell)
# Runs 4 strategies x 3 seeds x 30 rounds = 12 total runs
# Expected runtime: 3-4 hours

$ErrorActionPreference = "Stop"

# Set PYTHONPATH to project root
$env:PYTHONPATH = "c:\Users\matth\Desktop\computational-pathology-research"

$ROUNDS = 30
$NUM_SITES = 5
$SEEDS = @(42, 43, 44)
$STRATEGIES = @("equal", "volume", "prestige", "fair_weights_h")

Write-Host "=========================================="
Write-Host "PCam Federated Benchmark"
Write-Host "=========================================="
Write-Host "Configuration:"
Write-Host "  Rounds: $ROUNDS"
Write-Host "  Sites: $NUM_SITES"
Write-Host "  Seeds: $($SEEDS -join ', ')"
Write-Host "  Strategies: $($STRATEGIES -join ', ')"
Write-Host "  Total runs: $($STRATEGIES.Count * $SEEDS.Count)"
Write-Host "=========================================="
Write-Host ""

# Create results directory
$resultsDir = "results/pcam_federated_benchmark"
if (-not (Test-Path $resultsDir)) {
    New-Item -ItemType Directory -Path $resultsDir | Out-Null
}

# Track progress
$totalRuns = $STRATEGIES.Count * $SEEDS.Count
$currentRun = 0

# Run all combinations
foreach ($strategy in $STRATEGIES) {
    foreach ($seed in $SEEDS) {
        $currentRun++
        
        Write-Host "----------------------------------------"
        Write-Host "Run $currentRun/$totalRuns : $strategy (seed=$seed)"
        Write-Host "----------------------------------------"
        
        $outputDir = "results/pcam_federated_benchmark/${strategy}_seed${seed}"
        
        python scripts/federated/run_pcam_federated_smoke.py --weighting $strategy --rounds $ROUNDS --num-sites $NUM_SITES --seed $seed --output-dir $outputDir
        
        if ($LASTEXITCODE -ne 0) {
            Write-Host "ERROR: Failed to run $strategy (seed=$seed)" -ForegroundColor Red
            exit 1
        }
        
        Write-Host "Completed: $strategy (seed=$seed)" -ForegroundColor Green
        Write-Host ""
    }
}

Write-Host "=========================================="
Write-Host "Benchmark Complete!" -ForegroundColor Green
Write-Host "=========================================="
Write-Host "Results saved to: $resultsDir"
Write-Host ""
Write-Host "Next step: Generate comparison report"
Write-Host "  python scripts/federated/analyze_pcam_benchmark.py"
