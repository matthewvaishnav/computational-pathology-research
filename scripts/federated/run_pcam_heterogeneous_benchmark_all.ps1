# PCam Heterogeneous Benchmark - All Strategies
# 4 strategies x 3 seeds x 30 rounds = 12 runs
# Expected runtime: ~1 hour

$ErrorActionPreference = "Stop"

$env:PYTHONPATH = "c:\Users\matth\Desktop\computational-pathology-research"

$ROUNDS = 30
$SEEDS = @(42, 43, 44)
$STRATEGIES = @("equal", "volume", "prestige", "fair_weights_h")

Write-Host "=========================================="
Write-Host "PCam Heterogeneous Benchmark"
Write-Host "=========================================="
Write-Host "Rounds: $ROUNDS"
Write-Host "Seeds: $($SEEDS -join ', ')"
Write-Host "Strategies: $($STRATEGIES -join ', ')"
Write-Host "Total runs: $($STRATEGIES.Count * $SEEDS.Count)"
Write-Host "=========================================="
Write-Host ""

$resultsDir = "results/pcam_heterogeneous_benchmark"
if (-not (Test-Path $resultsDir)) {
    New-Item -ItemType Directory -Path $resultsDir | Out-Null
}

$totalRuns = $STRATEGIES.Count * $SEEDS.Count
$currentRun = 0

foreach ($strategy in $STRATEGIES) {
    foreach ($seed in $SEEDS) {
        $currentRun++
        
        Write-Host "----------------------------------------"
        Write-Host "Run $currentRun/$totalRuns : $strategy (seed=$seed)"
        Write-Host "----------------------------------------"
        
        python scripts/federated/run_pcam_heterogeneous_benchmark.py --weighting $strategy --rounds $ROUNDS --seed $seed --output-dir $resultsDir
        
        if ($LASTEXITCODE -ne 0) {
            Write-Host "ERROR: Failed $strategy (seed=$seed)" -ForegroundColor Red
            exit 1
        }
        
        Write-Host "Completed: $strategy (seed=$seed)" -ForegroundColor Green
        Write-Host ""
    }
}

Write-Host "=========================================="
Write-Host "Benchmark Complete!" -ForegroundColor Green
Write-Host "=========================================="
Write-Host "Results: $resultsDir"
Write-Host ""
Write-Host "Next: python scripts/federated/analyze_pcam_heterogeneous_benchmark.py"
