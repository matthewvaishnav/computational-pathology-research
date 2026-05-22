@echo off
REM PCam Federated Benchmark Execution Script (Windows)
REM Runs 4 strategies × 3 seeds × 30 rounds = 12 total runs
REM
REM Usage: scripts\federated\run_pcam_benchmark.bat
REM
REM Expected runtime: 4-6 hours (12 runs × ~20-30 min each)

setlocal enabledelayedexpansion

set ROUNDS=30
set NUM_SITES=5
set SEEDS=42 43 44
set STRATEGIES=equal volume prestige fair_weights_h

echo ==========================================
echo PCam Federated Benchmark
echo ==========================================
echo Configuration:
echo   Rounds: %ROUNDS%
echo   Sites: %NUM_SITES%
echo   Seeds: %SEEDS%
echo   Strategies: %STRATEGIES%
echo   Total runs: 12
echo ==========================================
echo.

REM Create results directory
if not exist results\pcam_federated_benchmark mkdir results\pcam_federated_benchmark

REM Run all combinations
for %%s in (%STRATEGIES%) do (
    for %%d in (%SEEDS%) do (
        echo ----------------------------------------
        echo Running: %%s (seed=%%d)
        echo ----------------------------------------
        
        python scripts\federated\run_pcam_federated_smoke.py ^
            --weighting %%s ^
            --rounds %ROUNDS% ^
            --num-sites %NUM_SITES% ^
            --seed %%d ^
            --output-dir results\pcam_federated_benchmark\%%s_seed%%d
        
        if errorlevel 1 (
            echo ERROR: Failed to run %%s (seed=%%d)
            exit /b 1
        )
        
        echo ✓ Completed: %%s (seed=%%d)
        echo.
    )
)

echo ==========================================
echo Benchmark Complete!
echo ==========================================
echo Results saved to: results\pcam_federated_benchmark\
echo.
echo Next step: Generate comparison report
echo   python scripts\federated\analyze_pcam_benchmark.py
