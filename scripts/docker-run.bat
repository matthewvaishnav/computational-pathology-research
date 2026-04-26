@echo off
REM Docker run script for HistoCore streaming (Windows)

setlocal enabledelayedexpansion

REM Config
set IMAGE_NAME=histocore/streaming
set VERSION=%1
if "%VERSION%"=="" set VERSION=latest
set MODE=%2
if "%MODE%"=="" set MODE=dev

echo Starting HistoCore streaming...

REM Check GPU (basic check)
nvidia-smi >nul 2>&1
if errorlevel 1 (
    set GPU_FLAG=
    echo Warning: No NVIDIA GPU detected
) else (
    set GPU_FLAG=--gpus all
    echo GPU support enabled
)

REM Development mode
if "%MODE%"=="dev" (
    echo Mode: Development
    docker run -it --rm ^
        %GPU_FLAG% ^
        -p 8000:8000 ^
        -p 8001:8001 ^
        -p 8002:8002 ^
        -v %cd%/data:/app/data ^
        -v %cd%/logs:/app/logs ^
        -v %cd%/cache:/app/cache ^
        -e HISTOCORE_LOG_LEVEL=DEBUG ^
        %IMAGE_NAME%:%VERSION%
    goto :eof
)

REM Production mode
if "%MODE%"=="prod" (
    echo Mode: Production
    docker-compose up -d
    echo Services starting...
    echo Dashboard: http://localhost:8000
    echo Grafana: http://localhost:3000 ^(admin/histocore123^)
    echo Prometheus: http://localhost:9090
    goto :eof
)

REM Interactive shell
if "%MODE%"=="shell" (
    echo Mode: Interactive shell
    docker run -it --rm ^
        %GPU_FLAG% ^
        -v %cd%:/app ^
        %IMAGE_NAME%:%VERSION% ^
        /bin/bash
    goto :eof
)

echo Usage: %0 [version] [dev^|prod^|shell]