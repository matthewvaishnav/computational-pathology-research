@echo off
REM Docker build script for HistoCore streaming (Windows)

setlocal enabledelayedexpansion

REM Config
set IMAGE_NAME=histocore/streaming
set VERSION=%1
if "%VERSION%"=="" set VERSION=latest
set BUILD_TARGET=%2
if "%BUILD_TARGET%"=="" set BUILD_TARGET=production

echo Building HistoCore Docker image...
echo Image: %IMAGE_NAME%:%VERSION%
echo Target: %BUILD_TARGET%

REM Check Docker
docker --version >nul 2>&1
if errorlevel 1 (
    echo Error: Docker not found. Install Docker Desktop first.
    exit /b 1
)

REM Build image
echo Building Docker image...
docker build ^
    --target %BUILD_TARGET% ^
    --tag %IMAGE_NAME%:%VERSION% ^
    --tag %IMAGE_NAME%:latest ^
    --build-arg BUILDKIT_INLINE_CACHE=1 ^
    .

if errorlevel 1 (
    echo Build failed
    exit /b 1
)

echo Build successful
docker images | findstr %IMAGE_NAME%

REM Optional test
if "%3"=="test" (
    echo Running quick test...
    docker run --rm %IMAGE_NAME%:%VERSION% python -c "import src.streaming; print('Import successful')"
)

echo Done. Run with: docker-compose up