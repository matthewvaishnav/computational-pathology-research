@echo off
REM Windows batch file wrapper for Makefile commands
REM Usage: make.bat <target>

setlocal enabledelayedexpansion

if "%1"=="" goto help
if "%1"=="help" goto help
if "%1"=="install" goto install
if "%1"=="test" goto test
if "%1"=="demo" goto demo
if "%1"=="docker-build" goto docker-build
if "%1"=="clean" goto clean

echo Unknown target: %1
echo Run 'make.bat help' for available targets
exit /b 1

:help
echo Computational Pathology Research Framework
echo ===========================================
echo.
echo Available targets:
echo   install        - Install production dependencies
echo   install-dev    - Install development dependencies
echo   test           - Run tests
echo   test-cov       - Run tests with coverage
echo   lint           - Run linting checks
echo   format         - Format code
echo   demo           - Run quick demo
echo   train          - Train model
echo   evaluate       - Evaluate model
echo   docker-build   - Build Docker image
echo   docker-run     - Run Docker container
echo   docker-stop    - Stop Docker container
echo   clean          - Clean generated files
echo.
echo Quick Start:
echo   make.bat install
echo   make.bat test
echo   make.bat demo
echo.
goto end

:install
echo Installing production dependencies...
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
echo Installation complete
goto end

:install-dev
echo Installing development dependencies...
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install -e .
echo Development installation complete
goto end

:test
echo Running tests...
pytest tests/ -v
goto end

:test-cov
echo Running tests with coverage...
pytest tests/ -v --cov=src --cov-report=term --cov-report=html
echo Coverage report generated in htmlcov/
goto end

:lint
echo Running linting checks...
flake8 src/ tests/
black --check src/ tests/
isort --check-only src/ tests/
goto end

:format
echo Formatting code...
black src/ tests/
isort src/ tests/
echo Code formatted
goto end

:demo
echo Running quick demo...
python run_quick_demo.py
echo Demo complete - check results/quick_demo/
goto end

:train
echo Training model...
python experiments/train.py --config-name default
goto end

:evaluate
echo Evaluating model...
python experiments/evaluate.py --checkpoint checkpoints/best_model.pth --data-dir ./data
goto end

:docker-build
echo Building Docker image...
docker build -t pathology-api:latest .
echo Docker image built
goto end

:docker-run
echo Starting Docker container...
docker-compose up -d api
echo Container running at http://localhost:8000
goto end

:docker-stop
echo Stopping Docker container...
docker-compose down
echo Container stopped
goto end

:clean
echo Cleaning generated files...
for /d /r . %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d"
for /d /r . %%d in (*.egg-info) do @if exist "%%d" rd /s /q "%%d"
for /d /r . %%d in (.pytest_cache) do @if exist "%%d" rd /s /q "%%d"
for /d /r . %%d in (.mypy_cache) do @if exist "%%d" rd /s /q "%%d"
del /s /q *.pyc 2>nul
del /s /q *.pyo 2>nul
if exist htmlcov rd /s /q htmlcov
if exist dist rd /s /q dist
if exist build rd /s /q build
echo Cleanup complete
goto end

:end
endlocal
