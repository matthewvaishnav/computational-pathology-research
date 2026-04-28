@echo off
REM Run Integration Tests for Medical AI Platform
REM This script runs the comprehensive integration test suite

echo ========================================
echo Medical AI Platform - Integration Tests
echo ========================================
echo.

REM Check if API server is running
echo Checking if API server is running...
curl -s http://localhost:8000/health >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo ERROR: API server is not running!
    echo.
    echo Please start the API server first:
    echo   scripts\start_api_server.bat
    echo.
    echo Or run in another terminal window.
    echo.
    pause
    exit /b 1
)

echo API server is running - proceeding with tests
echo.
echo ========================================
echo.

REM Run the integration tests
python tests/integration/run_integration_tests.py %*

echo.
echo ========================================
echo Integration Tests Complete
echo ========================================
echo.
pause
