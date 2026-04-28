@echo off
REM Quick Integration Test for Medical AI Platform
REM This script runs a fast health check and basic validation

echo ========================================
echo Medical AI Platform - Quick Test
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

echo API server is running - proceeding with quick test
echo.
echo ========================================
echo.

REM Run the quick test
python tests/integration/quick_test.py %*

echo.
echo ========================================
echo Quick Test Complete
echo ========================================
echo.
pause
