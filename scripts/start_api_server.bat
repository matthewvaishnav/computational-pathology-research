@echo off
REM Start Medical AI Platform API Server
REM This script starts the FastAPI server for integration testing

echo ========================================
echo Medical AI Platform - API Server
echo ========================================
echo.
echo Starting API server on http://localhost:8000
echo Press Ctrl+C to stop the server
echo.
echo API Documentation will be available at:
echo   - Swagger UI: http://localhost:8000/docs
echo   - ReDoc: http://localhost:8000/redoc
echo.
echo ========================================
echo.

python src/api/main.py
