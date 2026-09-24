@echo off
setlocal

REM Start the local computational-pathology research stack.
REM This is development infrastructure, not a clinical or production deployment.

docker --version >nul 2>&1
if errorlevel 1 (
    echo Docker is required.
    exit /b 1
)

docker compose version >nul 2>&1
if errorlevel 1 (
    echo Docker Compose v2 ^('docker compose'^) is required.
    exit /b 1
)

if not exist ".env" (
    copy /Y .env.docker.example .env >nul
    echo Created .env from .env.docker.example.
    echo Replace every CHANGE_ME value, then run this script again.
    exit /b 1
)

findstr /C:"CHANGE_ME" .env >nul
if not errorlevel 1 (
    echo .env still contains CHANGE_ME placeholders. Refusing to start.
    exit /b 1
)

if not exist "logs\api" mkdir logs\api
if not exist "logs\nginx" mkdir logs\nginx
if not exist "data\uploads" mkdir data\uploads
if not exist "data\exports" mkdir data\exports

echo Validating Compose configuration...
docker compose config >nul
if errorlevel 1 exit /b 1

echo Building and starting local research services...
docker compose up -d --build
if errorlevel 1 exit /b 1

echo.
echo Local stack:
echo   API:        http://localhost:8000
echo   API docs:   http://localhost:8000/docs
echo   Grafana:    http://localhost:3000
echo   Prometheus: http://localhost:9090
echo.
docker compose ps
echo.
echo Stop with: docker compose down
