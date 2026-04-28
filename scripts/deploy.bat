@echo off
REM Medical AI Platform - Docker Deployment Script (Windows)

echo 🚀 Medical AI Platform - Docker Deployment
echo ==========================================

REM Check if Docker is installed
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Docker is not installed. Please install Docker Desktop first.
    pause
    exit /b 1
)

REM Check if Docker Compose is installed
docker-compose --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Docker Compose is not installed. Please install Docker Compose first.
    pause
    exit /b 1
)

REM Create necessary directories
echo 📁 Creating directories...
if not exist "logs\api" mkdir logs\api
if not exist "logs\worker" mkdir logs\worker
if not exist "logs\nginx" mkdir logs\nginx
if not exist "data\uploads" mkdir data\uploads
if not exist "data\exports" mkdir data\exports
if not exist "docker\ssl" mkdir docker\ssl

REM Copy environment file
if not exist ".env" (
    echo 📋 Copying environment configuration...
    copy .env.docker .env
    echo ⚠️  Please edit .env file with your configuration before proceeding!
    echo    Especially change the SECRET_KEY and database passwords.
    pause
)

REM Build images
echo 🔨 Building Docker images...
docker-compose build --no-cache

REM Start services
echo 🚀 Starting services...
docker-compose up -d

REM Wait for services to be ready
echo ⏳ Waiting for services to start...
timeout /t 30 /nobreak >nul

REM Check service health
echo 🔍 Checking service health...
docker-compose ps

REM Display access information
echo.
echo 🎉 Deployment Complete!
echo ======================
echo API Server:      http://localhost:8000
echo API Docs:        http://localhost:8000/docs
echo Grafana:         http://localhost:3000 (admin/admin123)
echo Prometheus:      http://localhost:9090
echo Database:        localhost:5432 (medai/medai123)
echo.
echo 📊 Service Status:
docker-compose ps

echo.
echo 📝 Next Steps:
echo 1. Visit http://localhost:8000/docs to explore the API
echo 2. Upload a test image to verify inference works
echo 3. Check Grafana dashboards for monitoring
echo 4. Review logs: docker-compose logs -f [service-name]
echo.
echo 🛑 To stop: docker-compose down
echo 🔄 To restart: docker-compose restart
echo 📋 To view logs: docker-compose logs -f

pause