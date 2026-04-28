#!/bin/bash
# Medical AI Platform - Docker Deployment Script

set -e

echo "🚀 Medical AI Platform - Docker Deployment"
echo "=========================================="

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p logs/{api,worker,nginx}
mkdir -p data/{uploads,exports}
mkdir -p docker/ssl
mkdir -p docker/grafana/{dashboards,datasources}

# Copy environment file
if [ ! -f .env ]; then
    echo "📋 Copying environment configuration..."
    cp .env.docker .env
    echo "⚠️  Please edit .env file with your configuration before proceeding!"
    echo "   Especially change the SECRET_KEY and database passwords."
    read -p "Press Enter to continue after editing .env file..."
fi

# Build images
echo "🔨 Building Docker images..."
docker-compose build --no-cache

# Start services
echo "🚀 Starting services..."
docker-compose up -d

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 30

# Check service health
echo "🔍 Checking service health..."
services=("postgres" "redis" "api" "nginx")
for service in "${services[@]}"; do
    if docker-compose ps $service | grep -q "Up"; then
        echo "✅ $service is running"
    else
        echo "❌ $service failed to start"
        docker-compose logs $service
    fi
done

# Run database migrations (if needed)
echo "🗄️  Running database setup..."
docker-compose exec -T postgres psql -U medai -d medical_ai -c "SELECT version();" > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✅ Database is ready"
else
    echo "❌ Database connection failed"
fi

# Display access information
echo ""
echo "🎉 Deployment Complete!"
echo "======================"
echo "API Server:      http://localhost:8000"
echo "API Docs:        http://localhost:8000/docs"
echo "Grafana:         http://localhost:3000 (admin/admin123)"
echo "Prometheus:      http://localhost:9090"
echo "Database:        localhost:5432 (medai/medai123)"
echo ""
echo "📊 Service Status:"
docker-compose ps

echo ""
echo "📝 Next Steps:"
echo "1. Visit http://localhost:8000/docs to explore the API"
echo "2. Upload a test image to verify inference works"
echo "3. Check Grafana dashboards for monitoring"
echo "4. Review logs: docker-compose logs -f [service-name]"
echo ""
echo "🛑 To stop: docker-compose down"
echo "🔄 To restart: docker-compose restart"
echo "📋 To view logs: docker-compose logs -f"