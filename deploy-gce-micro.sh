#!/bin/bash
# Google Cloud Engine e2-micro deployment script

set -e

echo "🚀 Starting GCE e2-micro deployment..."

# Check if .env file exists
if [ ! -f .env ]; then
    echo "❌ Error: .env file not found!"
    echo "Please create a .env file with your GOOGLE_API_KEY"
    exit 1
fi

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Installing Docker..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker $USER
    echo "✅ Docker installed. Please log out and log back in to apply group changes."
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Installing Docker Compose..."
    sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
    echo "✅ Docker Compose installed."
fi

# Stop existing containers
echo "🛑 Stopping existing containers..."
docker-compose -f docker-compose.gce-micro.yml down --remove-orphans || true

# Clean up old images (optional, saves disk space)
echo "🧹 Cleaning up old Docker images..."
docker image prune -f || true

# Build and start the application
echo "🏗️ Building and starting the application..."
docker-compose -f docker-compose.gce-micro.yml up --build -d

# Wait for the application to start
echo "⏳ Waiting for application to start..."
sleep 30

# Check if the application is running
echo "🔍 Checking application status..."
if curl -f http://localhost:5000/ > /dev/null 2>&1; then
    echo "✅ Application is running successfully!"
    echo "🌐 Your RAG application is available at:"
    echo "   - Local: http://localhost:5000"
    echo "   - External: http://$(curl -s ifconfig.me):80"
else
    echo "❌ Application failed to start. Checking logs..."
    docker-compose -f docker-compose.gce-micro.yml logs --tail=50
    exit 1
fi

# Display system resource usage
echo "📊 Current system resources:"
echo "Memory usage:"
free -h
echo "CPU usage:"
top -bn1 | grep "Cpu(s)" | awk '{print $2 + $4"%"}'

# Display running containers
echo "📦 Running containers:"
docker ps

echo "✅ Deployment complete!"
echo "🔧 To view logs: docker-compose -f docker-compose.gce-micro.yml logs -f"
echo "🛑 To stop: docker-compose -f docker-compose.gce-micro.yml down" 