#!/bin/bash
# Universal start script for LangChain RAG Demo

# Set default environment if not specified
export ENV=${ENV:-development}
export PORT=${PORT:-5000}

echo "🚀 Starting LangChain RAG Demo in $ENV mode on port $PORT"

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file..."
    cat > .env << EOF
# Add your environment variables here
# GOOGLE_API_KEY=your_gemini_api_key_here
EOF
    echo "⚠️  Please add your GOOGLE_API_KEY to the .env file"
fi

# Choose startup method based on environment
case $ENV in
    development|dev)
        echo "🔧 Starting in development mode"
        export FLASK_ENV=development
        export FLASK_DEBUG=1
        if [ -f .venv/bin/activate ]; then
            source .venv/bin/activate
        fi
        python -m flask --app main run --host=0.0.0.0 --port=$PORT --debug
        ;;
    production|prod)
        echo "🏭 Starting in production mode"
        export ENV=production
        if [ -f .venv/bin/activate ]; then
            source .venv/bin/activate
            gunicorn --config gunicorn.conf.py wsgi:app
        else
            echo "Using Docker..."
            docker compose up --build -d
        fi
        ;;
    micro)
        echo "🐣 Starting in micro mode (for small servers)"
        export ENV=micro
        export CPU_LIMIT=0.8
        export MEMORY_LIMIT=800M
        export CPU_RESERVATION=0.2
        export MEMORY_RESERVATION=200M
        if [ -f .venv/bin/activate ]; then
            source .venv/bin/activate
            gunicorn --config gunicorn.conf.py wsgi:app
        else
            echo "Using Docker..."
            docker compose up --build -d
        fi
        ;;
    docker)
        echo "🐳 Starting with Docker"
        docker compose up --build -d
        ;;
    *)
        echo "Unknown environment: $ENV"
        echo "Available environments: development, production, micro, docker"
        exit 1
        ;;
esac

echo "✅ Started successfully!"
echo "🌐 Access your app at: http://localhost:$PORT" 