# Production Deployment Guide

This guide explains how to run the LangChain RAG Demo application in production using Gunicorn as the WSGI server.

## Production Setup

### 1. Using Docker (Recommended)

The easiest way to run in production is using Docker:

```bash
# Build and run with docker-compose
docker-compose up --build

# Or build and run with Docker directly
docker build -t langchain-rag-demo .
docker run -p 5000:5000 --env-file .env langchain-rag-demo
```

### 2. Using Local Environment

If you prefer to run locally with Gunicorn:

```bash
# Install dependencies
pip install -r requirements.txt

# Run production server
./prodserver.sh

# Or run Gunicorn directly
gunicorn --config gunicorn.conf.py wsgi:app
```

### 3. Environment Variables

Make sure to set these environment variables in production:

```bash
# Required for Gemini API
GOOGLE_API_KEY=your_gemini_api_key

# Optional: Custom port (defaults to 5000)
PORT=8080

# Optional: Flask environment
FLASK_ENV=production
```

## Production Configuration

### Gunicorn Settings

The production setup uses the following Gunicorn configuration (`gunicorn.conf.py`):

- **Workers**: CPU count * 2 + 1 (auto-scaled based on available CPUs)
- **Timeout**: 120 seconds (suitable for RAG processing)
- **Keepalive**: 2 seconds
- **Max requests**: 1000 per worker (prevents memory leaks)
- **Logging**: Structured logging to stdout/stderr

### Security Considerations

1. **SSL/HTTPS**: Uncomment and configure SSL settings in `gunicorn.conf.py` for HTTPS
2. **Reverse Proxy**: Consider using Nginx or Apache as a reverse proxy
3. **Environment Variables**: Use proper secret management for API keys
4. **Firewall**: Ensure only necessary ports are open

### Performance Tuning

#### Resource Limits

You can adjust resource limits in `docker-compose.yml`:

```yaml
services:
  app:
    # ... other settings
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 2G
        reservations:
          memory: 1G
```

#### Gunicorn Workers

Adjust workers in `gunicorn.conf.py` based on your server specs:

```python
# For CPU-intensive tasks (RAG processing)
workers = multiprocessing.cpu_count() * 2 + 1

# For I/O-intensive tasks
workers = multiprocessing.cpu_count() * 4 + 1
```

## Monitoring

### Health Checks

The Docker setup includes health checks:

```bash
# Check if container is healthy
docker ps

# Manual health check
curl -f http://localhost:5000/
```

### Logs

```bash
# View logs with docker-compose
docker-compose logs -f

# View logs with Docker
docker logs -f <container_id>
```

## Deployment Scripts

- **`devserver.sh`**: Development server with Flask's built-in server
- **`prodserver.sh`**: Production server with Gunicorn
- **`gunicorn.conf.py`**: Gunicorn configuration file
- **`wsgi.py`**: WSGI entry point for the application

## Scaling

### Horizontal Scaling

For high-traffic deployments, consider:

1. **Load Balancer**: Use Nginx, HAProxy, or cloud load balancers
2. **Multiple Instances**: Run multiple containers behind a load balancer
3. **Container Orchestration**: Use Kubernetes or Docker Swarm

### Vertical Scaling

- Increase CPU/memory limits in docker-compose.yml
- Adjust Gunicorn worker count based on available resources
- Optimize FAISS index size and query performance

## Troubleshooting

### Common Issues

1. **Port Already in Use**: Change the PORT environment variable
2. **Out of Memory**: Reduce worker count or increase memory limits
3. **Slow Responses**: Increase timeout values in gunicorn.conf.py
4. **API Key Errors**: Ensure GOOGLE_API_KEY is properly set

### Debug Mode

For debugging production issues:

```bash
# Run with debug logging
gunicorn --config gunicorn.conf.py --log-level debug wsgi:app
```

### Performance Monitoring

Consider adding monitoring tools:

- **APM**: New Relic, Datadog, or Prometheus
- **Logging**: ELK stack or similar
- **Metrics**: Custom metrics for RAG performance 