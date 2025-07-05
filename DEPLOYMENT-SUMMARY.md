# Deployment Configuration Summary

This document summarizes all the deployment configurations available for the LangChain RAG Demo application.

## 📋 Available Configurations

### 1. Development Environment
- **Script**: `devserver.sh`
- **Server**: Flask development server
- **Use case**: Local development and testing
- **Features**: Hot reload, debug mode, single-threaded

### 2. Production Environment (General)
- **Script**: `prodserver.sh`
- **Config**: `gunicorn.conf.py`
- **Docker**: `docker-compose.yml`
- **Server**: Gunicorn with auto-scaling workers
- **Use case**: General production deployment

### 3. GCE e2-micro Optimized
- **Script**: `deploy-gce-micro.sh`
- **Config**: `gunicorn.gce-micro.conf.py`
- **Docker**: `docker-compose.gce-micro.yml`
- **Monitor**: `monitor-gce.sh`
- **Use case**: Google Cloud Engine e2-micro instances

## 🔄 Restart Policies

### Docker Compose Restart Policies

| Policy | Description | Use Case |
|--------|-------------|----------|
| `no` | Never restart | Development/debugging |
| `always` | Always restart | Critical services |
| `on-failure` | Restart only on failure | Fault tolerance |
| `unless-stopped` | Restart unless manually stopped | **RECOMMENDED** |

### Current Configuration
- **Standard**: `restart: unless-stopped`
- **GCE e2-micro**: `restart: unless-stopped`

## 🏗️ Resource Configuration Comparison

### Standard Production (`docker-compose.yml`)
```yaml
deploy:
  resources:
    limits:
      cpus: '0.8'
      memory: 800M
    reservations:
      cpus: '0.2'
      memory: 200M
```

### GCE e2-micro Optimized (`docker-compose.gce-micro.yml`)
```yaml
deploy:
  resources:
    limits:
      cpus: '0.8'          # 80% of 1 vCPU
      memory: 800M         # 80% of 1GB RAM
    reservations:
      cpus: '0.2'          # Minimum 20% CPU
      memory: 200M         # Minimum 200MB RAM
```

## ⚙️ Gunicorn Configuration Comparison

### Standard Production (`gunicorn.conf.py`)
```python
workers = multiprocessing.cpu_count() * 2 + 1  # Auto-scaling
worker_connections = 1000
timeout = 120
max_requests = 1000
```

### GCE e2-micro Optimized (`gunicorn.gce-micro.conf.py`)
```python
workers = 2                    # Fixed for single vCPU
worker_connections = 100       # Reduced for limited resources
timeout = 180                  # Increased for slower processing
max_requests = 500            # Reduced to prevent memory buildup
```

## 🐳 Docker Configuration Files

### File Structure
```
├── Dockerfile                      # Base Docker image
├── docker-compose.yml              # Standard production
├── docker-compose.gce-micro.yml    # GCE e2-micro optimized
├── wsgi.py                         # WSGI entry point
├── gunicorn.conf.py                # Standard Gunicorn config
└── gunicorn.gce-micro.conf.py      # GCE e2-micro Gunicorn config
```

### Port Configuration
- **Standard**: `5000:5000`
- **GCE e2-micro**: `80:5000` (serves on port 80 externally)

## 🚀 Deployment Commands

### Development
```bash
./devserver.sh
```

### Production (Standard)
```bash
# Local
./prodserver.sh

# Docker
docker-compose up --build -d
```

### Production (GCE e2-micro)
```bash
# Automated deployment
./deploy-gce-micro.sh

# Manual Docker
docker-compose -f docker-compose.gce-micro.yml up --build -d
```

## 🔍 Monitoring and Management

### Health Checks
- **Standard**: Every 30 seconds
- **GCE e2-micro**: Every 60 seconds (resource conservation)

### Monitoring Tools
- **General**: Built-in Docker health checks
- **GCE e2-micro**: `./monitor-gce.sh` (comprehensive system monitoring)

### Log Management
```bash
# Standard
docker-compose logs -f

# GCE e2-micro
docker-compose -f docker-compose.gce-micro.yml logs -f
```

## 📊 Performance Characteristics

### Standard Production
- **Concurrency**: High (auto-scaling workers)
- **Memory**: Medium to high usage
- **CPU**: Efficient multi-core utilization
- **Throughput**: High request handling

### GCE e2-micro Optimized
- **Concurrency**: Limited (2 workers)
- **Memory**: Conservative usage (< 800MB)
- **CPU**: Single-core optimized
- **Throughput**: Moderate, optimized for cost

## 🛠️ Troubleshooting Quick Reference

### Common Issues by Environment

#### Standard Production
- **High memory usage**: Reduce worker count
- **Slow responses**: Check CPU/memory limits
- **Connection issues**: Verify port configuration

#### GCE e2-micro
- **Out of memory**: Reduce to 1 worker
- **Slow performance**: Increase timeout values
- **Container crashes**: Check resource limits

### Debug Commands
```bash
# System resources
free -h                    # Memory
df -h                     # Disk space
top                       # CPU usage

# Docker
docker ps                 # Container status
docker stats             # Resource usage
docker logs <container>  # Container logs

# Application
curl -f http://localhost:5000/  # Health check
```

## 📈 Scaling Options

### Vertical Scaling
1. **Increase resources** in docker-compose.yml
2. **Adjust worker count** in gunicorn.conf.py
3. **Optimize timeout values**

### Horizontal Scaling
1. **Multiple containers** with load balancer
2. **Container orchestration** (Kubernetes, Docker Swarm)
3. **Cloud services** (Google Cloud Run, AWS ECS)

## 🔐 Security Considerations

### Environment Variables
- Store sensitive data in `.env` files
- Use secrets management for production
- Never commit API keys to version control

### Network Security
- Configure firewall rules appropriately
- Use HTTPS in production
- Implement rate limiting if needed

## 💰 Cost Optimization

### GCE e2-micro Free Tier
- **1 instance free** per month
- **30 GB storage** included
- **5 GB snapshots** included
- **Egress traffic** limits apply

### Resource Efficiency
- Use `preload_app = True` to share memory
- Implement proper log rotation
- Monitor and optimize worker counts

## 📚 Documentation Files

- **PRODUCTION.md**: General production deployment guide
- **DEPLOY-GCE-MICRO.md**: GCE e2-micro specific guide
- **DEPLOYMENT-SUMMARY.md**: This summary document

## 🎯 Recommended Deployment Path

1. **Development**: Use `devserver.sh` for local testing
2. **Testing**: Use standard `docker-compose.yml` for staging
3. **Production**: 
   - **Small scale**: Use GCE e2-micro configuration
   - **Medium scale**: Use standard production configuration
   - **Large scale**: Implement horizontal scaling

---

**Quick Start Commands:**
```bash
# Development
./devserver.sh

# Production (standard)
docker-compose up --build -d

# Production (GCE e2-micro)
./deploy-gce-micro.sh
``` 