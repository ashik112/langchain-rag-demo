# Google Cloud Engine e2-micro Deployment Guide

Complete guide for deploying the LangChain RAG Demo on Google Cloud Engine e2-micro instances.

## 🏗️ GCE e2-micro Specifications

- **vCPUs**: 1 (burstable)
- **Memory**: 1 GB
- **Storage**: 10 GB persistent disk
- **Network**: 1 Gbps
- **Cost**: ~$5/month (with free tier: 1 instance free per month)

## 🚀 Quick Deployment

### 1. Create GCE Instance

```bash
# Create a new e2-micro instance
gcloud compute instances create langchain-rag-demo \
    --zone=us-central1-a \
    --machine-type=e2-micro \
    --network-tier=PREMIUM \
    --maintenance-policy=MIGRATE \
    --provisioning-model=STANDARD \
    --service-account=your-service-account@your-project.iam.gserviceaccount.com \
    --scopes=https://www.googleapis.com/auth/cloud-platform \
    --tags=http-server,https-server \
    --create-disk=auto-delete=yes,boot=yes,device-name=langchain-rag-demo,image=projects/ubuntu-os-cloud/global/images/ubuntu-2204-jammy-v20240319,mode=rw,size=10,type=projects/your-project/zones/us-central1-a/diskTypes/pd-standard \
    --no-shielded-secure-boot \
    --shielded-vtpm \
    --shielded-integrity-monitoring \
    --labels=environment=production,app=langchain-rag-demo \
    --reservation-affinity=any
```

### 2. Configure Firewall Rules

```bash
# Allow HTTP traffic
gcloud compute firewall-rules create allow-http-langchain \
    --allow tcp:80 \
    --source-ranges 0.0.0.0/0 \
    --target-tags http-server \
    --description "Allow HTTP traffic for LangChain RAG Demo"

# Allow HTTPS traffic
gcloud compute firewall-rules create allow-https-langchain \
    --allow tcp:443 \
    --source-ranges 0.0.0.0/0 \
    --target-tags https-server \
    --description "Allow HTTPS traffic for LangChain RAG Demo"
```

### 3. Deploy Application

```bash
# SSH into your instance
gcloud compute ssh langchain-rag-demo --zone=us-central1-a

# Clone your repository
git clone https://github.com/your-username/langchain-rag-demo.git
cd langchain-rag-demo

# Create environment file
nano .env
# Add your environment variables:
# GOOGLE_API_KEY=your_gemini_api_key_here

# Run the deployment script
./deploy-gce-micro.sh
```

## 📋 Restart Policies

### Docker Compose Restart Policy

The `docker-compose.gce-micro.yml` includes `restart: unless-stopped` which means:

- ✅ **Container crashes**: Automatically restarts
- ✅ **System reboot**: Automatically restarts
- ✅ **Docker daemon restart**: Automatically restarts  
- ❌ **Manual stop**: Won't restart until manually started

### Available Restart Policies

```yaml
restart: "no"              # Never restart
restart: always            # Always restart
restart: on-failure        # Only restart on failure
restart: unless-stopped    # Restart unless manually stopped (RECOMMENDED)
```

### System-Level Auto-Start

To ensure Docker starts on boot:

```bash
# Enable Docker service
sudo systemctl enable docker

# Start Docker service
sudo systemctl start docker

# Auto-start your application on boot
sudo crontab -e
# Add this line:
@reboot cd /home/your-username/langchain-rag-demo && /usr/local/bin/docker-compose -f docker-compose.gce-micro.yml up -d
```

## ⚙️ Configuration Details

### Resource Optimization for e2-micro

```yaml
# docker-compose.gce-micro.yml
deploy:
  resources:
    limits:
      cpus: '0.8'        # 80% of 1 vCPU
      memory: 800M       # 80% of 1GB RAM
    reservations:
      cpus: '0.2'        # Minimum 20% CPU
      memory: 200M       # Minimum 200MB RAM
```

### Gunicorn Configuration

```python
# gunicorn.gce-micro.conf.py
workers = 2                    # Conservative for single vCPU
worker_connections = 100       # Reduced for limited resources
timeout = 180                  # Increased for slower processing
max_requests = 500            # Reduced to prevent memory buildup
preload_app = True            # Share memory between workers
```

## 🔧 Management Commands

### Start/Stop Application

```bash
# Start application
docker-compose -f docker-compose.gce-micro.yml up -d

# Stop application
docker-compose -f docker-compose.gce-micro.yml down

# Restart application
docker-compose -f docker-compose.gce-micro.yml restart

# View logs
docker-compose -f docker-compose.gce-micro.yml logs -f
```

### System Monitoring

```bash
# Check system resources
free -h                    # Memory usage
df -h                     # Disk usage
top                       # CPU usage
docker stats              # Container stats

# Check application health
curl -f http://localhost:5000/
```

## 📊 Performance Monitoring

### Resource Usage Monitoring

```bash
# Monitor memory usage
watch -n 5 'free -h && docker stats --no-stream'

# Monitor disk usage
watch -n 30 'df -h'

# Monitor application logs
docker-compose -f docker-compose.gce-micro.yml logs -f --tail=100
```

### Health Check Configuration

```yaml
healthcheck:
  test: ["CMD-SHELL", "curl -f http://localhost:5000/ || exit 1"]
  interval: 60s      # Check every minute
  timeout: 15s       # Wait 15 seconds for response
  retries: 3         # Retry 3 times before marking unhealthy
  start_period: 60s  # Wait 60 seconds before first check
```

## 🛡️ Security Considerations

### Firewall Configuration

```bash
# Check current firewall rules
gcloud compute firewall-rules list

# More restrictive rule (replace YOUR_IP with your actual IP)
gcloud compute firewall-rules create langchain-restricted \
    --allow tcp:80,tcp:443 \
    --source-ranges YOUR_IP/32 \
    --target-tags http-server
```

### Environment Variables

```bash
# Store sensitive data in Google Secret Manager
gcloud secrets create google-api-key --data-file=- <<< "your_api_key_here"

# Use in your application
gcloud secrets versions access latest --secret="google-api-key"
```

## 🔍 Troubleshooting

### Common Issues

1. **Out of Memory**
   ```bash
   # Check memory usage
   free -h
   # Reduce worker count in gunicorn.gce-micro.conf.py
   workers = 1
   ```

2. **Application Won't Start**
   ```bash
   # Check logs
   docker-compose -f docker-compose.gce-micro.yml logs
   # Check disk space
   df -h
   ```

3. **Slow Performance**
   ```bash
   # Increase timeout
   timeout = 300  # in gunicorn.gce-micro.conf.py
   ```

4. **Port Issues**
   ```bash
   # Check what's using port 80
   sudo netstat -tlnp | grep :80
   # Kill process if needed
   sudo kill <pid>
   ```

### Debug Commands

```bash
# Check container status
docker ps -a

# Check container logs
docker logs <container_id>

# Execute commands in container
docker exec -it <container_id> /bin/bash

# Check resource usage
docker stats

# Check system logs
journalctl -u docker.service
```

## 📈 Scaling Options

### Vertical Scaling (Upgrade Instance)

```bash
# Stop instance
gcloud compute instances stop langchain-rag-demo --zone=us-central1-a

# Resize to e2-small (2 vCPUs, 2GB RAM)
gcloud compute instances set-machine-type langchain-rag-demo \
    --machine-type=e2-small \
    --zone=us-central1-a

# Start instance
gcloud compute instances start langchain-rag-demo --zone=us-central1-a
```

### Horizontal Scaling (Load Balancer)

For high-traffic scenarios, consider:
- Multiple e2-micro instances
- Google Cloud Load Balancer
- Cloud Run (serverless alternative)

## 💰 Cost Optimization

### Free Tier Benefits

- 1 e2-micro instance per month: **FREE**
- 30 GB-months standard persistent disk: **FREE**
- 5 GB-months snapshot storage: **FREE**

### Cost Monitoring

```bash
# Check current usage
gcloud compute instances list
gcloud compute disks list

# Set billing alerts in Google Cloud Console
# Navigate to: Billing → Budgets & alerts
```

## 🔄 Backup and Recovery

### Backup Strategy

```bash
# Create snapshot of persistent disk
gcloud compute disks snapshot langchain-rag-demo \
    --snapshot-names=langchain-rag-demo-backup-$(date +%Y%m%d) \
    --zone=us-central1-a

# Backup application data
docker-compose -f docker-compose.gce-micro.yml exec app \
    tar -czf /tmp/backup.tar.gz /app/faiss_index /app/assets
```

### Recovery Process

```bash
# Restore from snapshot
gcloud compute disks create langchain-rag-demo-restored \
    --source-snapshot=langchain-rag-demo-backup-YYYYMMDD \
    --zone=us-central1-a
```

## 📞 Support

For issues or questions:
1. Check the logs: `docker-compose -f docker-compose.gce-micro.yml logs`
2. Review this guide's troubleshooting section
3. Check Google Cloud documentation
4. Monitor system resources and adjust configuration as needed

---

**Note**: e2-micro instances are burstable, meaning they can use more than 1 vCPU for short periods but are limited by credits. Monitor your CPU usage to avoid credit depletion. 