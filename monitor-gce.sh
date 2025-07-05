#!/bin/bash
# GCE e2-micro monitoring script

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🔍 GCE e2-micro System Monitor${NC}"
echo "=================================="

# System information
echo -e "${BLUE}📊 System Information${NC}"
echo "Date: $(date)"
echo "Uptime: $(uptime -p)"
echo "Load Average: $(cat /proc/loadavg | cut -d' ' -f1-3)"
echo

# Memory usage
echo -e "${BLUE}💾 Memory Usage${NC}"
free -h | grep -E "Mem|Swap"
MEMORY_USAGE=$(free | grep Mem | awk '{print int($3/$2 * 100.0)}')
if [ $MEMORY_USAGE -gt 80 ]; then
    echo -e "${RED}⚠️  High memory usage: ${MEMORY_USAGE}%${NC}"
elif [ $MEMORY_USAGE -gt 60 ]; then
    echo -e "${YELLOW}⚠️  Moderate memory usage: ${MEMORY_USAGE}%${NC}"
else
    echo -e "${GREEN}✅ Memory usage OK: ${MEMORY_USAGE}%${NC}"
fi
echo

# Disk usage
echo -e "${BLUE}💿 Disk Usage${NC}"
df -h | head -1
df -h | grep -E "/$|/dev"
DISK_USAGE=$(df / | grep / | awk '{print int($5)}' | sed 's/%//')
if [ $DISK_USAGE -gt 80 ]; then
    echo -e "${RED}⚠️  High disk usage: ${DISK_USAGE}%${NC}"
elif [ $DISK_USAGE -gt 60 ]; then
    echo -e "${YELLOW}⚠️  Moderate disk usage: ${DISK_USAGE}%${NC}"
else
    echo -e "${GREEN}✅ Disk usage OK: ${DISK_USAGE}%${NC}"
fi
echo

# CPU usage
echo -e "${BLUE}🖥️  CPU Usage${NC}"
CPU_USAGE=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)
echo "CPU Usage: ${CPU_USAGE}%"
if (( $(echo "$CPU_USAGE > 80" | bc -l) )); then
    echo -e "${RED}⚠️  High CPU usage: ${CPU_USAGE}%${NC}"
elif (( $(echo "$CPU_USAGE > 60" | bc -l) )); then
    echo -e "${YELLOW}⚠️  Moderate CPU usage: ${CPU_USAGE}%${NC}"
else
    echo -e "${GREEN}✅ CPU usage OK: ${CPU_USAGE}%${NC}"
fi
echo

# Docker status
echo -e "${BLUE}🐳 Docker Status${NC}"
if systemctl is-active --quiet docker; then
    echo -e "${GREEN}✅ Docker service is running${NC}"
    
    # Container status
    echo -e "${BLUE}📦 Container Status${NC}"
    if docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep -q "langchain-rag-demo"; then
        docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep -E "NAMES|langchain-rag-demo"
        echo -e "${GREEN}✅ Application container is running${NC}"
    else
        echo -e "${RED}❌ Application container is not running${NC}"
    fi
    
    # Container resource usage
    echo -e "${BLUE}📊 Container Resources${NC}"
    docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}" | head -2
    
else
    echo -e "${RED}❌ Docker service is not running${NC}"
fi
echo

# Application health check
echo -e "${BLUE}🏥 Application Health Check${NC}"
if curl -f -s -o /dev/null -w "%{http_code}" http://localhost:5000/ | grep -q "200"; then
    echo -e "${GREEN}✅ Application is responding (HTTP 200)${NC}"
    RESPONSE_TIME=$(curl -o /dev/null -s -w "%{time_total}" http://localhost:5000/)
    echo "Response time: ${RESPONSE_TIME}s"
    if (( $(echo "$RESPONSE_TIME > 5" | bc -l) )); then
        echo -e "${YELLOW}⚠️  Slow response time${NC}"
    fi
else
    echo -e "${RED}❌ Application is not responding${NC}"
fi
echo

# Network connectivity
echo -e "${BLUE}🌐 Network Connectivity${NC}"
if ping -c 1 google.com > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Internet connectivity OK${NC}"
else
    echo -e "${RED}❌ No internet connectivity${NC}"
fi

# Check for common issues
echo -e "${BLUE}🔍 Common Issues Check${NC}"

# Check if running out of space
if [ $DISK_USAGE -gt 90 ]; then
    echo -e "${RED}⚠️  Critical: Disk space very low${NC}"
    echo "   Run: docker system prune -f"
fi

# Check if swap is being used heavily
SWAP_USAGE=$(free | grep Swap | awk '{if ($2 > 0) print int($3/$2 * 100.0); else print 0}')
if [ $SWAP_USAGE -gt 50 ]; then
    echo -e "${YELLOW}⚠️  High swap usage: ${SWAP_USAGE}%${NC}"
    echo "   Consider reducing worker count"
fi

# Check log file sizes
LOG_SIZE=$(du -sh /var/lib/docker/containers/*/*-json.log 2>/dev/null | awk '{sum += $1} END {print sum}' || echo "0")
if [ "$LOG_SIZE" -gt 100 ]; then
    echo -e "${YELLOW}⚠️  Large log files detected${NC}"
    echo "   Run: docker-compose -f docker-compose.gce-micro.yml logs --tail=0"
fi

echo
echo -e "${BLUE}🚀 Quick Actions${NC}"
echo "View logs: docker-compose -f docker-compose.gce-micro.yml logs -f"
echo "Restart app: docker-compose -f docker-compose.gce-micro.yml restart"
echo "Stop app: docker-compose -f docker-compose.gce-micro.yml down"
echo "Clean Docker: docker system prune -f"
echo "Monitor resources: watch -n 5 'free -h && docker stats --no-stream'"
echo
echo -e "${GREEN}✅ Monitoring complete${NC}" 