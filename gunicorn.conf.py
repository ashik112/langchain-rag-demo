import os
import multiprocessing

# Environment-based configuration
ENV = os.getenv('ENV', 'production')
PORT = os.getenv('PORT', '5000')

# Server socket
bind = f"0.0.0.0:{PORT}"

# Adaptive worker configuration based on environment
if ENV == 'development':
    workers = 1
    worker_connections = 50
    timeout = 60
    max_requests = 100
    reload = True
elif ENV == 'micro':
    # For small servers like e2-micro
    workers = 2
    worker_connections = 100
    timeout = 180
    max_requests = 500
else:
    # Production
    workers = multiprocessing.cpu_count() * 2 + 1
    worker_connections = 1000
    timeout = 120
    max_requests = 1000

# Common settings
worker_class = "sync"
keepalive = 2
preload_app = True
max_requests_jitter = 50

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"

# Process naming
proc_name = "langchain-rag-demo" 