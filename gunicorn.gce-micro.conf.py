# Gunicorn configuration file optimized for GCE e2-micro instance
# e2-micro: 1 vCPU, 1GB RAM, 10GB persistent disk
import os

# Server socket
bind = f"0.0.0.0:{os.environ.get('PORT', 5000)}"
backlog = 512  # Reduced for limited resources

# Worker processes - optimized for e2-micro (1 vCPU, 1GB RAM)
workers = 2  # Conservative for single vCPU
worker_class = "sync"
worker_connections = 100  # Reduced for limited resources
timeout = 180  # Increased for slower processing
keepalive = 5  # Increased to reduce connection overhead
max_requests = 500  # Reduced to prevent memory buildup
max_requests_jitter = 25

# Memory management
preload_app = True  # Share memory between workers
max_worker_memory = 200  # MB - restart worker if it exceeds this

# Security
limit_request_line = 2048  # Reduced
limit_request_fields = 50  # Reduced
limit_request_field_size = 4096  # Reduced

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"
access_log_format = '%(h)s %(t)s "%(r)s" %(s)s %(b)s %(D)s'  # Simplified format

# Process naming
proc_name = "langchain-rag-demo-gce"

# Server mechanics
daemon = False
pidfile = None
user = None
group = None
tmp_upload_dir = None

# Environment variables
raw_env = [
    "PYTHONUNBUFFERED=1",
    "FLASK_ENV=production",
    "MALLOC_TRIM_THRESHOLD=100000",  # Reduce memory fragmentation
]

# Worker process lifecycle with memory monitoring
def when_ready(server):
    server.log.info("Server is ready for GCE e2-micro. Spawning %d workers", workers)

def worker_int(worker):
    worker.log.info("Worker %s received INT or QUIT signal", worker.pid)

def pre_fork(server, worker):
    server.log.info("Worker spawned (pid: %s)", worker.pid)

def post_fork(server, worker):
    server.log.info("Worker spawned (pid: %s)", worker.pid)

def post_worker_init(worker):
    worker.log.info("Worker initialized (pid: %s)", worker.pid)

def worker_abort(worker):
    worker.log.info("Worker aborted (pid: %s)", worker.pid)

# Custom configuration for e2-micro performance
def on_starting(server):
    server.log.info("Starting Gunicorn on GCE e2-micro instance")
    server.log.info("Configuration: %d workers, %d connections per worker", workers, worker_connections)

def on_reload(server):
    server.log.info("Reloading Gunicorn configuration")

def on_exit(server):
    server.log.info("Shutting down Gunicorn") 