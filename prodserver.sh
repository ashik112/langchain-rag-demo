#!/bin/bash
# Production server script using Gunicorn
source .venv/bin/activate

# Set production environment variables
export FLASK_ENV=production
export PYTHONUNBUFFERED=1

# Run with Gunicorn using configuration file
gunicorn --config gunicorn.conf.py wsgi:app 