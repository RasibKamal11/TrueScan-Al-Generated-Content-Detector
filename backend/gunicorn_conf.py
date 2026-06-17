"""
TrueScan — Production Gunicorn Configuration
=============================================
Orchestrates multiple ASGI Uvicorn workers to scale FastAPI across CPU cores,
enabling robust concurrent connection handling in production.
"""
from __future__ import annotations

import multiprocessing
import os

# Server Bind Address
port = os.environ.get("PORT", "8000")
bind = os.environ.get("GUNICORN_BIND", f"0.0.0.0:{port}")

# CPU Core-Weighted Worker Scaling
workers = int(os.environ.get("WEB_CONCURRENCY", multiprocessing.cpu_count() * 2 + 1))

# Worker Class for ASGI / FastAPI Compatibility
worker_class = "uvicorn.workers.UvicornWorker"

# Timeout & Keep-Alive Settings for Heavy ML Inference
timeout = int(os.environ.get("GUNICORN_TIMEOUT", "120"))
keepalive = int(os.environ.get("GUNICORN_KEEPALIVE", "5"))

# Logging Channels
errorlog = "-"
accesslog = "-"
loglevel = os.environ.get("LOG_LEVEL", "INFO").lower()
capture_output = True
