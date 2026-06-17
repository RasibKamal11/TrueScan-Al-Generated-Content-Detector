"""
TrueScan — Celery Worker Task Registry
========================================
Registers a unified Celery task that handles distributed background execution
when REDIS_URL is configured, falling back gracefully to the local thread pool.
"""
from __future__ import annotations

import os
from celery import Celery
from loguru import logger

REDIS_URL = os.environ.get("REDIS_URL", "")

# Initialize Celery app
celery_app = Celery("truescan")

if REDIS_URL:
    try:
        celery_app.conf.update(
            broker_url=REDIS_URL,
            result_backend=REDIS_URL,
            task_serializer="json",
            accept_content=["json"],
            result_serializer="json",
            timezone="UTC",
            enable_utc=True,
        )
        logger.success("Celery distributed task queue active (Redis)")
    except Exception as e:
        logger.error(f"Failed to initialize Celery with Redis: {e}")
        REDIS_URL = ""

@celery_app.task(name="execute_async_job")
def execute_async_job(job_id: str, job_type: str, payload: str):
    """Unified background task executor for Celery workers."""
    logger.info(f"Celery worker executing job {job_id} of type: {job_type}")
    from jobs import _task_registry, _mark_running, _mark_done, _mark_failed
    
    _mark_running(job_id)
    try:
        fn = _task_registry.get(job_type)
        if fn is None:
            raise ValueError(f"Unknown job type: {job_type}")
        result = fn(payload)
        _mark_done(job_id, result)
        logger.success(f"Celery job {job_id} completed successfully")
        return result
    except Exception as e:
        logger.error(f"Celery job {job_id} failed: {e}")
        _mark_failed(job_id, str(e))
        raise e
