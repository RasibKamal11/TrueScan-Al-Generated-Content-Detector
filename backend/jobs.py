"""
TrueScan — SQLite-Backed Async Job Queue
==========================================
Replaces Celery + Redis with a lightweight, zero-dependency task queue.

Architecture:
  • Jobs table in SQLite (same file as main scans.db)
  • A single daemon WorkerThread picks jobs from the queue FIFO
  • FastAPI endpoints submit jobs and poll status

Status lifecycle:  PENDING → RUNNING → DONE | FAILED

Upgrade path:
  - Replace WorkerThread with a Celery worker
  - Replace SQLite storage with Redis / Postgres
  - The REST API surface stays identical
"""

from __future__ import annotations

import json
import os
import queue
import sqlite3
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict

from loguru import logger


# ── Database & SQLAlchemy Setup ───────────────────────────────────────────────

from database import SessionLocal, JobRecord


# ── Job model ─────────────────────────────────────────────────────────────────

class Job:
    __slots__ = ("id", "type", "payload")

    def __init__(self, job_id: str, job_type: str, payload: Any):
        self.id      = job_id
        self.type    = job_type
        self.payload = payload


# ── Worker Thread ─────────────────────────────────────────────────────────────

from concurrent.futures import ThreadPoolExecutor

class WorkerPool:
    """
    Multi-threaded background worker pool.
    Replaces single-threaded WorkerThread for parallel task execution.
    """

    def __init__(self, task_registry: Dict[str, Callable], max_workers: int = 4):
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="job-worker")
        self._reg = task_registry

    def submit(self, job: Job) -> None:
        self._executor.submit(self._execute, job)

    def _execute(self, job: Job) -> None:
        _mark_running(job.id)
        try:
            fn = self._reg.get(job.type)
            if fn is None:
                raise ValueError(f"Unknown job type: {job.type}")
            result = fn(job.payload)
            _mark_done(job.id, result)
            logger.success(f"Job {job.id} ({job.type}) completed")
        except Exception as e:
            logger.error(f"Job {job.id} failed: {e}")
            _mark_failed(job.id, str(e))


# ── Database CRUD helpers ─────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _mark_running(job_id: str) -> None:
    db = SessionLocal()
    try:
        job = db.query(JobRecord).filter(JobRecord.id == job_id).first()
        if job:
            job.status = "RUNNING"
            job.started_at = _now()
            db.commit()
    finally:
        db.close()


def _mark_done(job_id: str, result: Any) -> None:
    db = SessionLocal()
    try:
        job = db.query(JobRecord).filter(JobRecord.id == job_id).first()
        if job:
            job.status = "DONE"
            job.result = json.dumps(result)
            job.finished_at = _now()
            db.commit()
    finally:
        db.close()


def _mark_failed(job_id: str, error: str) -> None:
    db = SessionLocal()
    try:
        job = db.query(JobRecord).filter(JobRecord.id == job_id).first()
        if job:
            job.status = "FAILED"
            job.error = error
            job.finished_at = _now()
            db.commit()
    finally:
        db.close()


# ── Public API ────────────────────────────────────────────────────────────────

def enqueue(job_type: str, payload: Any) -> str:
    """Insert a job into the DB and enqueue for processing. Returns job_id."""
    job_id = str(uuid.uuid4())
    db = SessionLocal()
    try:
        new_job = JobRecord(
            id=job_id,
            type=job_type,
            status="PENDING",
            payload=json.dumps(payload),
            created_at=_now()
        )
        db.add(new_job)
        db.commit()
    finally:
        db.close()

    # Dynamic Celery vs Local thread pool routing
    redis_url = os.environ.get("REDIS_URL", "")
    if redis_url:
        try:
            from worker import execute_async_job
            execute_async_job.delay(job_id, job_type, payload)
            logger.info(f"Enqueued job {job_id} dynamically via Celery (Redis)")
            return job_id
        except Exception as e:
            logger.warning(f"Failed to submit Celery task: {e}. Falling back to local threads.")

    _worker.submit(Job(job_id, job_type, payload))
    logger.info(f"Enqueued job {job_id} type={job_type} via local threads")
    return job_id


def get_job_status(job_id: str) -> dict | None:
    """Return current job record or None if not found."""
    db = SessionLocal()
    try:
        row = db.query(JobRecord).filter(JobRecord.id == job_id).first()
        if not row:
            return None
        result_val = json.loads(row.result) if row.result else None
        return {
            "id":           row.id,
            "type":         row.type,
            "status":       row.status,
            "result":       result_val,
            "error":        row.error,
            "created_at":   row.created_at,
            "started_at":   row.started_at,
            "finished_at":  row.finished_at,
        }
    finally:
        db.close()


def list_jobs(limit: int = 20, status: str | None = None) -> list[dict]:
    """List recent jobs, optionally filtered by status."""
    db = SessionLocal()
    try:
        q = db.query(JobRecord)
        if status:
            q = q.filter(JobRecord.status == status)
        rows = q.order_by(JobRecord.created_at.desc()).limit(limit).all()
        return [
            {
                "id": r.id, "type": r.type, "status": r.status,
                "created_at": r.created_at, "finished_at": r.finished_at,
            }
            for r in rows
        ]
    finally:
        db.close()


# ── Task registry & worker singleton ─────────────────────────────────────────
#
# Tasks are registered here by routers at import time.
# To add a new async task type: jobs.register("my_type", my_function)

_task_registry: Dict[str, Callable] = {}


def register(job_type: str, fn: Callable) -> None:
    """Register a synchronous task function for async execution."""
    _task_registry[job_type] = fn
    logger.debug(f"Registered async task: {job_type}")


_worker = WorkerPool(_task_registry)
