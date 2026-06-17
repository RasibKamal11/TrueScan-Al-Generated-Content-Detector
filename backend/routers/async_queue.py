"""
TrueScan — Async Job Queue Router
====================================
POST /detect/async/submit          → submit a job, get job_id
GET  /detect/async/status/{job_id} → poll status + result
GET  /detect/async/jobs            → list recent jobs
DELETE /detect/async/jobs/{job_id} → cancel / delete job record

Replaces Celery + Redis with the SQLite-backed jobs.py queue.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from loguru import logger
import jobs

router = APIRouter(prefix="/detect/async", tags=["async-queue"])


# ── Request models ────────────────────────────────────────────────────────────

class AsyncSubmitRequest(BaseModel):
    type:    str   # "text" | "fake_news" | "code" | "url"
    payload: str   # text / code / URL string


# ── Task implementations (registered at import time) ──────────────────────────

def _run_text(payload: str) -> dict:
    from routers.detect import get_text_detector
    return get_text_detector().predict_detailed(payload)


def _run_fake_news(payload: str) -> dict:
    from routers.detect import get_fake_news_detector
    return get_fake_news_detector().predict_detailed(payload)


def _run_code(payload: str) -> dict:
    from routers.detect import get_code_detector
    return get_code_detector().predict_detailed(payload)


def _run_url(payload: str) -> dict:
    import requests as _req
    from bs4 import BeautifulSoup
    from routers.detect import get_text_detector
    resp = _req.get(payload, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
    soup = BeautifulSoup(resp.text, "html.parser")
    text = " ".join(p.get_text() for p in soup.find_all("p"))
    result = get_text_detector().predict_detailed(text)
    result["source_url"] = payload
    return result


# Register tasks
jobs.register("text",      _run_text)
jobs.register("fake_news", _run_fake_news)
jobs.register("code",      _run_code)
jobs.register("url",       _run_url)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/submit", status_code=202)
def submit_job(req: AsyncSubmitRequest):
    """
    Submit a detection job for background processing.
    Returns a job_id; poll /status/{job_id} to get the result.
    """
    allowed = {"text", "fake_news", "code", "url"}
    if req.type not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported async type '{req.type}'. Allowed: {sorted(allowed)}"
        )
    job_id = jobs.enqueue(req.type, req.payload)
    logger.info(f"Async job submitted: {job_id} type={req.type}")
    return {
        "job_id": job_id,
        "status": "PENDING",
        "poll_url": f"/detect/async/status/{job_id}",
    }


@router.get("/status/{job_id}")
def job_status(job_id: str):
    """Poll job status. Returns result when status == 'DONE'."""
    record = jobs.get_job_status(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return record


@router.get("/jobs")
def list_jobs(limit: int = 20, status: str | None = None):
    """List recent async jobs, optionally filtered by status (PENDING/RUNNING/DONE/FAILED)."""
    return {"jobs": jobs.list_jobs(limit=limit, status=status)}


@router.delete("/jobs/{job_id}", status_code=204)
def delete_job(job_id: str):
    """Remove a job record from the DB (does not cancel a running job)."""
    import sqlite3, os
    db = os.path.join(os.path.dirname(__file__), "..", "scans.db")
    con = sqlite3.connect(db)
    con.execute("DELETE FROM jobs WHERE id=?", (job_id,))
    con.commit()
    con.close()
