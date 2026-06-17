"""
TrueScan AI Content Detector — FastAPI Backend  v3.0
=====================================================
New in v3.0:
  • JWT Authentication (/auth/*)
  • Async Job Queue   (/detect/async/*)
  • Fact-Check API    (/factcheck/*)
  • Plagiarism Check  (/detect/plagiarism)
  • Persistent SQLite Cache (survives restarts)
  • Video Optical Flow (15-frame Farneback analysis)
"""
import sys
import os
from loguru import logger
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ── Logging setup ─────────────────────────────────────────────────────────────
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
logger.remove()
logger.add(sys.stderr, level=LOG_LEVEL, colorize=True,
           format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")
logger.add(os.path.join(LOG_DIR, "truescan_{time:YYYY-MM-DD}.log"),
           level="DEBUG", rotation="1 day", retention="7 days", compression="zip",
           encoding="utf-8")

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="TrueScan AI Content Detector API",
    version="3.0.0",
    description=(
        "Hybrid neural + ensemble AI content detection API. "
        "v3.0 adds JWT auth, async job queue, fact-check, and plagiarism detection."
    ),
)

# ── CORS ─────────────────────────────────────────────────────────────────────
_allowed_origins = os.environ.get(
    "ALLOWED_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:3000"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Rate limiting (slowapi) ───────────────────────────────────────────────────
try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded
    from slowapi.middleware import SlowAPIMiddleware

    limiter = Limiter(key_func=get_remote_address, default_limits=["60/minute"])
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.add_middleware(SlowAPIMiddleware)
    logger.success("Rate limiter active: 60 req/min per IP")
except ImportError:
    logger.warning("slowapi not installed — rate limiting disabled")

# ── Routers ───────────────────────────────────────────────────────────────────
from routers import detect
from routers import auth as auth_router
from routers import async_queue
from routers import factcheck
from routers import plagiarism

app.include_router(detect.router)
app.include_router(auth_router.router)
app.include_router(async_queue.router)
app.include_router(factcheck.router)
app.include_router(plagiarism.router)

logger.success("All routers registered: detect | auth | async_queue | factcheck | plagiarism")

# ── Global error handler ──────────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error on {request.url}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Please try again."},
    )


@app.get("/")
async def root():
    return {
        "message": "TrueScan API v3.0 is running",
        "docs":    "/docs",
        "features": [
            "AI text/image/audio/video/code detection",
            "Bulk file processing",
            "WebSocket streaming",
            "JWT authentication",
            "Async job queue",
            "Fact-check integration",
            "Plagiarism detection",
            "Persistent cache",
            "Video optical flow (15 frames)",
        ],
    }


@app.get("/health")
async def health_check():
    from cache import cache_stats
    return {
        "status":  "healthy",
        "version": "3.0.0",
        "cache":   cache_stats(),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
