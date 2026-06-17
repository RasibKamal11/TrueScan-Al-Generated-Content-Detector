from fastapi import APIRouter, File, UploadFile, HTTPException, Request, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from models.text_model import TextDetector
from models.image_model import ImageDetector
from models.video_model import VideoDetector
import shutil
import os
import re
import json
import csv
import io
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timezone
import time
import asyncio
import threading
import sqlite3
import uuid
from loguru import logger
from concurrent.futures import ThreadPoolExecutor
from cache import get_cached, set_cached, cache_stats
from urllib.parse import urljoin

router = APIRouter(prefix="/detect", tags=["detection"])

# ─── Thread-safe lazy model loading ──────────────────────────────────────────
_text_detector = None
_image_detector = None
_video_detector = None

_text_lock = threading.Lock()
_image_lock = threading.Lock()
_video_lock = threading.Lock()
_audio_lock = threading.Lock()
_code_lock = threading.Lock()

_audio_detector = None
_code_detector = None


def get_text_detector():
    global _text_detector
    if _text_detector is None:
        with _text_lock:
            if _text_detector is None:
                logger.info("Loading TextDetector...")
                _text_detector = TextDetector()
                logger.success("TextDetector loaded")
    return _text_detector


def get_image_detector():
    global _image_detector
    if _image_detector is None:
        with _image_lock:
            if _image_detector is None:
                logger.info("Loading ImageDetector...")
                _image_detector = ImageDetector()
                logger.success("ImageDetector loaded")
    return _image_detector


def get_video_detector():
    global _video_detector
    if _video_detector is None:
        with _video_lock:
            if _video_detector is None:
                logger.info("Loading VideoDetector...")
                _video_detector = VideoDetector()
                logger.success("VideoDetector loaded")
    return _video_detector



def get_audio_detector():
    global _audio_detector
    if _audio_detector is None:
        with _audio_lock:
            if _audio_detector is None:
                from models.audio_model import AudioDetector
                logger.info("Loading AudioDetector...")
                _audio_detector = AudioDetector()
                logger.success("AudioDetector loaded")
    return _audio_detector


def get_code_detector():
    global _code_detector
    if _code_detector is None:
        with _code_lock:
            if _code_detector is None:
                from models.code_model import CodeDetector
                logger.info("Loading CodeDetector...")
                _code_detector = CodeDetector()
                logger.success("CodeDetector loaded")
    return _code_detector


# ─── SQLite persistence ───────────────────────────────────────────────────────
_DB_PATH = os.path.join(os.path.dirname(__file__), "..", "scans.db")

def _init_db():
    con = sqlite3.connect(_DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS scans (
            id        TEXT PRIMARY KEY,
            type      TEXT NOT NULL,
            score     REAL NOT NULL,
            length    INTEGER,
            filename  TEXT,
            created_at TEXT NOT NULL
        )
    """)
    # Progressive migration: add result_payload column if it doesn't exist
    try:
        con.execute("ALTER TABLE scans ADD COLUMN result_payload TEXT")
        con.commit()
    except sqlite3.OperationalError:
        pass  # Column already exists
    con.close()

_init_db()


def _log_scan(scan_type: str, score: float, length: int = None, filename: str = None, result_payload: str = None) -> str:
    scan_id = str(uuid.uuid4())
    try:
        con = sqlite3.connect(_DB_PATH)
        con.execute(
            "INSERT INTO scans (id, type, score, length, filename, created_at, result_payload) VALUES (?,?,?,?,?,?,?)",
            (scan_id, scan_type, round(score, 4), length, filename,
             datetime.now(timezone.utc).isoformat(), result_payload)
        )
        con.commit()
        con.close()
    except Exception as e:
        logger.error(f"DB log failed: {e}")
    return scan_id


# ─── In-memory counter (quick read for StatsBar) ──────────────────────────────
_scan_counter = {"text": 0, "image": 0, "video": 0, "url": 0, "audio": 0, "code": 0, "total": 0}
_server_start_time = time.time()


# ─── Request / Response models ─────────────────────────────────────────────────
class TextRequest(BaseModel):
    text: str
    detailed: bool = False
    deep_scan: bool = False


class URLRequest(BaseModel):
    url: str


class BatchTextRequest(BaseModel):
    texts: list[str]


class CodeRequest(BaseModel):
    code: str
    deep_scan: bool = False


# ─── Endpoints ────────────────────────────────────────────────────────────────

@router.get("/stats")
def get_stats():
    """Return session scan stats plus all-time totals from SQLite."""
    uptime_seconds = int(time.time() - _server_start_time)
    hours, remainder = divmod(uptime_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    # Pull all-time totals from DB
    try:
        con = sqlite3.connect(_DB_PATH)
        row = con.execute("SELECT COUNT(*), AVG(score) FROM scans").fetchone()
        con.close()
        all_time_total = row[0] or 0
        avg_score = round((row[1] or 0) * 100, 1)
    except Exception:
        all_time_total = 0
        avg_score = 0.0

    return {
        "scans": _scan_counter,
        "uptime": f"{hours:02d}:{minutes:02d}:{seconds:02d}",
        "models_loaded": {
            "text": _text_detector is not None,
            "image": _image_detector is not None,
            "video": _video_detector is not None,
        },
        "all_time_total": all_time_total,
        "avg_risk_score": avg_score,
    }


@router.post("/text")
async def detect_text(request: TextRequest):
    try:
        # ─ Cache check ───────────────────────────────────────────────
        cache_key = f"{request.text}_{'deep' if request.deep_scan else 'detailed' if request.detailed else 'simple'}"
        cached = get_cached(cache_key, "text")
        if cached:
            return {**cached, "cache_hit": True}
        
        detector = get_text_detector()
        _scan_counter["text"] += 1
        _scan_counter["total"] += 1
        
        if request.deep_scan:
            result = await detector.predict_deep(request.text)
            scan_id = _log_scan("text_deep", result.get("ai_probability", 0), len(request.text), result_payload=json.dumps(result))
            result["id"] = scan_id
            set_cached(cache_key, "text", result)
            return result
            
        if request.detailed:
            result = detector.predict_detailed(request.text)
            scan_id = _log_scan("text", result.get("ai_probability", 0), len(request.text), result_payload=json.dumps(result))
            result["id"] = scan_id
            set_cached(cache_key, "text", result)
            return result
            
        score = detector.predict(request.text)
        result = {"type": "text", "ai_probability": score, "score": score}
        scan_id = _log_scan("text", score, len(request.text), result_payload=json.dumps(result))
        result["id"] = scan_id
        set_cached(cache_key, "text", result)
        return result
    except Exception as e:
        logger.error(f"Text detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))



def _fuse_url_content(url: str, text_content: str, result: dict, soup: BeautifulSoup) -> dict:
    """Helper to perform Late Modality Fusion by scanning images on the URL page."""
    fused_images = []
    
    try:
        # Find all <img> tags
        img_tags = soup.find_all("img")
        img_urls = []
        for img in img_tags:
            src = img.get("src")
            if src:
                abs_url = urljoin(url, src)
                # Filter out tracking pixels / tiny icons / non-http links
                if abs_url.startswith("http") and not any(k in abs_url.lower() for k in ["tracker", "analytics", "pixel", "icon", "logo", ".svg"]):
                    img_urls.append(abs_url)
        
        # Deduplicate and limit to first 3 images
        img_urls = list(dict.fromkeys(img_urls))[:3]
        
        if img_urls:
            logger.info(f"Modality Fusion: Scanning {len(img_urls)} images from {url}")
            image_detector = get_image_detector()
            image_scores = []
            
            # Download and predict image scores in parallel to prevent sequential network lag
            with ThreadPoolExecutor(max_workers=len(img_urls)) as executor:
                def process_image(img_url):
                    try:
                        # Timeout reduced from 4 to 2 seconds for faster response times
                        img_resp = requests.get(img_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=2)
                        if img_resp.status_code == 200 and len(img_resp.content) > 1024: # > 1KB
                            score = image_detector.predict(img_resp.content)
                            return {
                                "url": img_url,
                                "score": round(score, 4)
                            }
                    except Exception as img_err:
                        logger.warning(f"Failed to scan image {img_url}: {img_err}")
                    return None
                
                parallel_results = list(executor.map(process_image, img_urls))
                
            for res in parallel_results:
                if res is not None:
                    image_scores.append(res["score"])
                    fused_images.append(res)
            
            if image_scores:
                avg_image_score = sum(image_scores) / len(image_scores)
                text_score = result.get("ai_probability", 0.5)
                
                # Blend: 70% Text Score + 30% Image Score
                combined_score = 0.70 * text_score + 0.30 * avg_image_score
                combined_score = round(max(0.0, min(1.0, combined_score)), 4)
                
                logger.success(f"Modality Fusion Completed. Text Score: {text_score:.4f}, Avg Image Score: {avg_image_score:.4f} -> Blended: {combined_score:.4f}")
                
                result["score"] = combined_score
                result["ai_probability"] = combined_score
                # Re-calculate predicted source based on blended score
                if combined_score > 0.6:
                    result["predicted_source"] = "AI Hybrid Content (Text + Images)"
                elif combined_score < 0.4:
                    result["predicted_source"] = "Human"
                else:
                    result["predicted_source"] = "Mixed Content"
    except Exception as e:
        logger.error(f"Error in modality fusion: {e}")
        
    result["fused_images"] = fused_images
    return result


@router.post("/url")
def detect_url(request: URLRequest):
    try:
        _scan_counter["url"] += 1
        _scan_counter["total"] += 1
        response = requests.get(request.url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = [p.get_text() for p in soup.find_all("p")]
        text_content = " ".join(paragraphs)
        if not text_content.strip():
            raise HTTPException(status_code=400, detail="No readable text found at URL")
        
        # ─ Cache check ───────────────────────────────────────────────
        cache_key = f"{request.url}_detailed"
        cached = get_cached(cache_key, "url")
        if cached:
            return {**cached, "cache_hit": True}

        detector = get_text_detector()
        result = detector.predict_detailed(text_content)
        result["source_url"] = request.url
        
        # Run Modality Fusion
        result = _fuse_url_content(request.url, text_content, result, soup)

        scan_id = _log_scan("url", result.get("ai_probability", 0), len(text_content), result_payload=json.dumps(result))
        result["id"] = scan_id
        set_cached(cache_key, "url", result)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"URL detection error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process URL: {str(e)}")


@router.post("/image")
def detect_image(file: UploadFile = File(...)):
    try:
        contents = file.file.read()
        detector = get_image_detector()
        _scan_counter["image"] += 1
        _scan_counter["total"] += 1
        
        if hasattr(detector, "predict_detailed"):
            result = detector.predict_detailed(contents)
        else:
            score = detector.predict(contents)
            result = {
                "type": "image",
                "ai_probability": round(score, 4),
                "score": round(score, 4),
            }
        result["filename"] = file.filename
        
        scan_id = _log_scan("image", result.get("ai_probability", 0), len(contents), file.filename, result_payload=json.dumps(result))
        result["id"] = scan_id
        return result
    except Exception as e:
        logger.error(f"Image detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch")
def detect_batch(request: BatchTextRequest):
    try:
        results = []
        detector = get_text_detector()
        for text in request.texts:
            score = detector.predict(text)
            _log_scan("text", score, len(text))
            results.append({"text": text[:80] + ("..." if len(text) > 80 else ""), "ai_probability": score})
        _scan_counter["text"] += len(request.texts)
        _scan_counter["total"] += len(request.texts)
        return {"results": results, "count": len(results)}
    except Exception as e:
        logger.error(f"Batch detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/video")
def detect_video(file: UploadFile = File(...)):
    import tempfile
    tmp_path = None
    try:
        suffix = os.path.splitext(file.filename or "")[1] or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            shutil.copyfileobj(file.file, tmp)
        detector = get_video_detector()
        _scan_counter["video"] += 1
        _scan_counter["total"] += 1
        
        if hasattr(detector, "predict_detailed"):
            result = detector.predict_detailed(tmp_path)
        else:
            score = detector.predict(tmp_path)
            result = {
                "type": "video",
                "ai_probability": round(score, 4),
                "score": round(score, 4),
            }
        result["filename"] = file.filename
        
        size = os.path.getsize(tmp_path)
        scan_id = _log_scan("video", result.get("ai_probability", 0), size, file.filename, result_payload=json.dumps(result))
        result["id"] = scan_id
        return result
    except Exception as e:
        logger.error(f"Video detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


@router.get("/history")
def get_history(limit: int = 50, offset: int = 0):
    """Return persisted scan history from SQLite."""
    try:
        con = sqlite3.connect(_DB_PATH)
        rows = con.execute(
            "SELECT id, type, score, length, filename, created_at FROM scans ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (limit, offset)
        ).fetchall()
        total = con.execute("SELECT COUNT(*) FROM scans").fetchone()[0]
        con.close()
        return {
            "history": [
                {"id": r[0], "type": r[1], "score": r[2], "length": r[3], "filename": r[4], "created_at": r[5]}
                for r in rows
            ],
            "total": total,
        }
    except Exception as e:
        logger.error(f"History fetch error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/results/{scan_id}")
def get_scan_result(scan_id: str):
    """Retrieve detailed persisted scan result by ID for public sharing."""
    try:
        con = sqlite3.connect(_DB_PATH)
        row = con.execute(
            "SELECT type, score, length, filename, created_at, result_payload FROM scans WHERE id = ?",
            (scan_id,)
        ).fetchone()
        con.close()
        if not row:
            raise HTTPException(status_code=404, detail="Scan result not found")
        
        type_, score, length, filename, created_at, result_payload = row
        
        if result_payload:
            payload = json.loads(result_payload)
        else:
            payload = {
                "type": type_,
                "ai_probability": score,
                "score": score,
                "length": length,
                "filename": filename,
                "created_at": created_at,
            }
        
        payload["id"] = scan_id
        return payload
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching scan result: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ─── Audio Detection ───────────────────────────────────────────────────────────

@router.post("/audio")
def detect_audio(file: UploadFile = File(...)):
    """Detect AI-generated audio (ElevenLabs, Suno, Bark, Play.ht, etc.)"""
    try:
        audio_bytes = file.file.read()
        detector = get_audio_detector()
        _scan_counter["audio"] += 1
        _scan_counter["total"] += 1
        result = detector.predict_detailed(audio_bytes, filename=file.filename or "")
        scan_id = _log_scan("audio", result.get("ai_probability", 0), len(audio_bytes), file.filename, result_payload=json.dumps(result))
        result["id"] = scan_id
        return result
    except Exception as e:
        logger.error(f"Audio detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ─── Code Detection ───────────────────────────────────────────────────────────

@router.post("/code")
async def detect_code(request: CodeRequest):
    """Detect AI-generated code (Copilot, ChatGPT, Claude, Cursor, etc.)"""
    try:
        detector = get_code_detector()
        _scan_counter["code"] += 1
        _scan_counter["total"] += 1
        
        if request.deep_scan:
            # Code detector might not have predict_deep, fallback to detailed if so
            if hasattr(detector, "predict_deep"):
                result = await detector.predict_deep(request.code)
            else:
                result = detector.predict_detailed(request.code)
        else:
            result = detector.predict_detailed(request.code)
            
        scan_id = _log_scan("code", result.get("ai_probability", 0), len(request.code), result_payload=json.dumps(result))
        result["id"] = scan_id
        return result
    except Exception as e:
        logger.error(f"Code detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ─── Bulk File Upload ─────────────────────────────────────────────────────────

@router.post("/bulk-file")
async def detect_bulk_file(file: UploadFile = File(...)):
    """
    Accept a .txt or .csv file with multiple texts and return per-item AI scores.
    TXT: one paragraph/item per blank-line-separated block.
    CSV: first column is treated as the text.
    """
    try:
        content_bytes = await file.read()
        content = content_bytes.decode("utf-8", errors="replace")

        texts: list[str] = []

        if file.filename and file.filename.lower().endswith(".csv"):
            reader = csv.reader(io.StringIO(content))
            for row in reader:
                if row and row[0].strip():
                    texts.append(row[0].strip())
        else:
            # TXT: split on double newlines (paragraph blocks)
            blocks = re.split(r"\n\s*\n", content) if "re" in dir() else content.split("\n\n")
            texts = [b.strip() for b in blocks if len(b.strip()) > 20]

        if not texts:
            raise HTTPException(status_code=400, detail="No usable text found in file")

        detector = get_text_detector()
        results = []
        
        # Parallelize bulk processing for speed
        with ThreadPoolExecutor(max_workers=min(len(texts), 16)) as executor:
            def process_text(txt):
                score = detector.predict(txt)
                _log_scan("text", score, len(txt))
                return {
                    "preview": txt[:100] + ("..." if len(txt) > 100 else ""),
                    "ai_probability": round(score, 4),
                    "score": round(score, 4),
                    "length": len(txt),
                }
            
            # Map items to workers
            results = list(executor.map(process_text, texts[:200]))

        _scan_counter["text"] += len(results)
        _scan_counter["total"] += len(results)

        overall = sum(r["ai_probability"] for r in results) / len(results)
        high_risk = sum(1 for r in results if r["ai_probability"] > 0.6)

        return {
            "type":          "bulk",
            "filename":      file.filename,
            "total_items":   len(results),
            "high_risk":     high_risk,
            "avg_score":     round(overall, 4),
            "ai_probability": round(overall, 4),
            "score":         round(overall, 4),
            "results":       results,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Bulk file detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ─── WebSocket Real-Time Progress ─────────────────────────────────────────────

# Step messages streamed during inference
_WS_STEPS: dict[str, list[str]] = {
    "text": [
        "Tokenizing input text...",
        "Running RoBERTa transformer inference...",
        "Computing perplexity score...",
        "Measuring burstiness coefficient...",
        "Evaluating vocabulary richness...",
        "Aggregating sentence-level signals...",
        "Blending neural + ensemble scores...",
        "Finalizing detection result...",
    ],

    "image": [
        "Preprocessing image tensor...",
        "Applying ResNet feature extraction...",
        "Detecting GAN / diffusion artifacts...",
        "Analyzing pixel coherence maps...",
        "Checking for SDXL / DALL-E signatures...",
        "Running classifier head...",
        "Finalizing detection result...",
    ],
    "audio": [
        "Decoding audio stream...",
        "Extracting pitch (F0) contour...",
        "Analysing spectral flatness...",
        "Computing zero-crossing rate variance...",
        "Running MFCC uniformity analysis...",
        "Detecting prosody patterns...",
        "Blending feature scores...",
        "Finalizing detection result...",
    ],
    "code": [
        "Parsing code structure...",
        "Detecting language...",
        "Analysing comment verbosity...",
        "Checking naming conventions...",
        "Detecting boilerplate patterns...",
        "Scanning for human signals...",
        "Computing cognitive complexity...",
        "Finalizing detection result...",
    ],
    "url": [
        "Fetching URL content...",
        "Parsing HTML structure...",
        "Extracting text content...",
        "Tokenizing extracted text...",
        "Running NLP analysis...",
        "Computing credibility metrics...",
        "Evaluating writing patterns...",
        "Finalizing detection result...",
    ],
    "video": [
        "Extracting video keyframes...",
        "Processing frame sequences...",
        "Detecting temporal inconsistencies...",
        "Analysing motion artifacts...",
        "Running frame-by-frame analysis...",
        "Aggregating temporal scores...",
        "Finalizing detection result...",
    ],
}


@router.websocket("/ws/detect")
async def websocket_detect(websocket: WebSocket):
    """
    WebSocket endpoint for real-time detection progress streaming.

    Client sends JSON: { "type": "text"|"code"|"audio"|..., "payload": "..." }
    Server streams: { "step": int, "total": int, "message": str, "progress": float }
    Then sends final: { "done": true, "result": { ... } }
    """
    await websocket.accept()
    try:
        raw = await websocket.receive_text()
        data = json.loads(raw)
        detect_type = data.get("type", "text")
        payload = data.get("payload", "")
        deep_scan = data.get("deep_scan", False)

        steps = _WS_STEPS.get(detect_type, _WS_STEPS["text"])
        total = len(steps)

        # ─ Cache check ───────────────────────────────────────────────
        if detect_type in ["text", "code", "url"]:
            cache_key = f"{payload}_{'deep' if deep_scan else 'detailed'}"
            cached = get_cached(cache_key, detect_type)
            if cached:
                logger.info(f"WebSocket cache hit for type: {detect_type}")
                await websocket.send_text(json.dumps({
                    "step":     total - 1,
                    "total":    total,
                    "message":  steps[-1],
                    "progress": 100.0,
                    "done":     True,
                    "result":   {**cached, "cache_hit": True},
                }))
                return

        # Helper to run the appropriate inference function asynchronously
        async def perform_inference():
            if detect_type == "text":
                if deep_scan:
                    return await get_text_detector().predict_deep(payload)
                else:
                    return await asyncio.to_thread(get_text_detector().predict_detailed, payload)
            elif detect_type == "code":
                det = get_code_detector()
                if deep_scan and hasattr(det, "predict_deep"):
                    return await det.predict_deep(payload)
                else:
                    return await asyncio.to_thread(det.predict_detailed, payload)
            elif detect_type == "url":
                def run_url_inference():
                    resp = requests.get(payload, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
                    soup = __import__("bs4").BeautifulSoup(resp.text, "html.parser")
                    text = " ".join(p.get_text() for p in soup.find_all("p"))
                    return text
                
                text = await asyncio.to_thread(run_url_inference)
                if deep_scan:
                    result = await get_text_detector().predict_deep(text)
                else:
                    result = await asyncio.to_thread(get_text_detector().predict_detailed, text)
                result["source_url"] = payload
                return result
            else:
                return {"type": detect_type, "ai_probability": 0.5, "note": "Upload binary via REST /detect/" + detect_type}

        # Start inference in background
        inference_task = asyncio.create_task(perform_inference())

        # Stream progress steps concurrently
        step_delay = 0.08  # fast progress animation
        current_step = 0
        while not inference_task.done() and current_step < total - 1:
            await websocket.send_text(json.dumps({
                "step":     current_step,
                "total":    total,
                "message":  steps[current_step],
                "progress": round((current_step + 1) / total * 90, 1),
                "done":     False,
            }))
            current_step += 1
            
            try:
                # Wait for next step delay or until the inference completes
                await asyncio.wait_for(asyncio.shield(inference_task), timeout=step_delay)
            except asyncio.TimeoutError:
                pass
            except Exception:
                break

        # Wait for inference to finish if it hasn't already
        result: dict = {}
        try:
            result = await inference_task
            # Save to cache on success
            if detect_type in ["text", "code", "url"] and result and "error" not in result:
                cache_key = f"{payload}_{'deep' if deep_scan else 'detailed'}"
                set_cached(cache_key, detect_type, result)
        except Exception as e:
            logger.error(f"WS inference error: {e}")
            result = {"error": str(e)}

        # Send final step with result
        await websocket.send_text(json.dumps({
            "step":     total - 1,
            "total":    total,
            "message":  steps[-1],
            "progress": 100.0,
            "done":     True,
            "result":   result,
        }))

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.send_text(json.dumps({"error": str(e), "done": True}))
        except Exception:
            pass

