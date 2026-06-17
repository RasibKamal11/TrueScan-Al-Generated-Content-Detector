from __future__ import annotations
import os
import re
import urllib.parse
from typing import Any
import requests
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from loguru import logger
from models.external_api import get_external_integrator

class FactCheckRequest(BaseModel):
    text: str
    lang: str = "en"
    max_results: int = 5

router = APIRouter(prefix="/factcheck", tags=["fact-check"])

_GOOGLE_KEY     = os.environ.get("GOOGLE_FACTCHECK_API_KEY", "")
_CLAIMBUSTER_KEY = os.environ.get("CLAIMBUSTER_API_KEY", "")

_GOOGLE_URL     = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
_CLAIMBUSTER_URL = "https://idir.uta.edu/factchecker/api/score/text/"

def _duckduckgo_evidence(text: str) -> list[dict]:
    """
    Free live search fallback using DuckDuckGo.
    No API key required. Provides current web context for claims.
    """
    try:
        from duckduckgo_search import DDGS
        results = []
        with DDGS() as ddgs:
            # Extract main keywords for better search
            keywords = " ".join(re.findall(r"\b\w{5,}\b", text))[:100]
            if not keywords: keywords = text[:100]
            
            for r in ddgs.text(keywords, max_results=3):
                results.append({
                    "source": "duckduckgo_live",
                    "claim": r.get("title", ""),
                    "rating": "Web Context / Search Result",
                    "publisher": "DuckDuckGo / Web",
                    "url": r.get("href", ""),
                    "snippet": r.get("body", "")
                })
        return results
    except Exception as e:
        logger.warning(f"DuckDuckGo search error: {e}")
        return []

def _google_serper_evidence(text: str) -> list[dict]:
    """Provides high-quality Google Search evidence via Serper."""
    try:
        integrator = get_external_integrator()
        # Extract keywords
        keywords = " ".join(re.findall(r"\b\w{5,}\b", text))[:100]
        if not keywords: keywords = text[:100]
        
        return integrator.google_search(keywords, num=3)
    except Exception:
        return []

def _google_factcheck(text: str, lang: str, max_results: int) -> list[dict]:
    if not _GOOGLE_KEY: return []
    try:
        resp = requests.get(_GOOGLE_URL, params={"query": text[:500], "languageCode": lang, "pageSize": max_results, "key": _GOOGLE_KEY}, timeout=8)
        resp.raise_for_status()
        claims = resp.json().get("claims", [])
        results = []
        for c in claims:
            review = c.get("claimReview", [{}])[0]
            results.append({
                "source": "google_factcheck", "claim": c.get("text", ""), "claimant": c.get("claimant", ""),
                "rating": review.get("textualRating", "Unknown"), "publisher": review.get("publisher", {}).get("name", ""),
                "url": review.get("url", ""), "date": c.get("claimDate", "")
            })
        return results
    except Exception: return []

def _claimbuster_factcheck(text: str) -> list[dict]:
    if not _CLAIMBUSTER_KEY: return []
    try:
        resp = requests.get(_CLAIMBUSTER_URL, params={"input_claim": text[:300]}, headers={"x-api-key": _CLAIMBUSTER_KEY}, timeout=8)
        resp.raise_for_status()
        data = resp.json()
        score = data.get("results", [{}])[0].get("score", 0) if data.get("results") else 0
        return [{
            "source": "claimbuster", "claim": text[:200], "claimant": "", "rating": f"Check-worthiness: {round(score, 3)}",
            "publisher": "ClaimBuster", "url": "https://idir.uta.edu/claimbuster/", "date": "", "score": round(score, 4)
        }]
    except Exception: return []

_SENSATIONAL_PATTERNS = [
    r"\b(BREAKING|SHOCKING|EXCLUSIVE|EXPOSED|BOMBSHELL|LEAKED|SECRET|COVER-UP)\b",
    r"\b(they don't want you to know|mainstream media won't tell)\b",
    r"\b(studies show|scientists confirm|experts say)\b",
]

def _heuristic_analysis(text: str) -> dict:
    flags = []
    for pat in _SENSATIONAL_PATTERNS:
        matches = re.findall(pat, text, re.IGNORECASE)
        if matches: flags.extend([str(m) for m in matches])
    score = min(1.0, len(flags) / 5)
    return {
        "source": "heuristic", "sensational_flags": list(set(flags)), "sensationalism_score": round(score, 3),
        "assessment": "High sensationalism" if score > 0.5 else "Moderate" if score > 0.2 else "Low"
    }

@router.post("/claims")
def factcheck_claims(req: FactCheckRequest):
    if len(req.text.strip()) < 10: raise HTTPException(status_code=400, detail="Text too short")
    g_res = _google_factcheck(req.text, req.lang, req.max_results)
    c_res = _claimbuster_factcheck(req.text)
    s_res = _google_serper_evidence(req.text)
    if not s_res:
        s_res = _duckduckgo_evidence(req.text)
    h_res = _heuristic_analysis(req.text)

    evidence = "No significant misinformation patterns detected via APIs."
    if g_res: evidence = f"Matches verified fact-check from {g_res[0]['publisher']}: '{g_res[0]['rating']}'."
    elif s_res: evidence = f"Web context found: '{s_res[0]['snippet'][:150]}...' from {s_res[0]['title']}."
    elif c_res and c_res[0].get("score", 0) > 0.6: evidence = "High check-worthiness score, suggests factual claim needing verification."
    elif h_res.get("sensationalism_score", 0) > 0.5: evidence = f"Sensationalist language detected: {', '.join(h_res['sensational_flags'][:2])}."

    return {
        "text_preview": req.text[:200], "matched_claims": g_res + c_res,
        "heuristic_analysis": h_res, "contextual_evidence": evidence,
        "total_matches": len(g_res) + len(c_res)
    }

@router.get("/providers")
def list_providers():
    return {"google": bool(_GOOGLE_KEY), "claimbuster": bool(_CLAIMBUSTER_KEY), "heuristic": True}
