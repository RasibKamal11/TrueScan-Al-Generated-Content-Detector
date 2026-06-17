"""
TrueScan — Plagiarism / Source Attribution Router
===================================================
POST /detect/plagiarism   → check text for copied/sourced content

Strategy:
  1. Split text into sentences
  2. For each sentence > 20 chars, search DuckDuckGo Instant Answer API
     and record matching URLs (no API key required)
  3. Compute an originality score = 1 - (matched / total)
  4. Return per-sentence matches and overall attribution report

This provides approximate source attribution without a Turnitin-style crawled
index. It works for clearly copied web content; it will miss paraphrasing.

No API key required — uses DuckDuckGo's free instant-answer API.
"""

from __future__ import annotations

import re
import time
from typing import Optional

import requests
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from loguru import logger
from models.external_api import get_external_integrator

router = APIRouter(prefix="/detect", tags=["plagiarism"])


# ── Request models ────────────────────────────────────────────────────────────

class PlagiarismRequest(BaseModel):
    text:           str
    max_sentences:  int = 10     # how many sentences to check (cost: ~1 req/sentence)
    min_sent_len:   int = 30     # minimum characters for a sentence to be checked
    delay_ms:       int = 300    # ms delay between requests (rate-limit friendly)


# ── Sentence splitting ────────────────────────────────────────────────────────

def _split_sentences(text: str) -> list[str]:
    """Split text into sentences using simple regex."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]


# ── DuckDuckGo search ─────────────────────────────────────────────────────────

_DDGO_URL = "https://api.duckduckgo.com/"
_HEADERS  = {
    "User-Agent": "TrueScan/2.0 (plagiarism check; contact: admin@truescan.ai)",
    "Accept":     "application/json",
}


def _ddgo_search(query: str) -> list[dict]:
    """
    Use DuckDuckGo Instant Answer API to find web references for a query.
    Returns list of {title, url, snippet} dicts.
    """
    try:
        resp = requests.get(
            _DDGO_URL,
            params={"q": f'"{query[:120]}"', "format": "json", "no_html": "1"},
            headers=_HEADERS,
            timeout=6,
        )
        resp.raise_for_status()
        data = resp.json()

        results = []

        # AbstractURL (Wikipedia / rich result)
        if data.get("AbstractURL"):
            results.append({
                "title":   data.get("Heading", ""),
                "url":     data["AbstractURL"],
                "snippet": data.get("AbstractText", "")[:200],
                "source":  "ddgo_abstract",
            })

        # RelatedTopics
        for topic in data.get("RelatedTopics", [])[:3]:
            if isinstance(topic, dict) and topic.get("FirstURL"):
                results.append({
                    "title":   topic.get("Text", "")[:100],
                    "url":     topic["FirstURL"],
                    "snippet": topic.get("Text", "")[:200],
                    "source":  "ddgo_related",
                })

        return results
    except Exception as e:
        logger.debug(f"DDGo search error for query '{query[:40]}': {e}")
        return []


# ── Similarity heuristic ──────────────────────────────────────────────────────

def _compute_sentence_match_score(sentence: str, results: list[dict]) -> float:
    """
    Basic lexical overlap score between sentence and returned snippets.
    Returns 0.0–1.0 where higher = more likely copied.
    """
    if not results:
        return 0.0

    sent_words = set(re.findall(r'\b\w{4,}\b', sentence.lower()))
    if not sent_words:
        return 0.0

    best_overlap = 0.0
    for r in results:
        snippet_words = set(re.findall(r'\b\w{4,}\b', r.get("snippet", "").lower()))
        if snippet_words:
            overlap = len(sent_words & snippet_words) / len(sent_words)
            best_overlap = max(best_overlap, overlap)

    return round(best_overlap, 4)


# ── Main endpoint ─────────────────────────────────────────────────────────────

@router.post("/plagiarism")
def check_plagiarism(req: PlagiarismRequest):
    """
    Check text for potential plagiarism / external sources.
    Returns per-sentence source matches and an overall originality score.
    """
    if len(req.text.strip()) < 50:
        raise HTTPException(status_code=400, detail="Text too short for plagiarism analysis")

    sentences = _split_sentences(req.text)
    # Filter by minimum length and cap
    candidates = [s for s in sentences if len(s) >= req.min_sent_len]
    candidates = candidates[: req.max_sentences]

    if not candidates:
        raise HTTPException(status_code=400, detail="No sufficiently long sentences found")

    sentence_reports: list[dict] = []
    total_match_score = 0.0

    for sent in candidates:
        # Priority: Serper (Google) > DuckDuckGo
        integrator = get_external_integrator()
        results = integrator.google_search(sent)
        if not results:
            results = _ddgo_search(sent)
            
        match_score = _compute_sentence_match_score(sent, results)
        total_match_score += match_score

        sentence_reports.append({
            "sentence":    sent[:200],
            "match_score": match_score,
            "flagged":     match_score > 0.35,
            "sources":     results[:3],
        })

        if req.delay_ms > 0:
            time.sleep(req.delay_ms / 1000)

    avg_match = total_match_score / len(sentence_reports) if sentence_reports else 0.0
    originality = round(1.0 - avg_match, 4)
    flagged_count = sum(1 for r in sentence_reports if r["flagged"])

    risk_level = (
        "HIGH"   if originality < 0.6 else
        "MEDIUM" if originality < 0.8 else
        "LOW"
    )

    return {
        "type":              "plagiarism",
        "originality_score": originality,
        "avg_match_score":   round(avg_match, 4),
        "risk_level":        risk_level,
        "sentences_checked": len(sentence_reports),
        "sentences_flagged": flagged_count,
        "sentence_reports":  sentence_reports,
        "note":              (
            "DuckDuckGo-based approximate source matching. "
            "Verbatim copying of web content is most reliably detected. "
            "Paraphrasing requires a full crawled index (e.g., Turnitin)."
        ),
    }
