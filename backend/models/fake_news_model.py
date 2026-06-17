"""
TrueScan – Fake News / Misinformation Detector
================================================
Uses hamzab/roberta-fake-news-classification from HuggingFace.
Falls back to a simple heuristic scorer if the model can't be loaded
(e.g. no internet or the checkpoint is unavailable).
"""

from __future__ import annotations
import re
import math
import logging
from loguru import logger

# ── Optional heavy imports ────────────────────────────────────────────────────
try:
    from transformers import pipeline as hf_pipeline
    _HF_AVAILABLE = True
except ImportError:
    _HF_AVAILABLE = False

# Credible-source phrases that lower the "fake" score
_CREDIBLE_SIGNALS = [
    r"\baccording to\b", r"\bstudies show\b", r"\bresearchers found\b",
    r"\bpeer.reviewed\b", r"\bpublished in\b", r"\bofficial\b",
    r"\bsaid in a statement\b", r"\bdata shows?\b", r"\bstatistics\b",
]

# Misinformation signals that raise the "fake" score
_MISINFORMATION_SIGNALS = [
    r"\bthey don['\u2019]t want you to know\b", r"\bsecret cure\b",
    r"\bgovernment is hiding\b", r"\bshare before deleted\b",
    r"\bwake up\b", r"\bSHEEPLE\b", r"\bdeep state\b",
    r"\b100%\s*natural\b", r"\bmiracl\w+\b", r"\bconspiracy\b",
    r"\bNEW WORLD ORDER\b", r"\bFAKE NEWS MEDIA\b",
    r"\bBIG PHARMA\b", r"\bNOBODY TALKS ABOUT\b",
    r"\bDON'T LET THEM\b", r"\bBANNED VIDEO\b",
]

_credible_re = [re.compile(p, re.IGNORECASE) for p in _CREDIBLE_SIGNALS]
_misinfo_re  = [re.compile(p, re.IGNORECASE) for p in _MISINFORMATION_SIGNALS]


def _heuristic_score(text: str) -> float:
    """Fast rule-based misinformation scorer (0 = reliable, 1 = likely fake)."""
    words = text.split()
    n = max(len(words), 1)

    # Sentence-casing: lots of ALL-CAPS words is a red flag
    caps_ratio = sum(1 for w in words if w.isupper() and len(w) > 3) / n

    credible_hits = sum(1 for p in _credible_re if p.search(text))
    misinfo_hits  = sum(1 for p in _misinfo_re  if p.search(text))

    raw = (misinfo_hits * 0.15 + caps_ratio * 0.5) - (credible_hits * 0.10)
    return max(0.05, min(0.95, raw))


class FakeNewsDetector:
    """
    Wrapper around hamzab/roberta-fake-news-classification.

    Label mapping from that checkpoint:
        LABEL_0 → FAKE  (ai_probability → high)
        LABEL_1 → REAL  (ai_probability → low)
    """

    def __init__(self):
        self._pipe = None
        if _HF_AVAILABLE:
            try:
                logger.info("Downloading/loading hamzab/roberta-fake-news-classification …")
                self._pipe = hf_pipeline(
                    "text-classification",
                    model="hamzab/roberta-fake-news-classification",
                    truncation=True,
                    max_length=512,
                )
                logger.success("Fake-news model loaded (hamzab/roberta-fake-news-classification)")
            except Exception as e:
                logger.warning(f"Could not load fake-news model: {e}. Heuristic fallback active.")
        else:
            logger.warning("transformers not available – heuristic fake-news scorer active.")

    # ── Public API (mirrors TextDetector) ────────────────────────────────────

    def predict(self, text: str) -> float:
        """Return a single float: probability that the text is fake/misleading."""
        if self._pipe is None:
            return _heuristic_score(text)
        try:
            out = self._pipe(text[:2048])[0]
            label = out["label"]   # "FAKE" or "REAL" / "LABEL_0" or "LABEL_1"
            confidence = out["score"]
            # Some checkpoints return LABEL_0/LABEL_1
            if label in ("FAKE", "LABEL_0"):
                return round(confidence, 4)
            else:
                return round(1.0 - confidence, 4)
        except Exception as e:
            logger.error(f"Fake-news model inference error: {e}")
            return _heuristic_score(text)

    def predict_detailed(self, text: str) -> dict:
        """Return full result dict compatible with ExplainabilityPanel."""
        # Split into sentences for per-sentence scoring
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if len(s.strip()) > 20]
        if not sentences:
            sentences = [text]

        overall = self.predict(text)

        sentence_results = []
        for sent in sentences[:40]:  # cap at 40 for performance
            s_score = self.predict(sent)
            sentence_results.append({"text": sent, "ai_probability": s_score, "score": s_score})

        # Compute basic linguistic metrics (same shape as TextDetector)
        words = text.split()
        n_words = max(len(words), 1)
        unique = len(set(w.lower() for w in words))
        vocab_richness = round(unique / n_words, 3)

        # Sentence-length variance (burstiness proxy)
        sent_lengths = [len(s.split()) for s in sentences]
        if len(sent_lengths) > 1:
            mean_len = sum(sent_lengths) / len(sent_lengths)
            variance = sum((l - mean_len) ** 2 for l in sent_lengths) / len(sent_lengths)
            burstiness = round(math.sqrt(variance) / max(mean_len, 1), 3)
        else:
            burstiness = 0.0

        # Credible vs misinfo signal counts
        credible_hits = sum(1 for p in _credible_re if p.search(text))
        misinfo_hits  = sum(1 for p in _misinfo_re  if p.search(text))

        metrics = {
            "perplexity":        round((1 - overall) * 100, 1),
            "burstiness":        burstiness,
            "vocab_richness":    vocab_richness,
            "neural_repetition": round(overall * 0.8, 3),
            "credible_signals":  credible_hits,
            "misinfo_signals":   misinfo_hits,
        }

        return {
            "type":            "fake_news",
            "ai_probability":  overall,
            "score":           overall,
            "sentences":       sentence_results,
            "metrics":         metrics,
            "detection_mode":  "fake_news",
            "model":           "hamzab/roberta-fake-news-classification" if self._pipe else "heuristic",
        }
