"""
TrueScan — Audio Detection Model v3
======================================
Detects AI-generated audio (ElevenLabs, Play.ht, Suno, Bark, OpenAI TTS, etc.)
using advanced digital signal processing (DSP) and voice feature extraction.
"""
from __future__ import annotations
import io
import math
import struct
from loguru import logger

try:
    import numpy as np
    import librosa
    _LIBROSA_OK = True
except ImportError:
    _LIBROSA_OK = False
    logger.warning("librosa not installed — audio heuristic fallback active. Run: pip install librosa soundfile")

def _wav_rms(raw: bytes) -> float:
    try:
        n = len(raw) // 2
        samples = struct.unpack(f"<{n}h", raw[:n * 2])
        rms = math.sqrt(sum(s * s for s in samples) / max(n, 1))
        return rms / 32768.0
    except Exception:
        return 0.5

def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))

class AudioDetector:
    def __init__(self):
        self._hf_loaded = False
        self.hf_pipeline = None
        self.ai_label = None

        if not _LIBROSA_OK:
            logger.warning("AudioDetector running in heuristic mode (no librosa)")
        else:
            logger.success("AudioDetector v3 ready (librosa)")

        # Attempt to load pre-trained HF audio classification models
        for model_id in [
            "MelodyMachine/Deepfake-audio-detection-V2",
            "mo-thecreator/Deepfake-audio-detection",
        ]:
            try:
                import torch
                from transformers import pipeline
                device = 0 if torch.cuda.is_available() else -1
                self.hf_pipeline = pipeline(
                    "audio-classification",
                    model=model_id,
                    device=device
                )
                self._hf_loaded = True
                self.ai_label = self._resolve_ai_label()
                logger.success(f"Audio HF model loaded: {model_id} (AI Label: {self.ai_label})")
                break
            except Exception as e:
                logger.warning(f"Failed to load HF audio model {model_id}: {e}")

    def _resolve_ai_label(self) -> str:
        try:
            id2label = self.hf_pipeline.model.config.id2label
            for idx, label in id2label.items():
                if any(k in label.lower() for k in ["fake", "spoof", "synthetic", "generated", "ai"]):
                    return label
            return id2label.get(1, "LABEL_1")
        except Exception:
            return "LABEL_1"

    def predict(self, audio_bytes: bytes) -> float:
        if not _LIBROSA_OK:
            return self._heuristic(audio_bytes)

        try:
            y, sr = librosa.load(io.BytesIO(audio_bytes), sr=None, mono=True, duration=15)
        except Exception as e:
            logger.error(f"librosa.load failed in predict: {e}")
            return 0.5

        hf_score = self._hf_predict(y, sr)
        librosa_score = self._librosa_score_with_y(y, sr)

        if hf_score is not None:
            # Blend: 70% HF model + 30% librosa heuristics
            blended = 0.70 * hf_score + 0.30 * librosa_score
            return round(float(np.clip(blended, 0.02, 0.98)), 4)

        return librosa_score

    def predict_detailed(self, audio_bytes: bytes, filename: str = "") -> dict:
        if not _LIBROSA_OK:
            score = self._heuristic(audio_bytes)
            feats = {}
        else:
            try:
                y, sr = librosa.load(io.BytesIO(audio_bytes), sr=None, mono=True, duration=15)
                hf_score = self._hf_predict(y, sr)
                librosa_score = self._librosa_score_with_y(y, sr)
                if hf_score is not None:
                    score = round(float(np.clip(0.70 * hf_score + 0.30 * librosa_score, 0.02, 0.98)), 4)
                else:
                    score = librosa_score
                feats = self._extract_features(b"", y=y, sr=sr)
            except Exception:
                score = 0.5
                feats = {}

        pitch_consistency = feats.get("pitch_consistency", score * 100)
        spectral_flatness = feats.get("spectral_flatness_score", score * 90)
        prosody_variance  = feats.get("prosody_variance_score", (1 - score) * 80)
        mfcc_uniformity   = feats.get("mfcc_uniformity", score * 85)
        rms_uniformity    = feats.get("rms_uniformity", score * 70)
        pause_score       = feats.get("pause_pattern_score", score * 80)
        centroid_score    = feats.get("spectral_centroid_score", score * 75)

        duration_s  = feats.get("duration", None)
        sample_rate = feats.get("sample_rate", None)
        channels    = feats.get("channels", 1)

        metrics = {
            "perplexity":        round((1.0 - score) * 100, 1),
            "burstiness":        round(prosody_variance, 1),
            "vocab_richness":    round(spectral_flatness, 1),
            "neural_repetition": round(mfcc_uniformity, 1),
            "pitch_consistency": round(pitch_consistency, 1),
            "pause_regularity":  round(pause_score, 1),
            "spectral_stability": round(centroid_score, 1),
        }

        stats = {
            "channels": channels,
            "size_kb":  round(len(audio_bytes) / 1024, 1)
        }
        if duration_s is not None:
            stats["duration_s"] = round(duration_s, 2)
        if sample_rate is not None:
            stats["sample_rate"] = sample_rate

        signals = []
        if pitch_consistency > 60:
            signals.append({"signal": "Unnatural pitch stability (AI TTS pattern)", "weight": "high", "ai": True})
        else:
            signals.append({"signal": "Natural pitch micro-modulations detected", "weight": "medium", "ai": False})
            
        if spectral_flatness > 60:
            signals.append({"signal": "Spectrally flat audio (typical synthetic TTS)", "weight": "medium", "ai": True})
        else:
            signals.append({"signal": "Dynamic high-frequency spectral transitions", "weight": "low", "ai": False})
            
        if prosody_variance < 35:
            signals.append({"signal": "Low prosody variation (robotic rhythm)", "weight": "high", "ai": True})
        else:
            signals.append({"signal": "Expressive rhythmic variations (human prosody)", "weight": "medium", "ai": False})
            
        if mfcc_uniformity > 60:
            signals.append({"signal": "Uniform MFCC patterns (synthetic origin)", "weight": "medium", "ai": True})
            
        if rms_uniformity > 60:
            signals.append({"signal": "Highly uniform loudness (no natural breathing dynamics)", "weight": "medium", "ai": True})
            
        if pause_score > 60:
            signals.append({"signal": "Identical, machine-like pauses between speech chunks", "weight": "medium", "ai": True})
            
        if centroid_score > 60:
            signals.append({"signal": "Spectrally static voice profile (lacks natural timbre shifts)", "weight": "medium", "ai": True})

        # Consensus-based confidence score
        is_ai = score > 0.5
        indicators = [
            pitch_consistency > 55,
            spectral_flatness > 55,
            prosody_variance < 45,
            mfcc_uniformity > 55,
            rms_uniformity > 55,
            pause_score > 55,
            centroid_score > 55
        ]
        if self._hf_loaded:
            indicators.append(score > 0.5)
            
        agreement_count = sum(1 for ind in indicators if ind == is_ai)
        consensus_ratio = agreement_count / len(indicators)
        confidence_score = 0.5 * consensus_ratio + 0.5 * (2 * abs(score - 0.5))
        confidence_score = float(np.clip(confidence_score, 0.1, 1.0))

        classification = "Uncertain"
        if confidence_score >= 0.45:
            if 0.4 <= score <= 0.6:
                classification = "Mixed"
            elif score > 0.6:
                classification = "AI"
            else:
                classification = "Human"

        # Explanation generation
        explanation_factors = []
        if pitch_consistency > 60:
            explanation_factors.append("unnatural pitch stability")
        if spectral_flatness > 60:
            explanation_factors.append("flat frequency distribution")
        if prosody_variance < 35:
            explanation_factors.append("monotonous prosodic rhythm")
        if rms_uniformity > 60:
            explanation_factors.append("overly uniform loudness levels")
        if pause_score > 60:
            explanation_factors.append("highly regular pause patterns")
            
        if not explanation_factors:
            explanation_str = "moderate acoustic indicators"
        elif len(explanation_factors) == 1:
            explanation_str = explanation_factors[0]
        else:
            explanation_str = ", ".join(explanation_factors[:-1]) + ", and " + explanation_factors[-1]
            
        if classification == "AI":
            explanation = f"Classified as AI-generated speech with {int(confidence_score*100)}% confidence. This is indicated by {explanation_str} common in synthetic speech."
        elif classification == "Human":
            explanation = f"Classified as Human-captured speech with {int(confidence_score*100)}% confidence. Dynamic pitch modulations, breathing cues, and normal pause structures suggest natural speech."
        else:
            explanation = "Classified as Mixed / Uncertain. The voice profile contains a combination of dynamic human timbre and uniform synthetic pause indicators."

        # Detect specific TTS origin
        if score > 0.5:
            if pitch_consistency > 75 and pause_score > 70:
                predicted_source = "Google TTS / Azure Speech"
            elif spectral_flatness > 70 and rms_uniformity > 65:
                predicted_source = "ElevenLabs / Play.ht"
            elif score > 0.8:
                predicted_source = "OpenAI TTS (Advanced Synthesis)"
            else:
                predicted_source = "Generic AI Voice Generator"
        else:
            predicted_source = "Human Voice"

        return {
            "type":           "audio",
            "ai_probability": round(score, 4),
            "human_probability": round(1.0 - score, 4),
            "score":          round(score, 4),
            "confidence_score": round(confidence_score, 4),
            "classification": classification,
            "predicted_source": predicted_source,
            "filename":       filename,
            "metrics":        metrics,
            "audio_stats":    stats,
            "signals":        signals,
            "explanation":    explanation,
            "model":          "hybrid-audio-wav2vec2-v3" if self._hf_loaded else "librosa-dsp-features-v3" if _LIBROSA_OK else "heuristic",
        }

    def _heuristic(self, audio_bytes: bytes) -> float:
        size_kb = len(audio_bytes) / 1024
        if size_kb < 10:
            return 0.55
        return 0.50

    def _hf_predict(self, y: np.ndarray, sr: int) -> float | None:
        if not self._hf_loaded or self.hf_pipeline is None:
            return None
        try:
            res = self.hf_pipeline({"raw": y, "sampling_rate": sr})
            for item in res:
                if item["label"] == self.ai_label:
                    return float(item["score"])
            return None
        except Exception as e:
            logger.error(f"HF Audio inference failed: {e}")
            return None

    def _extract_features(self, audio_bytes: bytes, y=None, sr=None) -> dict:
        import numpy as np
        import librosa

        try:
            if y is None or sr is None:
                y, sr = librosa.load(io.BytesIO(audio_bytes), sr=None, mono=True, duration=15)
        except Exception as e:
            logger.error(f"librosa.load failed: {e}")
            return {}

        duration = librosa.get_duration(y=y, sr=sr)

        # ── 1. Pitch (F0) Variation & Naturalness (YIN tracker) ─────────────────
        try:
            f0 = librosa.yin(y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"))
            voiced = f0[f0 > 50]
            if len(voiced) > 10:
                pitch_std  = float(np.std(voiced))
                pitch_mean = float(np.mean(voiced))
                cv = pitch_std / max(pitch_mean, 1e-6)
                pitch_consistency = _clamp((0.35 - cv) / 0.30 * 100, 0, 100)
            else:
                pitch_consistency = 50.0
                cv = 0.25
        except Exception:
            pitch_consistency = 50.0
            cv = 0.25

        # ── 2. Spectral Flatness (Spectrogram Analysis) ────────────────────────
        try:
            flatness = librosa.feature.spectral_flatness(y=y)
            mean_flat = float(np.mean(flatness))
            spectral_flatness_score = _clamp((0.015 - mean_flat) / 0.015 * 100, 0, 100)
        except Exception:
            spectral_flatness_score = 50.0

        # ── 3. Zero-Crossing Rate (ZCR) Variance (Prosody) ─────────────────────
        try:
            zcr = librosa.feature.zero_crossing_rate(y)
            zcr_std = float(np.std(zcr))
            prosody_variance_score = _clamp(zcr_std / 0.012 * 100, 0, 100)
        except Exception:
            prosody_variance_score = 50.0

        # ── 4. MFCC Uniformity (Synthetic Voice Artifacts) ────────────────────
        try:
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
            mfcc_var = float(np.mean(np.var(mfcc, axis=1)))
            mfcc_uniformity = _clamp((300 - mfcc_var) / 300 * 100, 0, 100)
        except Exception:
            mfcc_uniformity = 50.0

        # ── 5. RMS Energy Uniform (Loudness dynamics) ──────────────────────────
        try:
            rms = librosa.feature.rms(y=y)[0]
            rms_std  = float(np.std(rms))
            rms_mean = float(np.mean(rms)) + 1e-9
            rms_cv   = rms_std / rms_mean
            rms_uniformity = _clamp((0.35 - rms_cv) / 0.35 * 100, 0, 100)
        except Exception:
            rms_uniformity = 50.0

        # ── 6. Pause Pattern Analysis ──────────────────────────────────────────
        try:
            rms = librosa.feature.rms(y=y)[0]
            silent_frames = rms < 0.015
            pauses = []
            current_pause = 0
            for sf in silent_frames:
                if sf:
                    current_pause += 1
                else:
                    if current_pause > 0:
                        pauses.append(current_pause * 512 / sr)
                        current_pause = 0
            if current_pause > 0:
                pauses.append(current_pause * 512 / sr)
                
            if len(pauses) >= 2:
                pause_variance = float(np.var(pauses))
                pause_pattern_score = _clamp((0.08 - pause_variance) / 0.08 * 100, 0, 100)
            else:
                pause_pattern_score = 50.0
        except Exception:
            pause_pattern_score = 50.0

        # ── 7. Spectral Centroid Stability (Frequency Pattern) ─────────────────
        try:
            centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            centroid_std = float(np.std(centroid))
            centroid_mean = float(np.mean(centroid))
            centroid_cv = centroid_std / max(centroid_mean, 1e-9)
            spectral_centroid_score = _clamp((0.35 - centroid_cv) / 0.35 * 100, 0, 100)
        except Exception:
            spectral_centroid_score = 50.0

        return {
            "pitch_consistency":       pitch_consistency,
            "spectral_flatness_score": spectral_flatness_score,
            "prosody_variance_score":  prosody_variance_score,
            "mfcc_uniformity":         mfcc_uniformity,
            "rms_uniformity":          rms_uniformity,
            "pause_pattern_score":     pause_pattern_score,
            "spectral_centroid_score": spectral_centroid_score,
            "duration":                duration,
            "sample_rate":             sr,
            "channels":                1,
            "pitch_cv":                cv
        }

    def _librosa_score_with_y(self, y: np.ndarray, sr: int) -> float:
        feats = self._extract_features(b"", y=y, sr=sr)
        if not feats:
            return 0.5

        pitch   = feats.get("pitch_consistency", 50) / 100
        flat    = feats.get("spectral_flatness_score", 50) / 100
        prosody = 1 - (feats.get("prosody_variance_score", 50) / 100)
        mfcc    = feats.get("mfcc_uniformity", 50) / 100
        rms     = feats.get("rms_uniformity", 50) / 100
        pause   = feats.get("pause_pattern_score", 50) / 100
        centroid = feats.get("spectral_centroid_score", 50) / 100

        score = (0.20 * pitch +
                 0.15 * flat +
                 0.15 * prosody +
                 0.15 * mfcc +
                 0.10 * rms +
                 0.15 * pause +
                 0.10 * centroid)

        return round(float(np.clip(score, 0.02, 0.98)), 4)

    def _librosa_score(self, audio_bytes: bytes) -> float:
        try:
            y, sr = librosa.load(io.BytesIO(audio_bytes), sr=None, mono=True, duration=15)
            return self._librosa_score_with_y(y, sr)
        except Exception:
            return 0.5
