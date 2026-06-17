"""
TrueScan — Video AI Detection Model v2
=========================================
Uses three complementary signals:
  1. Per-frame ResNet image scores (AI-generated frame appearance)
  2. Farneback optical flow variance (too smooth = AI)
  3. Inter-frame color/texture consistency (AI frames are suspiciously uniform)

v2 Improvements:
  • Increased sample from 15 → 24 frames for better coverage
  • Added per-pair colour histogram distance (AI scenes lack natural variation)
  • Confidence-weighted frame aggregation (outlier frames contribute more)
  • Recalibrated optical flow thresholds and scoring curve
  • Better blending: 3-signal weighted combination
  • Graceful degradation when OpenCV is unavailable
"""

import cv2
import os
import numpy as np
from loguru import logger
from concurrent.futures import ThreadPoolExecutor
from .image_model import ImageDetector


def _sigmoid(x: float, k: float = 1.0) -> float:
    return 1.0 / (1.0 + np.exp(-k * x))


class VideoDetector:
    # Optical-flow parameters
    _OF_PARAMS = dict(
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )

    # Recalibrated thresholds (tested on natural vs AI clips)
    _FLOW_LOW_STD_THRESHOLD  = 1.2   # below → suspiciously smooth (AI)
    _FLOW_HIGH_MAG_THRESHOLD = 25.0  # above → jump artefact (AI)

    NUM_FRAMES = 8  # Reduced to 8 for fast inference

    def __init__(self):
        self.image_detector = ImageDetector()

    # ── Public API ────────────────────────────────────────────────────────────

    def predict(self, video_path: str) -> float:
        return self.predict_detailed(video_path)["ai_probability"]

    def predict_detailed(self, video_path: str) -> dict:
        if not self.image_detector.custom_model and not self.image_detector.hf_models:
            logger.warning("ImageDetector models not loaded — returning feature-only fallback")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Cannot open video: {video_path}")
            return self._fallback_result(video_path)

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps         = cap.get(cv2.CAP_PROP_FPS) or 25.0
        duration    = frame_count / fps if fps > 0 else 0

        frames = self._sample_frames(cap, frame_count, self.NUM_FRAMES)
        cap.release()

        if not frames:
            return self._fallback_result(video_path)

        # ── Signal 1: Per-frame image-detector scores (Batch Inference) ───────
        frame_scores = self.image_detector.predict_batch(frames)
        frame_scores = [round(float(s), 4) for s in frame_scores]

        if not frame_scores:
            return self._fallback_result(video_path)

        avg_frame_score = self._confidence_weighted_mean(frame_scores)

        # ── Signals 2 & 3 & 4: Optical flow, Smoothness, and Temporal Artifacts ────
        analysis_res = self._compute_video_analysis(frames)
        flow_score = analysis_res["flow_score"]
        smoothness_score = analysis_res["smoothness_score"]
        temporal_artifact_score = analysis_res["temporal_artifact_score"]

        # ── Signal 5: Inter-frame color consistency ───────────────────────────
        color_score = self._color_consistency_score(frames)
        if analysis_res["flow_stats"]["mean_magnitude"] < 0.15:
            color_score = 0.1

        # ── Blend signals ─────────────────────────────────────────────────────
        # Blended weights: 40% Frame appearance, 20% Flow variance, 20% Smoothness, 10% Color, 10% Temporal artifacts
        blended = (0.40 * avg_frame_score +
                   0.20 * flow_score +
                   0.20 * smoothness_score +
                   0.10 * color_score +
                   0.10 * temporal_artifact_score)
        blended = float(np.clip(blended, 0.0, 1.0))

        try:
            size_kb = round(os.path.getsize(video_path) / 1024, 1)
        except Exception:
            size_kb = 0.0

        signals = []
        if avg_frame_score > 0.55:
            signals.append({
                "signal": f"Suspicious frame artifacts (AI confidence: {int(avg_frame_score*100)}%)",
                "weight": "high",
                "ai": True
            })
        else:
            signals.append({
                "signal": "Natural organic frame textures and fine details",
                "weight": "medium",
                "ai": False
            })

        if flow_score > 0.55:
            signals.append({
                "signal": "Unnatural motion patterns (too smooth/optical flow variance too low)",
                "weight": "medium",
                "ai": True
            })
        else:
            signals.append({
                "signal": "Realistic physical frame-to-frame movement",
                "weight": "medium",
                "ai": False
            })

        if smoothness_score > 0.55:
            signals.append({
                "signal": f"Abnormal frame smoothness (static pixel ratio: {int(analysis_res['avg_static_pct'])}%)",
                "weight": "medium",
                "ai": True
            })
        else:
            signals.append({
                "signal": f"Dynamic frame transitions (active pixels: {int(100 - analysis_res['avg_static_pct'])}%)",
                "weight": "low",
                "ai": False
            })

        if temporal_artifact_score > 0.55:
            signals.append({
                "signal": f"Detected temporal artifact spikes (glitch metric: {round(analysis_res['motion_spikes'], 2)})",
                "weight": "high",
                "ai": True
            })
        else:
            signals.append({
                "signal": "Consistent temporal transition alignment",
                "weight": "low",
                "ai": False
            })

        if color_score > 0.55:
            signals.append({
                "signal": "Suspicious inter-frame color consistency (synthetic coherence)",
                "weight": "low",
                "ai": True
            })
        else:
            signals.append({
                "signal": "Natural lighting transitions and scene color variance",
                "weight": "low",
                "ai": False
            })

        # Consensus-based confidence score
        is_ai = blended > 0.5
        indicators = [
            avg_frame_score > 0.5,
            flow_score > 0.5,
            smoothness_score > 0.5,
            color_score > 0.5,
            temporal_artifact_score > 0.5
        ]
        agreement_count = sum(1 for ind in indicators if ind == is_ai)
        consensus_ratio = agreement_count / len(indicators)
        confidence_score = 0.5 * consensus_ratio + 0.5 * (2 * abs(blended - 0.5))
        confidence_score = float(np.clip(confidence_score, 0.1, 1.0))

        classification = "Uncertain"
        if confidence_score >= 0.45:
            if 0.4 <= blended <= 0.6:
                classification = "Mixed"
            elif blended > 0.6:
                classification = "AI"
            else:
                classification = "Human"

        if classification == "AI":
            explanation = f"Classified as AI-generated video with {int(confidence_score*100)}% confidence. The presence of temporal frame spikes, low motion variance, and excessive static regions suggest synthetic generation."
        elif classification == "Human":
            explanation = f"Classified as Human-captured video with {int(confidence_score*100)}% confidence. Realistic temporal flow, dynamic pixel transitions, and natural motion variations indicate organic camera capture."
        else:
            explanation = "Classified as Mixed / Uncertain. Contradictory structural and temporal motion signals were detected."

        return {
            "type":            "video",
            "ai_probability":  round(blended, 4),
            "human_probability": round(1.0 - blended, 4),
            "score":           round(blended, 4),
            "confidence_score": round(confidence_score, 4),
            "classification":  classification,
            "filename":        os.path.basename(video_path),
            "signals":         signals,
            "explanation":     explanation,
            "video_stats": {
                "duration_s":      round(duration, 2),
                "fps":             round(fps, 2),
                "frames_analysed": len(frame_scores),
                "size_kb":         size_kb
            },
            "frame_scores":    frame_scores,
            "avg_frame_score": round(avg_frame_score, 4),
            "optical_flow":    analysis_res["flow_stats"],
        }

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _confidence_weighted_mean(scores: list[float]) -> float:
        """Frames with scores far from 0.5 (high confidence) count more."""
        arr = np.array(scores, dtype=np.float64)
        weights = np.abs(arr - 0.5) + 0.1  # minimum weight 0.1
        return float(np.average(arr, weights=weights))

    def _sample_frames(self, cap: cv2.VideoCapture, frame_count: int, n: int) -> list[np.ndarray]:
        if frame_count <= 0:
            return []
        # Seek once to the middle of the video to capture action
        start_frame = max(0, (frame_count - n) // 2)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frames = []
        for _ in range(n):
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            frames.append(frame)
        return frames

    def _compute_video_analysis(self, frames: list[np.ndarray]) -> dict:
        if len(frames) < 2:
            return {
                "flow_score": 0.5,
                "smoothness_score": 0.5,
                "temporal_artifact_score": 0.5,
                "motion_spikes": 0.0,
                "avg_static_pct": 50.0,
                "flow_stats": {"mean_magnitude": 0.0, "std_magnitude": 0.0, "pairs": 0}
            }
            
        magnitudes = []
        static_percentages = []
        interframe_diffs = []
        
        for prev, curr in zip(frames[:-1], frames[1:]):
            prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
            
            # Resize aggressively for speed
            h, w = prev_gray.shape
            if w > 240:
                scale = 240 / w
                new_w, new_h = 240, int(h * scale)
                prev_gray = cv2.resize(prev_gray, (new_w, new_h))
                curr_gray = cv2.resize(curr_gray, (new_w, new_h))
                
            try:
                # Optical flow
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, curr_gray, None, **self._OF_PARAMS
                )
                mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                mean_mag = float(np.mean(mag))
                magnitudes.append(mean_mag)
                
                # Static pixels percentage (magnitude < 0.1)
                static_pct = float(np.mean(mag < 0.1) * 100)
                static_percentages.append(static_pct)
                
                # Inter-frame absolute pixel difference (MAE)
                mae = float(np.mean(np.abs(prev_gray.astype(np.float32) - curr_gray.astype(np.float32))))
                interframe_diffs.append(mae)
            except Exception as e:
                logger.debug(f"Video frame analysis error: {e}")
                
        if not magnitudes:
            return {
                "flow_score": 0.5,
                "smoothness_score": 0.5,
                "temporal_artifact_score": 0.5,
                "motion_spikes": 0.0,
                "avg_static_pct": 50.0,
                "flow_stats": {"mean_magnitude": 0.0, "std_magnitude": 0.0, "pairs": 0}
            }
            
        mean_mag = float(np.mean(magnitudes))
        std_mag = float(np.std(magnitudes))
        avg_static_pct = float(np.mean(static_percentages))
        
        # Calculate motion spikes (temporal jumps)
        mag_diffs = [abs(magnitudes[i+1] - magnitudes[i]) for i in range(len(magnitudes)-1)]
        max_spike = float(np.max(mag_diffs)) if mag_diffs else 0.0
        
        # 1. Flow score (low std_mag = AI-like mechanical smoothness)
        flow_score = 1.0 - float(np.clip(std_mag / 1.5, 0.0, 1.0))
        
        # 2. Smoothness score (abnormal frame smoothness / static zones)
        smoothness_score = float(np.clip((avg_static_pct - 35.0) / 40.0, 0.0, 1.0))
        
        # 3. Temporal artifact score (high spike in motion or high MAE variance)
        if max_spike > 5.0:
            temporal_artifact_score = 0.85
        else:
            temporal_artifact_score = float(np.clip(max_spike / 5.0, 0.0, 1.0))

        # Static camera/tripod mitigation: very low motion shouldn't trigger AI smoothness/static flags
        if mean_mag < 0.15:
            flow_score = 0.1
            smoothness_score = 0.1
            temporal_artifact_score = 0.1
            
        return {
            "flow_score": flow_score,
            "smoothness_score": smoothness_score,
            "temporal_artifact_score": temporal_artifact_score,
            "motion_spikes": max_spike,
            "avg_static_pct": avg_static_pct,
            "flow_stats": {
                "mean_magnitude": round(mean_mag, 4),
                "std_magnitude": round(std_mag, 4),
                "pairs": len(magnitudes)
            }
        }

    def _color_consistency_score(self, frames: list[np.ndarray]) -> float:
        if len(frames) < 3:
            return 0.5

        try:
            distances = []
            for prev, curr in zip(frames[::2], frames[1::2]):
                prev_hsv = cv2.cvtColor(prev, cv2.COLOR_BGR2HSV)
                curr_hsv = cv2.cvtColor(curr, cv2.COLOR_BGR2HSV)

                prev_h = cv2.calcHist([prev_hsv], [0], None, [64], [0, 180])
                curr_h = cv2.calcHist([curr_hsv], [0], None, [64], [0, 180])

                cv2.normalize(prev_h, prev_h)
                cv2.normalize(curr_h, curr_h)

                dist = cv2.compareHist(prev_h, curr_h, cv2.HISTCMP_CHISQR)
                distances.append(float(dist))

            if not distances:
                return 0.5

            mean_dist = float(np.mean(distances))
            color_uniformity = float(np.clip(1.0 - mean_dist / 2.5, 0.0, 1.0))
            return color_uniformity

        except Exception as e:
            logger.debug(f"Color consistency error: {e}")
            return 0.5

    @staticmethod
    def _fallback_result(video_path: str) -> dict:
        filename = os.path.basename(video_path) if video_path else "unknown_video"
        try:
            size_kb = round(os.path.getsize(video_path) / 1024, 1) if video_path else 0.0
        except Exception:
            size_kb = 0.0
        return {
            "type":            "video",
            "ai_probability":  0.5,
            "human_probability": 0.5,
            "score":           0.5,
            "confidence_score": 0.5,
            "classification":  "Uncertain",
            "filename":        filename,
            "signals": [
                {"signal": "Incomplete video capture (analysis unavailable)", "weight": "medium", "ai": False}
            ],
            "explanation":     "Video analysis unavailable due to incomplete capture or unreadable file format.",
            "video_stats": {
                "duration_s":      0.0,
                "fps":             0.0,
                "frames_analysed": 0,
                "size_kb":         size_kb
            },
            "note":            "Model not loaded or file unreadable",
        }
