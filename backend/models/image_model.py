from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch
import torch.nn.functional as F
import io
import os
from torchvision import transforms
import numpy as np

_KNOWN_AI_LABEL_IDX = {
    "Organika/sdxl-detector": 0,
    "umm-maybe/AI-image-detector": 1,
    "haywoodsloan/ai-image-detector-deploy": 1,
    "Nahrawy/AIorNot": 1,
    "ridouaneg/ai-image-detector": 0,
    "carbon225/vit-base-patch16-224-hf-concept": 1,
}

_HF_MODEL_CANDIDATES = [
    "umm-maybe/AI-image-detector",
    "Organika/sdxl-detector",
    "Nahrawy/AIorNot",
]

_TEMPERATURE = 1.5

class ImageDetector:
    def __init__(self):
        print("Loading Image Detection Model v2...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.custom_model = None
        self.hf_models = []
        self.use_custom = False

        custom_path = os.path.join(os.path.dirname(__file__), "image_resnet_finetuned.pth")
        if os.path.exists(custom_path):
            try:
                from torchvision import models
                import torch.nn as nn
                loaded = False
                for arch_name in ["resnet18", "resnet34", "resnet50"]:
                    try:
                        arch_fn = getattr(models, arch_name)
                        m = arch_fn(weights=None)
                        m.fc = nn.Linear(m.fc.in_features, 2)
                        m.load_state_dict(torch.load(custom_path, map_location=self.device))
                        m = m.to(self.device)
                        m.eval()
                        self.custom_model = m
                        self.use_custom = True
                        print(f"  [OK] Custom {arch_name} model loaded successfully")
                        loaded = True
                        break
                    except Exception as e_arch:
                        # Avoid print if it's the encoder issue, log gracefully
                        try:
                            print(f"  [INFO] Tried loading custom model as {arch_name} but failed: {e_arch}")
                        except Exception:
                            pass
                
                if loaded:
                    self.transform = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize([0.4736, 0.4663, 0.4210], [0.2033, 0.2025, 0.2030])
                    ])
                else:
                    print("  [WARN] Custom model could not be loaded with any supported ResNet architecture.")
            except Exception as e:
                print(f"  [WARN] Custom model loading workflow failed: {e}")

        # Always attempt to load Hugging Face models for hybrid blending
        for model_id in _HF_MODEL_CANDIDATES:
            try:
                # Try loading offline first
                try:
                    processor = AutoImageProcessor.from_pretrained(model_id, local_files_only=True)
                    model = AutoModelForImageClassification.from_pretrained(model_id, local_files_only=True).to(self.device)
                except Exception:
                    # Fallback to online
                    processor = AutoImageProcessor.from_pretrained(model_id)
                    model = AutoModelForImageClassification.from_pretrained(model_id).to(self.device)
                
                model.eval()
                ai_idx = self._resolve_ai_idx(model_id, model.config.id2label)
                self.hf_models.append((processor, model, ai_idx, model_id))
                print(f"  [OK] HF Image model loaded: {model_id}")
                if len(self.hf_models) >= 2: break
            except Exception as e:
                try:
                    print(f"  [WARN] {model_id} failed: {e}")
                except Exception:
                    pass

    @staticmethod
    def _resolve_ai_idx(model_id: str, id2label: dict) -> int:
        if model_id in _KNOWN_AI_LABEL_IDX: return _KNOWN_AI_LABEL_IDX[model_id]
        for idx, label in id2label.items():
            if any(k in label.lower() for k in ["ai", "fake", "artificial", "synthetic"]): return idx
        return 1

    def predict(self, image_bytes: bytes) -> float:
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            # Get DL score
            dl_score = self.predict_images([image])[0]
            
            # Get feature-based score
            img_np = np.array(image)
            meta_res, noise_score, texture_score, artifact_score = self._analyze_image_features(image, img_np)
            
            if meta_res["ai_metadata_found"]:
                feature_score = 1.0
            elif meta_res["camera_detected"]:
                feature_score = 0.15 * (noise_score + texture_score + artifact_score) / 3.0
            else:
                feature_score = 0.35 * noise_score + 0.35 * texture_score + 0.30 * artifact_score
                
            feature_score = float(np.clip(feature_score, 0.0, 1.0))
            
            # Blend: if DL is loaded and returns a valid score (not flat 0.5 due to failure)
            has_dl = self.use_custom or len(self.hf_models) > 0
            if has_dl and dl_score != 0.5:
                blended = 0.70 * dl_score + 0.30 * feature_score
            else:
                blended = feature_score
                
            return float(self._calibrate(blended))
        except Exception:
            return 0.5

    def predict_batch(self, frames: list[np.ndarray]) -> list[float]:
        if not frames:
            return []
        
        pil_images = []
        rgb_frames = []
        for f in frames:
            try:
                rgb_f = f[:, :, ::-1]
                pil_img = Image.fromarray(rgb_f)
                pil_images.append(pil_img)
                rgb_frames.append(rgb_f)
            except Exception:
                pil_images.append(None)
                rgb_frames.append(None)

        # Run DL inference in a single batch
        valid_pil_images = [img for img in pil_images if img is not None]
        if valid_pil_images:
            dl_scores_valid = self.predict_images(valid_pil_images)
        else:
            dl_scores_valid = []

        dl_scores = []
        valid_idx = 0
        for img in pil_images:
            if img is not None:
                dl_scores.append(dl_scores_valid[valid_idx])
                valid_idx += 1
            else:
                dl_scores.append(0.5)

        scores = []
        for idx, pil_img in enumerate(pil_images):
            if pil_img is None:
                scores.append(0.5)
                continue
            try:
                rgb_f = rgb_frames[idx]
                dl_score = dl_scores[idx]
                meta_res, noise_score, texture_score, artifact_score = self._analyze_image_features(pil_img, rgb_f)
                
                if meta_res["ai_metadata_found"]:
                    feature_score = 1.0
                elif meta_res["camera_detected"]:
                    feature_score = 0.15 * (noise_score + texture_score + artifact_score) / 3.0
                else:
                    feature_score = 0.35 * noise_score + 0.35 * texture_score + 0.30 * artifact_score
                    
                feature_score = float(np.clip(feature_score, 0.0, 1.0))
                
                has_dl = self.use_custom or len(self.hf_models) > 0
                if has_dl and dl_score != 0.5:
                    blended = 0.70 * dl_score + 0.30 * feature_score
                else:
                    blended = feature_score
                    
                scores.append(float(self._calibrate(blended)))
            except Exception:
                scores.append(0.5)
        return scores

    def _calibrate(self, p: float) -> float:
        import math
        p = max(0.001, min(0.999, p))
        logit = math.log(p / (1 - p))
        return max(0.02, min(0.98, 1 / (1 + math.exp(-logit * 1.6))))

    def predict_images(self, images: list[Image.Image]) -> list[float]:
        if not self.custom_model and not self.hf_models: return [0.5] * len(images)
        
        custom_scores = None
        if self.use_custom and self.custom_model:
            try:
                tensors = [self.transform(img) for img in images]
                batch_tensor = torch.stack(tensors).to(self.device)
                with torch.no_grad():
                    out = self.custom_model(batch_tensor)
                    probs = F.softmax(out / _TEMPERATURE, dim=1)
                custom_scores = [float(s) for s in probs[:, 0].cpu().numpy()]
            except Exception as e:
                print(f"  [WARN] Custom model prediction failed: {e}")

        model_results = []
        for processor, model, ai_idx, name in self.hf_models:
            try:
                inputs = processor(images=images, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    logits = model(**inputs).logits
                    probs = F.softmax(logits / _TEMPERATURE, dim=-1)
                model_results.append(probs[:, ai_idx].cpu().numpy())
            except Exception: pass
        
        # Blend scores
        final_scores = []
        for idx in range(len(images)):
            c_score = custom_scores[idx] if custom_scores is not None else None
            h_score = np.mean([res[idx] for res in model_results]) if model_results else None

            if c_score is not None and h_score is not None:
                # Blend: 40% Custom model + 60% HF models (HF models are generally more robust)
                blended = 0.40 * c_score + 0.60 * h_score
            elif c_score is not None:
                blended = c_score
            elif h_score is not None:
                blended = h_score
            else:
                blended = 0.5
                
            final_scores.append(float(self._calibrate(blended)))
            
        return final_scores

    def _analyze_image_features(self, image: Image.Image, img_np: np.ndarray) -> tuple[dict, float, float, float]:
        import cv2
        # 1. Metadata
        info = image.getexif()
        has_exif = bool(info)
        camera_detected = False
        software_detected = False
        ai_metadata_found = False
        software_name = ""
        camera_make = ""
        
        if info:
            for tag, val in info.items():
                if tag == 271 or tag == 272: # Make / Model
                    if val and isinstance(val, str) and len(val.strip()) > 1:
                        camera_detected = True
                        camera_make = str(val).strip()
                elif tag == 305: # Software
                    if val and isinstance(val, str):
                        software_name = str(val).strip()
                        software_detected = True
                        if any(k in software_name.lower() for k in ["stable diffusion", "midjourney", "dall-e", "firefly", "novelai"]):
                            ai_metadata_found = True
                            
        if hasattr(image, "info") and image.info:
            for k, v in image.info.items():
                if isinstance(k, str) and isinstance(v, str):
                    if any(ak in v.lower() or ak in k.lower() for ak in ["stable diffusion", "midjourney", "dall-e", "novelai", "sdxl"]):
                        ai_metadata_found = True
                        software_name = "Generative AI"
                        
        meta_res = {
            "has_exif": has_exif,
            "camera_detected": camera_detected,
            "camera_make": camera_make,
            "software_detected": software_detected,
            "software_name": software_name,
            "ai_metadata_found": ai_metadata_found
        }
        
        # 2. Noise Pattern Uniformity
        if len(img_np.shape) == 3:
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_np
            
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        noise = np.abs(gray.astype(np.float32) - blurred.astype(np.float32))
        
        h, w = noise.shape
        block_h, block_w = h // 4, w // 4
        block_stds = []
        for i in range(4):
            for j in range(4):
                block = noise[i*block_h:(i+1)*block_h, j*block_w:(j+1)*block_w]
                block_stds.append(np.std(block))
                
        std_of_stds = float(np.std(block_stds))
        # uniform noise = std_of_stds is small -> higher AI score
        noise_score = 1.0 - float(np.clip(std_of_stds / 0.5, 0.0, 1.0))
        
        # 3. Texture Consistency ( Sobel gradient variation )
        sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        mean_mag = np.mean(magnitude)
        std_mag = np.std(magnitude)
        
        if mean_mag == 0:
            texture_score = 0.5
        else:
            cv_gradients = std_mag / mean_mag
            texture_score = 1.0 - float(np.clip(cv_gradients / 1.2, 0.0, 1.0))
            
        # 4. AI Artifact Indicators ( FFT high-frequency peaks )
        gray_resized = cv2.resize(gray, (256, 256))
        dft = np.fft.fft2(gray_resized)
        dft_shift = np.fft.fftshift(dft)
        magnitude_spectrum = np.abs(dft_shift)
        
        cy, cx = 128, 128
        y, x = np.ogrid[:256, :256]
        mask = (x - cx)**2 + (y - cy)**2 > 40**2
        hf_mag = magnitude_spectrum[mask]
        
        if len(hf_mag) == 0:
            artifact_score = 0.5
        else:
            hf_mean = np.mean(hf_mag)
            hf_max = np.max(hf_mag)
            peak_ratio = hf_max / (hf_mean + 1e-9)
            artifact_score = float(np.clip((peak_ratio - 6.0) / 10.0, 0.0, 1.0))
            
        return meta_res, noise_score, texture_score, artifact_score

    def predict_detailed(self, image_bytes: bytes) -> dict:
        score = self.predict(image_bytes)
        
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            img_np = np.array(image)
            meta_res, noise_score, texture_score, artifact_score = self._analyze_image_features(image, img_np)
        except Exception:
            meta_res = {"ai_metadata_found": False, "camera_detected": False, "camera_make": ""}
            noise_score, texture_score, artifact_score = 0.5, 0.5, 0.5
            
        signals = []
        if meta_res["ai_metadata_found"]:
            signals.append({
                "signal": f"Generative AI metadata tag found ({meta_res['software_name']})",
                "weight": "high",
                "ai": True
            })
        elif meta_res["camera_detected"]:
            signals.append({
                "signal": f"Physical camera signature detected ({meta_res['camera_make']})",
                "weight": "high",
                "ai": False
            })
            
        if noise_score > 0.6:
            signals.append({
                "signal": "Suspiciously uniform noise distribution (synthetic pixel coherence)",
                "weight": "medium",
                "ai": True
            })
        else:
            signals.append({
                "signal": "Natural camera sensor noise variation detected",
                "weight": "medium",
                "ai": False
            })
            
        if texture_score > 0.6:
            signals.append({
                "signal": "Unnaturally consistent gradients and smooth texture transitions",
                "weight": "medium",
                "ai": True
            })
        else:
            signals.append({
                "signal": "Realistic organic texture gradients and micro-structures",
                "weight": "medium",
                "ai": False
            })
            
        if artifact_score > 0.6:
            signals.append({
                "signal": "High-frequency checkerboard patterns in Fourier spectrum",
                "weight": "high",
                "ai": True
            })
        else:
            signals.append({
                "signal": "Clean high-frequency decay spectrum",
                "weight": "low",
                "ai": False
            })
            
        # Consensus-based confidence score
        is_ai = score > 0.5
        indicators = [
            noise_score > 0.5,
            texture_score > 0.5,
            artifact_score > 0.5
        ]
        if self.use_custom or len(self.hf_models) > 0:
            indicators.append(score > 0.5)
        if meta_res["ai_metadata_found"]:
            indicators.append(True)
        elif meta_res["camera_detected"]:
            indicators.append(False)
            
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
                
        if classification == "AI":
            explanation = f"Classified as AI-generated with {int(confidence_score*100)}% confidence. Uniform noise patterns and frequency anomalies indicate artificial synthesis."
        elif classification == "Human":
            explanation = f"Classified as Human-captured with {int(confidence_score*100)}% confidence. Natural sensor noise and organic gradients suggest a physical camera source."
        else:
            explanation = f"Classified as Mixed / Uncertain. Image exhibits contradictory noise or texture signals."

        return {
            "type": "image",
            "ai_probability": round(score, 4),
            "human_probability": round(1.0 - score, 4),
            "score": round(score, 4),
            "confidence_score": round(confidence_score, 4),
            "classification": classification,
            "predicted_source": meta_res["software_name"] if meta_res["ai_metadata_found"] else "Stable Diffusion / Midjourney" if score > 0.5 else "Human Camera",
            "heatmap": self._generate_heatmap(image_bytes),
            "signals": signals,
            "explanation": explanation,
            "model": "hybrid-image-v4",
        }

    def _generate_heatmap(self, image_bytes: bytes) -> str:
        import cv2, base64
        try:
            img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            lap = cv2.applyColorMap(np.uint8(np.absolute(cv2.Laplacian(gray, cv2.CV_64F))), cv2.COLORMAP_JET)
            blended = cv2.addWeighted(img, 0.5, lap, 0.5, 0)
            return base64.b64encode(cv2.imencode('.jpg', blended)[1]).decode('utf-8')
        except Exception: return ""
