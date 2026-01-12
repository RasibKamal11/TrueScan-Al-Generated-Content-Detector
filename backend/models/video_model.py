import cv2
import os
import numpy as np
from .image_model import ImageDetector
import tempfile

class VideoDetector:
    def __init__(self):
        self.image_detector = ImageDetector()

    def predict(self, video_path: str):
        # Frame extraction logic
        # We will extract 5 frames evenly spaced and average the AI probability
        if not self.image_detector.model:
            return 0.5

        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if frame_count <= 0:
            return 0.5

        frames_to_check = 5
        step = max(frame_count // frames_to_check, 1)
        
        scores = []
        
        for i in range(0, frame_count, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert BGR (OpenCV) to RGB (PIL/Transformers)
            # And encode to bytes for the ImageDetector (since it takes bytes)
            # Alternatively, refactor ImageDetector to take PIL Image.
            # For now, let's encode to jpg bytes to reuse existing interface.
            
            is_success, buffer = cv2.imencode(".jpg", frame)
            if is_success:
                score = self.image_detector.predict(buffer.tobytes())
                scores.append(score)
            
            if len(scores) >= frames_to_check:
                break
        
        cap.release()
        
        if not scores:
            return 0.5
            
        # Average probability
        avg_score = sum(scores) / len(scores)
        return avg_score
