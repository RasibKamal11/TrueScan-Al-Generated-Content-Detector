from fastapi import APIRouter, File, UploadFile, HTTPException
from pydantic import BaseModel
from models.text_model import TextDetector
from models.image_model import ImageDetector
from models.video_model import VideoDetector
import shutil
import os
import requests
from bs4 import BeautifulSoup

router = APIRouter(prefix="/detect", tags=["detection"])

# Initialize models (loading happens once ideally, or lazy loaded)
text_detector = TextDetector()
image_detector = ImageDetector()
video_detector = VideoDetector()

class TextRequest(BaseModel):
    text: str
    detailed: bool = False

class URLRequest(BaseModel):
    url: str

@router.post("/text")
def detect_text(request: TextRequest):
    try:
        if request.detailed:
            return text_detector.predict_detailed(request.text)
        
        score = text_detector.predict(request.text)
        return {"type": "text", "ai_probability": score}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/url")
def detect_url(request: URLRequest):
    try:
        # Fetch URL content
        response = requests.get(request.url, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Extract text (simple approach: get all paragraphs)
        paragraphs = [p.get_text() for p in soup.find_all('p')]
        text_content = " ".join(paragraphs)
        
        if not text_content:
            raise HTTPException(status_code=400, detail="No readable text found at URL")
            
        # Run detailed prediction
        return text_detector.predict_detailed(text_content)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process URL: {str(e)}")

@router.post("/image")
def detect_image(file: UploadFile = File(...)):
    try:
        # In a real app, read bytes or save temp file
        contents = file.file.read()
        score = image_detector.predict(contents)
        return {"type": "image", "ai_probability": score}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/video")
def detect_video(file: UploadFile = File(...)):
    try:
        # Save video to temp file for processing
        temp_filename = f"temp_{file.filename}"
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        score = video_detector.predict(temp_filename)
        
        # Cleanup
        os.remove(temp_filename)
        
        return {"type": "video", "ai_probability": score}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
