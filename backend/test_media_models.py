import sys
import os
import numpy as np

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), "models"))

from models.text_model import TextDetector
from models.image_model import ImageDetector
from models.video_model import VideoDetector
from models.code_model import CodeDetector
from models.audio_model import AudioDetector

def test_text_engine():
    print("\n--- Testing Text Detection Engine ---")
    detector = TextDetector()
    
    text = (
        "Yesterday, I went to the park and watched the dogs playing fetch. "
        "The weather was lovely and breezy. I hope to go back next weekend with my friends."
    )
    result = detector.predict_detailed(text)
    
    print("Keys returned:", list(result.keys()))
    print("AI Probability:", result["ai_probability"])
    print("Human Probability:", result["human_probability"])
    print("Confidence Score:", result["confidence_score"])
    print("Classification:", result["classification"])
    print("Predicted Source:", result["predicted_source"])
    print("Explanation:", result["explanation"])
    print("Metrics:", result["metrics"])
    
    assert "ai_probability" in result
    assert "human_probability" in result
    assert "confidence_score" in result
    assert "classification" in result
    assert "explanation" in result
    assert abs(result["ai_probability"] + result["human_probability"] - 1.0) < 1e-4
    print("[OK] Text Engine check passed.")

def test_image_engine():
    print("\n--- Testing Image Detection Engine ---")
    detector = ImageDetector()
    
    from PIL import Image
    import io
    img = Image.new("RGB", (256, 256), color="red")
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    image_bytes = img_byte_arr.getvalue()
    
    result = detector.predict_detailed(image_bytes)
    
    print("Keys returned:", list(result.keys()))
    print("AI Probability:", result["ai_probability"])
    print("Human Probability:", result["human_probability"])
    print("Confidence Score:", result["confidence_score"])
    print("Classification:", result["classification"])
    print("Explanation:", result["explanation"])
    
    assert "ai_probability" in result
    assert "human_probability" in result
    assert "confidence_score" in result
    assert "classification" in result
    assert "explanation" in result
    assert abs(result["ai_probability"] + result["human_probability"] - 1.0) < 1e-4
    print("[OK] Image Engine check passed.")

def test_video_engine():
    print("\n--- Testing Video Detection Engine ---")
    detector = VideoDetector()
    result = detector.predict_detailed("")
    
    print("Keys returned:", list(result.keys()))
    print("AI Probability:", result["ai_probability"])
    print("Human Probability:", result["human_probability"])
    print("Confidence Score:", result["confidence_score"])
    print("Classification:", result["classification"])
    print("Explanation:", result["explanation"])
    
    assert "ai_probability" in result
    assert "human_probability" in result
    assert "confidence_score" in result
    assert "classification" in result
    assert "explanation" in result
    assert abs(result["ai_probability"] + result["human_probability"] - 1.0) < 1e-4
    print("[OK] Video Engine check passed.")

def test_code_engine():
    print("\n--- Testing Code Detection Engine ---")
    detector = CodeDetector()
    
    code = """
def getUserById(userId: str) -> Optional[dict]:
    \"\"\"
    Retrieve user information from the database using the unique identifier.
    \"\"\"
    if userId is None:
        raise ValueError("User ID cannot be None")
    try:
        user_record = db.query("SELECT * FROM users WHERE id = ?", (userId,))
        return user_record[0]
    except Exception as e:
        logger.error(f"Error: {e}")
        raise
"""
    result = detector.predict_detailed(code)
    
    print("Keys returned:", list(result.keys()))
    print("Language detected:", result["language"])
    print("AI Probability:", result["ai_probability"])
    print("Human Probability:", result["human_probability"])
    print("Confidence Score:", result["confidence_score"])
    print("Classification:", result["classification"])
    print("Predicted Source:", result["predicted_source"])
    print("Explanation:", result["explanation"])
    print("Metrics:", result["metrics"])
    
    assert "ai_probability" in result
    assert "human_probability" in result
    assert "confidence_score" in result
    assert "classification" in result
    assert "explanation" in result
    assert abs(result["ai_probability"] + result["human_probability"] - 1.0) < 1e-4
    print("[OK] Code Engine check passed.")

def test_audio_engine():
    print("\n--- Testing Audio Detection Engine ---")
    detector = AudioDetector()
    
    # Simple wav byte array header
    dummy_wav = b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x40\x1f\x00\x00\x80\x3e\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
    
    result = detector.predict_detailed(dummy_wav, filename="test.wav")
    
    print("Keys returned:", list(result.keys()))
    print("AI Probability:", result["ai_probability"])
    print("Human Probability:", result["human_probability"])
    print("Confidence Score:", result["confidence_score"])
    print("Classification:", result["classification"])
    print("Predicted Source:", result["predicted_source"])
    print("Explanation:", result["explanation"])
    print("Metrics:", result["metrics"])
    
    assert "ai_probability" in result
    assert "human_probability" in result
    assert "confidence_score" in result
    assert "classification" in result
    assert "explanation" in result
    assert abs(result["ai_probability"] + result["human_probability"] - 1.0) < 1e-4
    print("[OK] Audio Engine check passed.")

if __name__ == "__main__":
    try:
        test_text_engine()
        test_image_engine()
        test_video_engine()
        test_code_engine()
        test_audio_engine()
        print("\nAll model engines verified successfully!")
    except Exception as e:
        print(f"\n[FAIL] Test encountered error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
