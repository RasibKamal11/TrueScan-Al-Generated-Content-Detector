
import requests
import json

BASE_URL = "http://127.0.0.1:8000/detect"

def test_detailed_text():
    print("Testing /detect/text with detailed=True...")
    payload = {
        "text": "This is a simple test. This should be a separate sentence.",
        "detailed": True
    }
    try:
        res = requests.post(f"{BASE_URL}/text", json=payload)
        res.raise_for_status()
        data = res.json()
        
        if "chunks" in data and len(data["chunks"]) >= 2:
            print("✅ Detailed text analysis passed. Chunks found.")
        else:
            print(f"❌ Detailed text analysis failed. Response: {data}")
    except Exception as e:
        print(f"❌ Detailed text analysis error: {e}")

def test_url_analysis():
    print("\nTesting /detect/url...")
    # Use a stable URL that is unlikely to block bots or be offline (example.com)
    payload = {
        "url": "https://example.com"
    }
    try:
        res = requests.post(f"{BASE_URL}/url", json=payload)
        res.raise_for_status()
        data = res.json()
        
        if "score" in data or "ai_probability" in data:
            print("✅ URL analysis passed.")
        else:
            print(f"❌ URL analysis failed. Response: {data}")
    except Exception as e:
        print(f"❌ URL analysis error: {e}")

if __name__ == "__main__":
    test_detailed_text()
    test_url_analysis()
