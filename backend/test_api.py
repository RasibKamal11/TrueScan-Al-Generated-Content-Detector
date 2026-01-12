import requests
import json

BASE_URL = "http://localhost:8000"

def test_text():
    print("Testing /detect/text...")
    try:
        payload = {"text": "This is a test sentence written by a human."}
        response = requests.post(f"{BASE_URL}/detect/text", json=payload)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text}")
    except Exception as e:
        print(f"FAILED: {e}")

def test_image():
    print("\nTesting /detect/image (Mocking file upload)...")
    # We need a dummy image. Let's create a small one or skip validation if logic allows.
    # The model expects a real image file.
    # Let's skip valid image creation for strictness, but we can send bytes.
    try:
        # Create a simple 1x1 black pixel image in memory
        from PIL import Image
        import io
        img = Image.new('RGB', (100, 100), color = 'red')
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG')
        img_byte_arr.seek(0)
        
        files = {'file': ('test.jpg', img_byte_arr, 'image/jpeg')}
        response = requests.post(f"{BASE_URL}/detect/image", files=files)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text}")
    except Exception as e:
        print(f"FAILED: {e}")

if __name__ == "__main__":
    test_text()
    test_image()
