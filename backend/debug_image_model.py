from models.image_model import ImageDetector
from PIL import Image
import io
import torch
import torch.nn.functional as F

def debug_image():
    print("Initializing ImageDetector...")
    detector = ImageDetector()
    
    # Create a red image
    img = Image.new('RGB', (224, 224), color = 'red')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_bytes = img_byte_arr.getvalue()
    
    print("\n--- Running Prediction on Red Image ---")
    score = detector.predict(img_bytes)
    print(f"Prediction Score: {score}")

    # Inspect internals if possible
    if hasattr(detector, 'model') and detector.model:
        print(f"\nModel Config ID2LABEL: {detector.model.config.id2label}")
        
    print("\n--- Manual Raw Prediction Check ---")
    # Manually run to see logits
    try:
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        inputs = detector.processor(images=image, return_tensors="pt").to(detector.device)
        with torch.no_grad():
            outputs = detector.model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1)
            
        print(f"Logits: {logits}")
        print(f"Probs: {probs}")
        
        # Check logic analysis
        id2label = detector.model.config.id2label
        print("Label Analysis:")
        for idx, prob in enumerate(probs[0]):
            label = id2label.get(idx, str(idx))
            print(f"  {idx}: {label} = {prob.item():.4f}")
            
    except Exception as e:
        print(f"Manual check failed: {e}")

if __name__ == "__main__":
    debug_image()
