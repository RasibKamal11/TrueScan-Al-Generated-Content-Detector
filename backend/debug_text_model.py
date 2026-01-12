from models.text_model import TextDetector
import torch
import torch.nn.functional as F

def debug_model():
    print("Initializing TextDetector...")
    detector = TextDetector()
    
    # Test cases
    ai_text = "As an AI language model, I cannot provide personal opinions. However, artificial intelligence is transforming industries by automating repetitive tasks."
    human_text = "I think the movie was okay, but the ending felt a bit rushed. I wouldn't watch it again."
    
    tests = [("AI Text", ai_text), ("Human Text", human_text)]
    
    print("\n--- Running Debug Tests ---")
    for label, text in tests:
        print(f"\nTesting: {label}")
        print(f"Text Snippet: {text[:50]}...")
        
        # Manually run the internal logic to see raw outputs
        inputs = detector.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(detector.device)
        with torch.no_grad():
            outputs = detector.model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1)
            
        print(f"Logits: {logits}")
        print(f"Probabilities: {probs}")
        print(f"Model ID2Label: {detector.model.config.id2label}")
        
        score = detector.predict(text)
        print(f"Final Prediction Score: {score}")

if __name__ == "__main__":
    debug_model()
