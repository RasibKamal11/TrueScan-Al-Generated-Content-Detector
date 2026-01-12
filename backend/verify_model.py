from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import torch.nn.functional as F

def check_model():
    model_name = "openai-community/roberta-base-openai-detector"
    print(f"Checking model: {model_name}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    
    # 1. Known Human Text
    human_text = "I went to the store today to buy some milk and eggs for breakfast. It was a sunny day."
    # 2. Known AI Text (ChatGPT style)
    ai_text = "As an AI language model, I maintain a neutral stance and provide information based on my training data."
    
    texts = [("Human", human_text), ("AI", ai_text)]
    
    print("\n--- Calibration Results ---")
    print(f"Model ID2LABEL: {model.config.id2label}")
    
    for label_type, text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1)
            
        print(f"\nType: {label_type}")
        print(f"Probabilities: {probs[0].tolist()}")
        print(f"Label 0 ({model.config.id2label.get(0, '0')}): {probs[0][0]:.4f}")
        print(f"Label 1 ({model.config.id2label.get(1, '1')}): {probs[0][1]:.4f}")
        
        predicted_id = torch.argmax(probs, dim=-1).item()
        print(f"Predicted Label: {model.config.id2label.get(predicted_id, predicted_id)}")

if __name__ == "__main__":
    check_model()
