from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import torch.nn.functional as F
import os
import re

class TextDetector:
    def __init__(self):
        print("Loading Text Detection Model...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Priority: Fine-tuned Model -> Fallback: Base Model
        self.fine_tuned_path = "models/text_best"
        
        # Base model (Better for ChatGPT/GPT-4 detection)
        self.base_model_name = "Hello-SimpleAI/chatgpt-detector-roberta"
        
        try:
            if os.path.exists(self.fine_tuned_path):
                print(f"Loading Fine-Tuned Text Model from {self.fine_tuned_path}...")
                self.tokenizer = AutoTokenizer.from_pretrained(self.fine_tuned_path)
                self.model = AutoModelForSequenceClassification.from_pretrained(self.fine_tuned_path).to(self.device)
                print("Fine-Tuned Text Model Loaded Successfully.")
            else:
                print(f"Fine-tuned model not found. Loading Base Model {self.base_model_name}...")
                self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(self.base_model_name).to(self.device)
                print(f"Base Text Model {self.base_model_name} Loaded Successfully.")
                
            self.model.eval()
            print(f"Model Labels: {self.model.config.id2label}")
        except Exception as e:
            print(f"Error loading text model: {e}")
            self.model = None

    def predict(self, text: str):
        if not self.model:
            return 0.5 

        # Tokenize
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1)
            
        # Logic for Hello-SimpleAI/chatgpt-detector-roberta
        # Labels: {0: 'Human', 1: 'ChatGPT'}
        # Index 1 is AI.
        
        id2label = self.model.config.id2label
        ai_probability = 0.0
        
        if "chatgpt-detector-roberta" in self.base_model_name:
             ai_probability = probs[0][1].item()
        
        # Fallback dynamic logic (Generic)
        else:
            fake_label_id = None
            for idx, label in id2label.items():
                l = label.lower()
                if "fake" in l or "generated" in l or "ai" in l or "chatgpt" in l or "gpt" in l:
                    fake_label_id = idx
                    break
            
            if fake_label_id is not None:
                ai_probability = probs[0][fake_label_id].item()
            else:
                # Fallback: usually 1 is positive/fake/ai in many datasets
                if 1 in id2label:
                    ai_probability = probs[0][1].item()
                else:
                    ai_probability = probs[0][0].item()

        return ai_probability

    def predict_detailed(self, text: str):
        if not self.model:
            return {"score": 0.5, "sentences": []}
            
        # 1. Simple sentence splitting
        # Split by . ! ? and newlines
        raw_chunks = re.split(r'(?<=[.!?\n])\s+', text)
        
        sentences = []
        scores = []
        for chunk in raw_chunks:
            chunk = chunk.strip()
            if not chunk: continue
            
            # Skip very short chunks/noise
            if len(chunk) < 10:
                continue

            score = self.predict(chunk)
            scores.append(score)
            sentences.append({
                "text": chunk,
                "ai_probability": score
            })
            
        if not sentences:
             # If splitting failed or empty, run on full text
             score = self.predict(text)
             return {
                 "score": score, 
                 "sentences": [{
                     "text": text,
                     "ai_probability": score
                 }]
             }
             
        overall_score = sum(scores) / len(scores)
        
        return {
            "score": overall_score,
            "sentences": sentences
        }
