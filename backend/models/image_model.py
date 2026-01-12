from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch
import torch.nn.functional as F
import io
import os
from torchvision import transforms

class ImageDetector:
    def __init__(self):
        print("Loading Image Detection Model...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Priority: Custom Trained Model -> Fallback: Pretrained
        self.custom_model_path = "models/image_best.pth"
        self.use_custom = False

        try:
            if os.path.exists(self.custom_model_path):
                print(f"Found custom trained model: {self.custom_model_path}")
                from torchvision import models, transforms
                import torch.nn as nn
                
                # Reconstruct ResNet50
                self.model = models.resnet50(pretrained=False)
                num_ftrs = self.model.fc.in_features
                self.model.fc = nn.Linear(num_ftrs, 2)
                
                # Load weights
                self.model.load_state_dict(torch.load(self.custom_model_path, map_location=self.device))
                self.model = self.model.to(self.device)
                self.model.eval()
                self.use_custom = True
                
                # Define transforms (must match training)
                self.transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
                print("Custom ResNet Model Loaded Successfully.")
            else:
                print("Custom model not found. Using Fallback.")
                model_name = "Organika/sdxl-detector"
                self.processor = AutoImageProcessor.from_pretrained(model_name)
                self.model = AutoModelForImageClassification.from_pretrained(model_name).to(self.device)
                self.model.eval()
                print(f"Fallback Image Model {model_name} Loaded Successfully.")
                
        except Exception as e:
            print(f"Error loading image model: {e}")
            self.model = None

    def predict(self, image_bytes):
        if not self.model:
            return 0.5

        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            
            if self.use_custom:
                # Custom Model Logic (ResNet)
                image_tensor = self.transform(image).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    outputs = self.model(image_tensor)
                    probs = F.softmax(outputs, dim=1)
                    # Assuming Class 0: Fake, Class 1: Real (based on training folder alphabetic order usually)
                    # Folder order: Fake, real. So 0=Fake, 1=Real.
                    # We want probability of AI (Fake).
                    # If 0 is Fake, then prob[0] is AI prob.
                    ai_probability = probs[0][0].item()
                    
            else:
                # Fallback Logic (HuggingFace)
                inputs = self.processor(images=image, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits = outputs.logits
                    probs = F.softmax(logits, dim=-1)

                id2label = self.model.config.id2label
                ai_label_id = None
                
                # Smart label detection
                for idx, label in id2label.items():
                    l = label.lower()
                    if "ai" in l or "fake" in l or "artificial" in l or "computer" in l or "generated" in l:
                        ai_label_id = idx
                        break
                
                if ai_label_id is not None:
                    ai_probability = probs[0][ai_label_id].item()
                else:
                    ai_probability = probs[0][1].item() 

            return ai_probability
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return 0.5
