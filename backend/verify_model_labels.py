from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_name = "Hello-SimpleAI/chatgpt-detector-roberta"
print(f"Loading {model_name}...")
model = AutoModelForSequenceClassification.from_pretrained(model_name)
print(f"ID2LABEL: {model.config.id2label}")
