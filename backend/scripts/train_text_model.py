import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
import pandas as pd

# CONFIG
# CONFIG
MODEL_NAME = "Hello-SimpleAI/chatgpt-detector-roberta"
BATCH_SIZE = 8
EPOCHS = 3
LEARNING_RATE = 2e-5
DATA_DIRS = [
    "datasets/text",  # Points to where balanced_ai_human_prompts.csv is
]

class AIContentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_and_merge_datasets():
    dfs = []
    print("Loading datasets...")
    
    for d_dir in DATA_DIRS:
        if not os.path.exists(d_dir):
            print(f"Skipping missing directory: {d_dir}")
            continue
            
        print(f"Scanning {d_dir}...")
        for root, dirs, files in os.walk(d_dir):
            for file in files:
                if file.endswith(".csv"):
                    path = os.path.join(root, file)
                    print(f"Found CSV: {path}")
                    try:
                        temp_df = pd.read_csv(path)
                        # Normalize columns
                        # we need 'text' and 'label' (0=human, 1=ai)
                        
                        # Common column names maps
                        # 'generated' -> label (1=generated)
                        # 'is_ai' -> label
                        # 'label' -> label
                        
                        if 'text' not in temp_df.columns:
                            # Try to find text column
                            # 'essay_text', 'content', 'prompt'
                            for potential in ['essay_text', 'content', 'prompt', 'sentence']:
                                if potential in temp_df.columns:
                                    temp_df.rename(columns={potential: 'text'}, inplace=True)
                                    break
                        
                        if 'label' not in temp_df.columns:
                            if 'generated' in temp_df.columns:
                                temp_df['label'] = temp_df['generated'].astype(int)
                            elif 'is_ai' in temp_df.columns:
                                temp_df['label'] = temp_df['is_ai'].astype(int)
                            else:
                                print(f"Warning: Could not identify label column in {path}. Columns: {temp_df.columns}")
                                continue
                                
                        if 'text' in temp_df.columns and 'label' in temp_df.columns:
                            dfs.append(temp_df[['text', 'label']])
                            print(f"Loaded {len(temp_df)} samples from {file}")
                    except Exception as e:
                        print(f"Error loading {path}: {e}")
    
    if not dfs:
        return None
        
    return pd.concat(dfs, ignore_index=True)

def train_model():
    df = load_and_merge_datasets()
    if df is None or len(df) == 0:
        print("No data loaded. Please run download_datasets.py")
        return
        
    print(f"Total samples: {len(df)}")
    print(df['label'].value_counts())
    
    # Stratified split to ensure balance
    train_df, val_df = train_test_split(df, test_size=0.1, stratify=df['label'], random_state=42)
    
    print(f"Training on {len(train_df)} samples, Validating on {len(val_df)} samples.")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    train_dataset = AIContentDataset(train_df['text'].to_numpy(), train_df['label'].to_numpy(), tokenizer)
    val_dataset = AIContentDataset(val_df['text'].to_numpy(), val_df['label'].to_numpy(), tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    
    best_loss = float('inf')
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f}")
        
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += outputs.loss.item()
                
                preds = torch.argmax(outputs.logits, dim=1)
                correct += (preds == labels).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = correct / len(val_dataset)
        print(f"Epoch {epoch+1}/{EPOCHS} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            print("Saving best model...")
            print("Saving best model...")
            model.save_pretrained("models/text_best")
            tokenizer.save_pretrained("models/text_best")
    
    print("Training complete.")

if __name__ == "__main__":
    train_model()
