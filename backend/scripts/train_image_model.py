import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import os

# CONFIG
DATA_DIRS = [
    "datasets/image/cifake",
    "datasets/image/140k_faces"
]
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 0.001

def train_image_model():
    print("Initializing Image Model Training...")
    
    # Enhanced Transformations
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.4736, 0.4663, 0.4210], [0.2033, 0.2025, 0.2030])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.4736, 0.4663, 0.4210], [0.2033, 0.2025, 0.2030])
        ]),
    }
    
    # Combined Dataset
    full_datasets = []
    
    for d_dir in DATA_DIRS:
        if os.path.exists(d_dir):
            # Check for standard train/val split or just flat structure
            # Case 1: has train/val (CIFAKE)
            train_dir = os.path.join(d_dir, 'train')
            if os.path.exists(train_dir):
                print(f"Loading dataset from: {train_dir}")
                full_datasets.append(datasets.ImageFolder(train_dir, data_transforms['train']))
            else:
                # Case 2: maybe it's the root itself or 140k faces structure
                # We will try to load from root if it has class folders
                try:
                    print(f"Loading dataset from root: {d_dir}")
                    full_datasets.append(datasets.ImageFolder(d_dir, data_transforms['train']))
                except Exception as e:
                    print(f"Skipping {d_dir}: {e}")
        else:
            print(f"Warning: Dataset directory {d_dir} not found. Skipping.")

    if not full_datasets:
        print("No datasets found. Please run scripts/download_datasets.py")
        return

    # Concat all datasets
    combined_dataset = torch.utils.data.ConcatDataset(full_datasets)
    
    # Split into Train/Val (80/20) since we are merging different sources
    train_size = int(0.8 * len(combined_dataset))
    val_size = len(combined_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(combined_dataset, [train_size, val_size])
    
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4),
        'val': DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    }
    dataset_sizes = {'train': train_size, 'val': val_size}
                   
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Use ResNet50
    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    # Check number of classes based on folder names? Or assume 2.
    # The dataset folder has 'Fake' and 'real', which maps to 2 classes.
    model.fc = nn.Linear(num_ftrs, 2) # 2 classes: Real vs Fake
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    best_acc = 0.0
    
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")
        print('-' * 10)
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
                
            running_loss = 0.0
            running_corrects = 0
            
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
            if phase == 'train':
                scheduler.step()
                
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
            
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                print("New best model! Saving...")
                torch.save(model.state_dict(), "models/image_best.pth")
        
    print("Training complete.")
    torch.save(model.state_dict(), "models/image_resnet_finetuned.pth")

if __name__ == "__main__":
    train_image_model()

