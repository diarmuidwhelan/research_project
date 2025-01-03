import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler  
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json

class NumberRecogniser(nn.Module):
    def __init__(self, num_classes=35):
        super(NumberRecogniser, self).__init__()
        
        # Load pretrained backbone
        self.backbone = models.resnet18(pretrained=True)
        
        # Freeze early layers
        for param in list(self.backbone.parameters())[:-4]:
            param.requires_grad = False
        
        # Get number of features from final layer
        num_features = self.backbone.fc.in_features
        
        # Modified classifier head
        self.backbone.fc = nn.Sequential(
            # First dense block
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            # Second dense block
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            # Output layer - 35 classes (1-35)
            nn.Linear(128, 35)  
        )
    
    def forward(self, x):
        return self.backbone(x)

class JerseyNumberDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        try:
            row = self.data.iloc[idx]
            image = Image.open(row['image_path']).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            
            # Convert 'n/a' to -1 for non-visible numbers
            if row['number'] == 'n/a':
                number = -1
            else:
                try:
                    number = int(row['number'])
                except ValueError:
                    number = -1
            
            return image, torch.tensor(number, dtype=torch.long)
        
        except Exception as e:
            print(f"Error loading image {row['image_path']}: {str(e)}")
            return torch.zeros((3, 224, 224)), torch.tensor(-1)

def parse_annotations(annotations_dir, images_dir):
    data = []
    print("Parsing annotations...")
    
    for xml_file in tqdm(os.listdir(annotations_dir)):
        if not xml_file.endswith('.xml'):
            continue
        
        try:
            tree = ET.parse(os.path.join(annotations_dir, xml_file))
            root = tree.getroot()
            
            filename = root.find('filename').text.strip()
            image_path = os.path.join(images_dir, filename)
            
            if not os.path.exists(image_path):
                continue
            
            for obj in root.findall('object'):
                name = obj.find('name').text.strip()
                parts = name.split('-')
                
                # Get jersey number from name (format: jersey-color-number)
                number = parts[2] if len(parts) > 2 else 'n/a'
                
                if number != 'n/a':
                    try:
                        # Verify it's a valid number
                        int(number)
                        data.append({
                            'image_path': image_path,
                            'number': number
                        })
                    except ValueError:
                        continue
        
        except Exception as e:
            print(f"Error processing {xml_file}: {str(e)}")
            continue
    
    df = pd.DataFrame(data)
    print(f"\nFound {len(df)} valid number annotations")
    
    if len(df) > 0:
        print("\nNumber distribution:")
        print(df['number'].value_counts().head())
    
    return df

def plot_training_curves(train_losses, val_losses, train_accs, val_accs):
    plt.figure(figsize=(12, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Training Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.close()

def save_metrics(metrics, filename='training_metrics.txt'):
    with open(filename, 'w') as f:
        for epoch, metric in enumerate(metrics, 1):
            f.write(f"Epoch {epoch}:\n")
            for key, value in metric.items():
                f.write(f"{key}: {value}\n")
            f.write("\n")

def process_labels(labels):
    """Convert jersey numbers to zero-based indices"""
    # Subtract 1 from labels to convert 1-35 to 0-34 for indexing
    return labels - 1

def convert_predictions(predictions):
    """Convert model predictions back to jersey numbers"""
    # Add 1 to predictions to convert 0-34 back to 1-35
    return predictions + 1
            
def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=20):
    best_val_acc = 0.0
    metrics = {
        'train_acc': [], 'val_acc': [],
        'train_loss': [], 'val_loss': []
    }
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Training'):
            inputs = inputs.to(device)
            # Convert labels to zero-based indices and move to device
            labels = (labels - 1).long().to(device)  #  Convert to long tensor first
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        train_acc = 100. * train_correct / train_total
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc='Validation'):
                inputs = inputs.to(device)
                # Convert labels to zero-based indices and move to device
                labels = (labels - 1).long().to(device)  #  Convert to long tensor first
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
          # Update metrics
        metrics['train_acc'].append(train_acc)
        metrics['val_acc'].append(val_acc)
        metrics['train_loss'].append(running_loss/len(train_loader))
        metrics['val_loss'].append(val_loss/len(val_loader))
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Training Loss: {running_loss/len(train_loader):.4f}')
        print(f'Training Accuracy: {train_acc:.2f}%')
        print(f'Validation Loss: {val_loss/len(val_loader):.4f}')
        print(f'Validation Accuracy: {val_acc:.2f}%')
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
            }, 'best_number_classifier.pth')
    
    return model, metrics

class JerseyDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        try:
            row = self.data.iloc[idx]
            image = Image.open(row['image_path']).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            
            # Convert 'n/a' to -1, otherwise convert to int and ensure 1-35 range
            if row['number'] == 'n/a':
                number = torch.tensor(-1, dtype=torch.long)
            else:
                try:
                    num = int(row['number'])
                    if 1 <= num <= 35:
                        number = torch.tensor(num, dtype=torch.long)
                    else:
                        number = torch.tensor(-1, dtype=torch.long)
                except ValueError:
                    number = torch.tensor(-1, dtype=torch.long)
            
            return image, number
            
        except Exception as e:
            print(f"Error loading image {row['image_path']}: {str(e)}")
            return torch.zeros((3, 224, 224)), torch.tensor(-1, dtype=torch.long)

def create_data_loaders(annotations_dir, images_dir, batch_size=32):
    """Create data loaders for training and validation"""
    # Parse annotations
    df = parse_annotations(annotations_dir, images_dir)
    
    # Filter for valid numbers (1-35)
    df = df[df['number'].apply(lambda x: x != 'n/a' and 1 <= int(x) <= 35)]
    
    # Split data
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Create transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = JerseyDataset(train_df, transform=transform)
    val_dataset = JerseyDataset(val_df, transform=transform)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    return train_loader, val_loader

def main():
    # Set paths and parameters
    annotations_dir = "/Users/dwhelan/Documents/GAACV/players_dataset/annotations"
    images_dir = "/Users/dwhelan/Documents/GAACV/players_dataset"
    num_classes = 35  # Numbers 1-35
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    model = NumberRecogniser(num_classes=num_classes).to(device)
    
    # Define loss and optimiser
    criterion = nn.CrossEntropyLoss(ignore_index=-1)  # Ignore invalid numbers
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        annotations_dir, 
        images_dir,
        batch_size=32
    )
    
    # Train model
    print("Starting training...")
    model, metrics = train_model(
        model, train_loader, val_loader,
        criterion, optimizer, device
    )
    
    # Save training metrics
    with open('training_metrics.json', 'w') as f:
        json.dump(metrics, f)
    
    print("Training completed!")

if __name__ == "__main__":
    main()
