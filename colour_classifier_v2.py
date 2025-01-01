import os
import xml.etree.ElementTree as ET
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
import json 
import matplotlib.pyplot as plt 

class JerseyDataset(Dataset):
    """Dataset for jersey color classification"""
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            image = Image.open(self.image_paths[idx]).convert('RGB')
            if self.transform:
                image = self.transform(image)
            label = self.labels[idx]
            return image, label
        except Exception as e:
            print(f"Error loading image {self.image_paths[idx]}: {str(e)}")
            dummy_image = torch.zeros((3, 224, 224))
            return dummy_image, self.labels[idx]

def parse_annotations(annotations_dir, images_dir):
    data = []
    print("Parsing annotations...")
    
    skipped_files = 0
    skipped_objects = 0
    
    for xml_file in os.listdir(annotations_dir):
        if not xml_file.endswith('.xml'):
            continue
            
        try:
            tree = ET.parse(os.path.join(annotations_dir, xml_file))
            root = tree.getroot()
            
            filename = root.find('filename').text.strip()
            image_path = os.path.join(images_dir, filename)
            
            # Skip if image doesn't exist
            if not os.path.exists(image_path):
                print(f"Image not found: {image_path}")
                skipped_files += 1
                continue
            
            for obj in root.findall('object'):
                try:
                    name = obj.find('name').text.strip()
                    
                    # Handle different name formats
                    parts = name.split('-')
                    
                    # Skip objects marked as 'ignore'
                    if name == 'ignore' or len(parts) < 2:
                        skipped_objects += 1
                        continue
                    
                    # Check if it's a jersey annotation
                    if parts[0] == 'jersey':
                        color = parts[1]
                        data.append({
                            'image_path': image_path,
                            'jersey_color': color
                        })
                    else:
                        skipped_objects += 1
                        
                except Exception as e:
                    print(f"Error processing object in {xml_file}: {str(e)}")
                    skipped_objects += 1
                    continue
                    
        except Exception as e:
            print(f"Error processing file {xml_file}: {str(e)}")
            skipped_files += 1
            continue
    
    df = pd.DataFrame(data)
    
    # Print summary statistics
    print("\nDataset Summary:")
    print(f"Total annotations: {len(df)}")
    if len(df) > 0:
        print("\nJersey color distribution:")
        print(df['jersey_color'].value_counts())
        print(f"\nNumber of unique images: {df['image_path'].nunique()}")
    print(f"\nSkipped files: {skipped_files}")
    print(f"Skipped objects: {skipped_objects}")
    
    return df

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    """Train model and save metrics"""
    best_val_acc = 0.0
    
    # Initialize metrics storage
    metrics = {
        'train_acc': [],
        'val_acc': [],
        'train_loss': [],
        'val_loss': []
    }
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Training'):
            inputs, labels = inputs.to(device), labels.to(device)
            
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
        train_loss = running_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc='Validation'):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
        val_loss = val_loss / len(val_loader)
        
        # Save metrics
        metrics['train_acc'].append(train_acc)
        metrics['val_acc'].append(val_acc)
        metrics['train_loss'].append(train_loss)
        metrics['val_loss'].append(val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Training Loss: {train_loss:.4f}')
        print(f'Training Accuracy: {train_acc:.2f}%')
        print(f'Validation Loss: {val_loss:.4f}')
        print(f'Validation Accuracy: {val_acc:.2f}%')
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
            }, 'best_jersey_classifier.pth')
    
    # Save metrics to file
    with open('jersey_classifier_metrics.json', 'w') as f:
        json.dump(metrics, f)
    
    return model, metrics

def plot_training_curves(metrics):
    """Plot training curves from metrics dictionary"""
    plt.figure(figsize=(15, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(metrics['train_acc'], 'b-', label='Training Accuracy', marker='o')
    plt.plot(metrics['val_acc'], 'r-', label='Validation Accuracy', marker='o')
    plt.title('Model Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(metrics['train_loss'], 'b-', label='Training Loss', marker='o')
    plt.plot(metrics['val_loss'], 'r-', label='Validation Loss', marker='o')
    plt.title('Model Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.close()

def create_data_loaders(df, batch_size=32):
    """Create train, validation, and test data loaders"""
    # Split data into train, validation, and test sets
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    
    # Define transformations
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Validation/Test transforms (no augmentation)
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = JerseyDataset(
        train_df['image_path'].values,
        train_df['label'].values,
        transform=train_transform
    )
    
    val_dataset = JerseyDataset(
        val_df['image_path'].values,
        val_df['label'].values,
        transform=eval_transform
    )
    
    test_dataset = JerseyDataset(
        test_df['image_path'].values,
        test_df['label'].values,
        transform=eval_transform
    )
    
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
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    return train_loader, val_loader, test_loader



    
def main():
    # Set paths
    annotations_dir = "/Users/dwhelan/Documents/GAACV/players_dataset/annotations"
    images_dir = "/Users/dwhelan/Documents/GAACV/players_dataset"
    
    # Set hyperparameters
    batch_size = 32
    num_epochs = 20
    learning_rate = 0.001
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Parse annotations
    print("Parsing annotations...")
    df = parse_annotations(annotations_dir, images_dir)
    
    if df.empty:
        print("Error: No valid annotations found. Please check the paths and data format.")
        return
    
    # Encode labels
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['jersey_color'])
    num_classes = len(le.classes_)
    
    # Save label encoder classes
    with open('jersey_color_classes.txt', 'w') as f:
        for i, class_name in enumerate(le.classes_):
            f.write(f"{i}: {class_name}\n")
    
    # Split data
    train_loader, val_loader, test_loader = create_data_loaders(df, batch_size)
    
    print(f"Training with {len(train_loader.dataset)} samples")
    print(f"Validating with {len(val_loader.dataset)} samples")
    print(f"Testing with {len(test_loader.dataset)} samples")
    
    # Initialise model
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)
    
    # Define loss function and optimiser
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        

    # Train model
    print("Starting training...")
    model, metrics = train_model(model, train_loader, val_loader, criterion, 
                               optimizer, num_epochs, device)
    
    # Plot training curves
    plot_training_curves(metrics)
    
    print("Training completed!")
    print(f"Model saved as 'best_jersey_classifier.pth'")
    print(f"Metrics saved as 'jersey_classifier_metrics.json'")
    print(f"Training curves saved as 'training_curves.png'")

    
    # Evaluate on test set
    model.eval()
    test_correct = 0
    test_total = 0
    
    print("\nEvaluating on test set...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()
    
    test_acc = 100. * test_correct / test_total
    print(f'Test Accuracy: {test_acc:.2f}%')

if __name__ == "__main__":
    main()
