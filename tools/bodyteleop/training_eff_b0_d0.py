# 1. Import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import argparse

class SelfDrivingDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.dataframe = pd.read_csv(csv_file)
        self.transform = transform
        self.label_map = {'F': 0, 'R': 1, 'L': 2}

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = self.dataframe.iloc[idx, 0]
        image = Image.open(img_name)
        
        # Convert grayscale to RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        label_str = self.dataframe.iloc[idx, 1]
        label = torch.tensor(self.label_map[label_str], dtype=torch.long)
        
        if self.transform:
            image = self.transform(image)

        return image, label

# Argument parser
parser = argparse.ArgumentParser(description="Training a Self-Driving Robot")
parser.add_argument("output_dir", type=str, help="Directory to save the checkpoints")
args = parser.parse_args()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = SelfDrivingDataset(csv_file="train_dataset.csv", transform=train_transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

val_dataset = SelfDrivingDataset(csv_file="val_dataset.csv", transform=transform)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

model = models.efficientnet_b0(pretrained=True)
num_features = model.classifier[-1].in_features

# Add dropout layer and the fully connected layer to the classifier
model.classifier = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(num_features, 3)
)
# num_features = model.classifier[-1].in_features
# model.classifier[-1] = nn.Linear(num_features, 3)

for param in model.parameters():
    param.requires_grad = False

for param in model.classifier.parameters():
    param.requires_grad = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    corrects = 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        corrects += torch.sum(preds == labels.data)
        
    return total_loss / len(loader.dataset), corrects.double() / len(loader.dataset)

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    corrects = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            corrects += torch.sum(preds == labels.data)

    return total_loss / len(loader.dataset), corrects.double() / len(loader.dataset)

num_epochs = 25
N = 5
best_val_acc = 0.0

for epoch in range(num_epochs):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = validate(model, val_loader, criterion, device)

    print(f"Epoch {epoch+1}/{num_epochs} - Train loss: {train_loss:.4f}, Train accuracy: {train_acc:.4f}, Validation loss: {val_loss:.4f}, Validation accuracy: {val_acc:.4f}")

    if (epoch + 1) % N == 0 or val_acc > best_val_acc:
        if val_acc > best_val_acc:
            best_val_acc = val_acc

        checkpoint_name = os.path.join(args.output_dir, f'efficientnet_b0-checkpoint_epoch{epoch+1}_valacc{val_acc:.4f}.pth')
        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_val_acc': best_val_acc
        }
        torch.save(checkpoint, checkpoint_name)
        print(f"Model checkpoint saved as {checkpoint_name}")

model_path = os.path.join(args.output_dir, "self_driving_rc_model_efficient_b0.pth")
torch.save(model.state_dict(), model_path)
model.load_state_dict(torch.load(model_path))
