#"""
#VGG6 CIFAR-10 Training Script
#Deep Learning Assignment
#Student: [SuriyaNarayani]
#Date: [26-Oct-2025]

#A complete training pipeline for VGG6 on CIFAR-10 with configurable parameters.
#"""
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import json
import os

# Set seed for reproducibility
torch.manual_seed(42)

class VGG6(nn.Module):
    def __init__(self, activation='relu'):
        super().__init__()
        if activation == 'silu': 
            act = nn.SiLU()
        else: 
            act = nn.ReLU()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), act,
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), act,
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), act,
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), act,
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(128, 10)
        )
    
    def forward(self, x):
        return self.classifier(self.features(x))

def train():
    print(" Starting VGG6 CIFAR-10 Training Demo")
    
    # UNIVERSAL PATH HANDLING
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        script_dir = os.getcwd()
    
    config_path = os.path.join(script_dir, 'config.json')
    
    # Load config
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f" Config: {config}")
    except Exception as e:
        print(f" Config error: {e}")
        return
    
    # SIMPLE TRANSFORMS
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # SMALL DATASET FOR DEMO (only 1000 samples)
    try:
        train_data = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
        # Use only a small subset for quick demo
        indices = torch.arange(1000)
        from torch.utils.data import Subset
        train_subset = Subset(train_data, indices)
        train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
        print(f" Using small dataset: {len(train_subset)} samples")
    except Exception as e:
        print(f" Data error: {e}")
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f" Device: {device}")
    
    # Setup model
    model = VGG6(activation=config['activation']).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.CrossEntropyLoss()
    
    print(" Training for 2 epochs (quick demo)...")
    
    # FAST TRAINING - only 2 epochs, small batches
    for epoch in range(2):
        total_loss = 0
        for i, (data, target) in enumerate(train_loader):
            if i >= 10:  # Only do 10 batches per epoch
                break
                
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / min(10, len(train_loader))
        print(f" Epoch {epoch+1}: Loss = {avg_loss:.4f}")
    
    # Save model
    model_path = os.path.join(script_dir, 'demo_model.pth')
    torch.save(model.state_dict(), model_path)
    print(f" Demo model saved: {model_path}")
    print(" DEMO COMPLETED SUCCESSFULLY!")
    print(" Note: This is a quick demo. For full training, increase epochs and dataset size.")

if __name__ == '__main__':
    train()
