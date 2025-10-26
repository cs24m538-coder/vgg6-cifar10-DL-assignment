import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import json

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
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_data = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VGG6(activation=config['activation']).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.CrossEntropyLoss()
    
    # Quick training
    for epoch in range(2):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
    
    torch.save(model.state_dict(), 'model.pth')
    print("Training completed!")

if __name__ == '__main__':
    train()
