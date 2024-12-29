import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import logging
from utils import get_shakespeare_dataloader
from torchvision.models import resnet18

def build_model(architecture="cnn", num_classes=100):
    """Builds and returns the specified model architecture."""
    if architecture.lower() == "cnn":
        return CNN(num_classes=num_classes)
    elif architecture.lower() == "resnet":
        return resnet18(pretrained=False, num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")

class CNN(nn.Module):
    def __init__(self, num_classes=100):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),  # Increased channels
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

def train_model(model, dataloader, criterion, optimizer, device):
    """Training loop for one epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, labels) in enumerate(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Clear gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        
        # Calculate loss
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistics
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if batch_idx % 100 == 0:
            logging.info(f'Batch: {batch_idx}, Loss: {loss.item():.3f}')

    accuracy = 100. * correct / total
    return total_loss / len(dataloader), accuracy

def evaluate_model(model, dataloader, device):
    """Evaluate the model on a given dataset."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100. * correct / total

def centralized_training(dataset="cifar100", architecture="cnn", epochs=10, batch_size=32, lr=0.01):
    """Centralized training with specified configuration"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Data loading
    if dataset.lower() == "cifar100":
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        trainset = torchvision.datasets.CIFAR100(
            root='./data', 
            train=True,
            download=True, 
            transform=transform
        )
        trainloader = DataLoader(
            trainset, 
            batch_size=batch_size,
            shuffle=True, 
            num_workers=2,
            drop_last=True
        )

        testset = torchvision.datasets.CIFAR100(
            root='./data', 
            train=False, 
            download=True, 
            transform=transform
        )
        testloader = DataLoader(
            testset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=2
        )
    elif dataset.lower() == "shakespeare":
        trainloader = get_shakespeare_dataloader(batch_size=batch_size)
        testloader = None  # Placeholder; requires valid Shakespeare test split
    else:
        raise ValueError(f"Dataset {dataset} not supported")

    # Model initialization
    model = build_model(architecture, num_classes=100 if dataset.lower() == "cifar100" else 2).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # Training loop
    for epoch in range(epochs):
        logging.info(f"\nEpoch {epoch+1}/{epochs}")
        loss, accuracy = train_model(model, trainloader, criterion, optimizer, device)
        logging.info(f"Train Loss: {loss:.3f}, Train Accuracy: {accuracy:.2f}%")
        scheduler.step()

    # Evaluate the model on the test set
    if testloader:
        test_accuracy = evaluate_model(model, testloader, device)
        logging.info(f"Test Accuracy: {test_accuracy:.2f}%")

    # Save the trained model
    torch.save(model.state_dict(), f"{architecture}_{dataset}.pth")
    logging.info(f"Model saved as {architecture}_{dataset}.pth")

    return model

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    model = centralized_training()
    logging.info("Training completed!")