import logging

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18

from utils import get_shakespeare_dataloader, get_shakespeare_client_datasets


def build_model(architecture="cnn", num_classes=100):
    """Builds and returns the specified model architecture."""
    if architecture.lower() == "cnn":
        return ModernCNN(num_classes=num_classes)
        return CNN(num_classes=num_classes)
        return ImprovedCNN(num_classes=num_classes)
    elif architecture.lower() == "resnet":
        return resnet18(pretrained=False, num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")


# CNN
# Client loss: 2.2547
# Global model test accuracy: 46.56%

# ImprovedCNN
# Round 10 completed. Average client loss: 2.0069
# Global model test accuracy: 50.53%

# ModernCNN
# Round 10 completed. Average client loss: 2.0199
# Global model test accuracy: 50.85%


class CNN(nn.Module):
    def __init__(self, num_classes=100):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=256, hidden_dim=512, num_layers=3):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Predict the last character in the sequence
        return out
    
def build_model(architecture="cnn", num_classes=100):
    """Builds and returns the specified model architecture."""
    if architecture.lower() == "cnn":
        return CNN(num_classes=num_classes)
    elif architecture.lower() == "resnet":
        return resnet18(pretrained=False, num_classes=num_classes)
    elif architecture.lower() == "lstm":
        return LSTM(vocab_size=num_classes, embedding_dim=512, hidden_dim=1024, num_layers=3)  # Use num_classes as vocab_size
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1), nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = nn.ReLU()(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return nn.ReLU()(out)


class ImprovedCNN(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        self.features = nn.Sequential(
            ResidualBlock(3, 32),
            nn.MaxPool2d(2, 2),
            ResidualBlock(32, 64),
            nn.MaxPool2d(2, 2),
            ResidualBlock(64, 128),
            nn.MaxPool2d(2, 2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


class ModernCNN(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
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
            logging.info(f"Batch: {batch_idx}, Loss: {loss.item():.3f}")

    accuracy = 100.0 * correct / total
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
    return 100.0 * correct / total


def centralized_training(
    dataset="cifar100", architecture="cnn", epochs=10, batch_size=32, lr=0.01
):
    """Centralized training with specified configuration"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Data loading
    if dataset.lower() == "cifar100":
        transform = transforms.Compose(
            [
                # Geometric transformations
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(15),
                transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
                # Color/intensity transformations
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                ),
                transforms.RandomAutocontrast(p=0.2),
                # Regularization transforms
                transforms.RandAugment(
                    num_ops=2, magnitude=9
                ),  # Automated augmentation
                transforms.RandomErasing(p=0.1),  # Helps prevent overfitting
                # Required transforms
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        trainset = torchvision.datasets.CIFAR100(
            root="./data", train=True, download=True, transform=transform
        )
        trainloader = DataLoader(
            trainset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            drop_last=True,
        )

        testset = torchvision.datasets.CIFAR100(
            root="./data",
            train=False,
            download=True,
            transform=transform,
        )
        testloader = DataLoader(
            testset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
        )
    elif dataset.lower() == "shakespeare":
        trainloader, testloader, idx_to_char = get_shakespeare_client_datasets(batch_size=batch_size, num_clients=1)
        trainloader = trainloader[0]
        num_classes = len(idx_to_char)
    else:
        raise ValueError(f"Dataset {dataset} not supported")

    # Model initialization
    model = build_model(
        architecture, num_classes=100 if dataset.lower() == "cifar100" else 2
    ).to(device)

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
