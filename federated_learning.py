# File: federated_learning.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
from typing import List, Dict
import copy
from tqdm import tqdm
from centralized_training import build_model
from utils import get_cifar100_dataloader, split_dataset

def train_local_model(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, 
                      optimizer: torch.optim.Optimizer, device: torch.device, epochs: int = 1) -> float:
    """Train a local model on client data."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for epoch in range(epochs):
        for inputs, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches if num_batches > 0 else float('inf')

def evaluate_global_model(model: nn.Module, dataloader: DataLoader, device: torch.device) -> float:
    """Evaluate the global model on a validation set."""
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

def federated_training(dataset: str = "cifar100", architecture: str = "cnn", rounds: int = 10, 
                       clients: int = 5, local_epochs: int = 1, batch_size: int = 32, lr: float = 0.01):
    """Run federated learning across multiple clients."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset and split into clients
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    if dataset.lower() == "cifar100":
        full_dataset = torchvision.datasets.CIFAR100(root="./data", train=True, download=True, transform=transform)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    client_datasets = split_dataset(full_dataset, num_clients=clients)
    client_loaders = [DataLoader(ds, batch_size=batch_size, shuffle=True) for ds in client_datasets]

    # Global model initialization
    global_model = build_model(architecture, num_classes=100 if dataset.lower() == "cifar100" else 2).to(device)
    global_model.train()

    # Federated training loop
    for round in range(rounds):
        print(f"\nRound {round + 1}/{rounds}")

        # Local training on clients
        client_models = [copy.deepcopy(global_model) for _ in range(clients)]
        client_losses = []

        for client_idx, client_model in enumerate(client_models):
            print(f"Training on Client {client_idx + 1}/{clients}")
            optimizer = optim.SGD(client_model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
            criterion = nn.CrossEntropyLoss()
            client_loss = train_local_model(client_model, client_loaders[client_idx], criterion, optimizer, device, epochs=local_epochs)
            client_losses.append(client_loss)

        # Aggregate client models
        with torch.no_grad():
            global_state_dict = global_model.state_dict()
            for key in global_state_dict.keys():
                global_state_dict[key] = torch.stack([
                    client_models[i].state_dict()[key].float() for i in range(clients)
                ]).mean(dim=0)
            global_model.load_state_dict(global_state_dict)

        # Evaluate global model
        test_loader = DataLoader(
            torchvision.datasets.CIFAR100(root="./data", train=False, download=True, transform=transform),
            batch_size=batch_size, shuffle=False
        )
        test_accuracy = evaluate_global_model(global_model, test_loader, device)

        print(f"Round {round + 1} completed. Average client loss: {sum(client_losses) / len(client_losses):.4f}")
        print(f"Global model test accuracy: {test_accuracy:.2f}%")

    # Save global model
    torch.save(global_model.state_dict(), f"federated_{architecture}_{dataset}.pth")
    print(f"Global model saved as federated_{architecture}_{dataset}.pth")

if __name__ == "__main__":
    federated_training(dataset="cifar100", 
                       architecture="cnn", 
                       rounds=10, 
                       clients=5, 
                       local_epochs=1, 
                       batch_size=32, 
                       lr=0.01)

