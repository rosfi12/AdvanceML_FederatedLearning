import copy
import logging
import os
from collections import defaultdict
from os import PathLike
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm.auto import tqdm

from centralized_training import ModernCNN


class MetricsTracker:
    def __init__(
        self,
        num_clients: int,
        log_dir: Union[str, PathLike] = os.path.join("runs", "fed_learning"),
    ):
        self.writer = SummaryWriter(log_dir)
        self.metrics: Dict[str, List[float]] = defaultdict(list)
        self.num_clients = num_clients
        self.round_metrics: Dict[str, float] = {}

    def update_client_metrics(
        self,
        client_id: int,
        round_idx: int,
        train_loss: float,
        train_acc: float,
        val_acc: float,
    ) -> None:
        """Update per-client metrics."""
        self.writer.add_scalar(f"Client_{client_id}/Train_Loss", train_loss, round_idx)
        self.writer.add_scalar(f"Client_{client_id}/Train_Acc", train_acc, round_idx)
        self.writer.add_scalar(f"Client_{client_id}/Val_Acc", val_acc, round_idx)

    def update_global_metrics(
        self, round_idx: int, test_acc: float, avg_client_acc: float
    ) -> None:
        """Update global model metrics."""
        self.writer.add_scalar("Global/Test_Accuracy", test_acc, round_idx)
        self.writer.add_scalar("Global/Avg_Client_Accuracy", avg_client_acc, round_idx)
        self.metrics["test_acc"].append(test_acc)
        self.metrics["avg_client_acc"].append(avg_client_acc)

    def plot_training_progress(
        self, save_path: Union[str, PathLike] = "training_progress.png"
    ) -> None:
        """Plot and save training metrics."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.metrics["test_acc"], label="Global Test Accuracy")
        plt.plot(self.metrics["avg_client_acc"], label="Avg Client Accuracy")
        plt.xlabel("Round")
        plt.ylabel("Accuracy (%)")
        plt.title("Federated Learning Progress")
        plt.legend()
        plt.savefig(save_path)
        plt.close()


def create_client_dataloaders(
    dataset: torchvision.datasets.VisionDataset,
    num_clients: int,
    batch_size: int,
    val_split: float = 0.1,
) -> List[Tuple[DataLoader, DataLoader]]:
    """Split dataset into client dataloaders with train/val splits."""
    # Calculate sizes
    total_size = len(dataset)
    client_size = total_size // num_clients
    client_sizes = [client_size] * num_clients
    client_sizes[-1] += total_size % num_clients

    # Split into clients
    client_datasets = random_split(dataset, client_sizes)
    client_loaders = []

    # Set DataLoader configuration based on device
    use_cuda = torch.cuda.is_available()
    # Basic settings that work for both CPU and CUDA
    dataloader_kwargs = {
        "batch_size": batch_size,
        "pin_memory": use_cuda,
    }

    for client_dataset in client_datasets:
        # Create train/val split
        val_size = int(len(client_dataset) * val_split)
        train_size = len(client_dataset) - val_size
        train_dataset, val_dataset = random_split(
            client_dataset, [train_size, val_size]
        )

        # Create dataloaders
        train_loader = DataLoader(train_dataset, shuffle=True, **dataloader_kwargs)
        val_loader = DataLoader(val_dataset, shuffle=False, **dataloader_kwargs)
        client_loaders.append((train_loader, val_loader))

    return client_loaders


def train_client(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    client_id: int,
    device: torch.device,
    epochs: int,
    lr: float,
    metrics: MetricsTracker,
    round_idx: int,
) -> Tuple[Optional[nn.Module], float, float]:
    """Train a client model and return metrics."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    best_val_acc = 0.0
    best_model = None

    train_acc = 0.0

    epoch_pbar = tqdm(range(epochs), desc=f"Client {client_id}", leave=False)
    for epoch in epoch_pbar:
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        batch_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False)
        for inputs, labels in batch_pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            batch_pbar.set_postfix(
                {"loss": f"{loss.item():.3f}", "acc": f"{100.*correct/total:.2f}%"}
            )

        train_loss = running_loss / len(train_loader)
        train_acc = 100.0 * correct / total
        val_acc = evaluate_model(model, val_loader, device)

        metrics.update_client_metrics(
            client_id, round_idx * epochs + epoch, train_loss, train_acc, val_acc
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = copy.deepcopy(model)

        scheduler.step()
        epoch_pbar.set_postfix(
            {
                "train_loss": f"{train_loss:.3f}",
                "train_acc": f"{train_acc:.2f}%",
                "val_acc": f"{val_acc:.2f}%",
            }
        )

    return best_model, train_acc, best_val_acc


def evaluate_model(
    model: nn.Module, dataloader: DataLoader, device: torch.device
) -> float:
    """Evaluate model accuracy."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100.0 * correct / total


def federated_learning(
    num_clients: int = 5,
    rounds: int = 10,
    epochs: int = 5,
    batch_size: int = 64,
    lr: float = 0.001,
    save_dir: Union[str, PathLike] = "federated_models",
) -> nn.Module:
    """Run federated learning simulation with enhanced metrics tracking."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    metrics = MetricsTracker(num_clients)
    Path(save_dir).mkdir(exist_ok=True)
    best_model_path = os.path.join(save_dir, "best_model.pth")

    transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=2),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ]
    )

    dataset = torchvision.datasets.CIFAR100(
        root=os.path.join(".", "data"), train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.CIFAR100(
        root=os.path.join(".", "data"),
        train=False,
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
                ),
            ]
        ),
    )

    client_loaders = create_client_dataloaders(dataset, num_clients, batch_size)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )

    global_model = ModernCNN(num_classes=100).to(device)
    best_accuracy = 0.0

    rounds_pbar = tqdm(range(rounds), desc="Federated Learning Rounds")
    for round_idx in rounds_pbar:
        client_models = []
        client_train_accs = []
        client_val_accs = []

        for client_idx, (train_loader, val_loader) in enumerate(client_loaders):
            client_model = copy.deepcopy(global_model)
            client_model, train_acc, val_acc = train_client(
                client_model,
                train_loader,
                val_loader,
                client_idx,
                device,
                epochs,
                lr,
                metrics,
                round_idx,
            )
            if client_model is not None:
                client_models.append(client_model)
                client_train_accs.append(train_acc)
                client_val_accs.append(val_acc)

        # Model aggregation (FedAvg)
        with torch.no_grad():
            global_dict = global_model.state_dict()
            for key in global_dict.keys():
                global_dict[key] = torch.mean(
                    torch.stack([model.state_dict()[key] for model in client_models]),
                    dim=0,
                )
            global_model.load_state_dict(global_dict)

        test_accuracy = evaluate_model(global_model, test_loader, device)
        avg_client_acc = sum(client_val_accs) / len(client_val_accs)

        metrics.update_global_metrics(round_idx, test_accuracy, avg_client_acc)

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            torch.save(global_model.state_dict(), best_model_path)

        rounds_pbar.set_postfix(
            {
                "test_acc": f"{test_accuracy:.2f}%",
                "best_acc": f"{best_accuracy:.2f}%",
                "client_avg": f"{avg_client_acc:.2f}%",
            }
        )

    metrics.plot_training_progress(os.path.join(save_dir, "training_progress.png"))
    metrics.writer.close()

    print(f"\nFederated Learning completed. Best test accuracy: {best_accuracy:.2f}%")
    return global_model


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    model = federated_learning(
        num_clients=5, rounds=10, epochs=5, batch_size=64, lr=0.001
    )
