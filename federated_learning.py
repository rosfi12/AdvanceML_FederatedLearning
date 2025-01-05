import copy
import logging
import os
import shutil
from collections import defaultdict
from datetime import datetime
from os import PathLike
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm.auto import tqdm

from centralized_training import ModernCNN

MIN_WORKERS = 2
logging.basicConfig(format="[%(levelname)s] %(message)s", level=logging.INFO)
logging.addLevelName(
    logging.INFO, "\033[1;32m%s\033[1;0m" % logging.getLevelName(logging.INFO)
)


class MetricsTracker:
    def __init__(
        self,
        num_clients: int,
        log_dir: Union[str, PathLike] = os.path.join("runs", "fed_learning"),
        run_name: Optional[str] = None,
    ):
        """Initialize metrics tracker with run identification."""
        # Create unique run name with timestamp if not provided
        if run_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"run_{timestamp}"

        # Create unique directory for this run
        self.log_dir = os.path.join(log_dir, run_name)

        # Clean up existing logs for this run
        if os.path.exists(self.log_dir):
            shutil.rmtree(self.log_dir)

        self.writer = SummaryWriter(self.log_dir)
        self.metrics: Dict[str, List[float]] = defaultdict(list)
        self.num_clients = num_clients
        self.round_metrics: Dict[str, Dict[str, List[float]]] = {}

    def update_client_metrics(
        self,
        client_id: int,
        round_idx: int,
        train_loss: float,
        train_acc: float,
        val_acc: float,
        lr: float,
    ) -> None:
        """Update per-client metrics with better organization."""
        # Per-client metrics
        self.writer.add_scalars(
            f"Client_{client_id}",
            {
                "Train_Loss": train_loss,
                "Train_Accuracy": train_acc,
                "Val_Accuracy": val_acc,
                "Learning_Rate": lr,
            },
            round_idx,
        )

        # Add to round aggregates
        round_key: str = f"Round_{round_idx}"
        if round_key not in self.round_metrics:
            self.round_metrics[round_key] = {
                "train_losses": [],
                "train_accs": [],
                "val_accs": [],
            }

        self.round_metrics[round_key]["train_losses"].append(train_loss)
        self.round_metrics[round_key]["train_accs"].append(train_acc)
        self.round_metrics[round_key]["val_accs"].append(val_acc)

    def update_round_metrics(self, round_idx: int) -> None:
        """Compute and save round-level aggregate metrics."""
        round_key = f"Round_{round_idx}"
        if round_key in self.round_metrics:
            metrics = self.round_metrics[round_key]

            # Calculate round statistics
            avg_train_loss = sum(metrics["train_losses"]) / len(metrics["train_losses"])
            avg_train_acc = sum(metrics["train_accs"]) / len(metrics["train_accs"])
            avg_val_acc = sum(metrics["val_accs"]) / len(metrics["val_accs"])

            # Log round averages
            self.writer.add_scalars(
                "Round_Averages",
                {
                    "Avg_Train_Loss": avg_train_loss,
                    "Avg_Train_Accuracy": avg_train_acc,
                    "Avg_Val_Accuracy": avg_val_acc,
                },
                round_idx,
            )

    def update_global_metrics(
        self, round_idx: int, test_acc: float, avg_client_acc: float
    ) -> None:
        """Update global model metrics."""
        self.writer.add_scalars(
            "Global",
            {"Test_Accuracy": test_acc, "Avg_Client_Accuracy": avg_client_acc},
            round_idx,
        )
        self.metrics["test_acc"].append(test_acc)
        self.metrics["avg_client_acc"].append(avg_client_acc)

        # Update round metrics
        self.update_round_metrics(round_idx)


def create_client_dataloaders(
    dataset: torchvision.datasets.VisionDataset,
    num_clients: int,
    batch_size: int,
    val_split: float = 0.1,
    cpu_count: int = os.cpu_count() or 1,
) -> List[Tuple[DataLoader, DataLoader]]:
    """Split dataset into client dataloaders with optimized parallel processing."""
    # Calculate optimal number of workers preventing over-subscription
    num_workers = min(MIN_WORKERS, cpu_count // num_clients)

    # Pre-compute all splits at once
    total_size = len(dataset)
    client_size = total_size // num_clients
    client_sizes = [client_size] * num_clients
    client_sizes[-1] += total_size % num_clients

    # Calculate validation sizes
    val_sizes = [int(size * val_split) for size in client_sizes]
    train_sizes = [size - val_size for size, val_size in zip(client_sizes, val_sizes)]

    # Create all splits at once
    all_splits = random_split(
        dataset, train_sizes + val_sizes, generator=torch.Generator().manual_seed(42)
    )

    # Configure shared memory settings
    torch.multiprocessing.set_sharing_strategy("file_system")

    # Optimize DataLoader settings
    use_cuda = torch.cuda.is_available()
    dataloader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": use_cuda,
        "persistent_workers": True,  # Keep workers alive between epochs
        "prefetch_factor": 2,
        "pin_memory_device": "cuda" if use_cuda else "",
        "generator": torch.Generator().manual_seed(42),
    }

    client_loaders = []

    # Create dataloaders for each client
    for i in range(num_clients):
        train_dataset = all_splits[i]
        val_dataset = all_splits[i + num_clients]

        train_loader = DataLoader(
            train_dataset, shuffle=True, drop_last=True, **dataloader_kwargs
        )
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

    epochs_pbar = tqdm(
        range(epochs),
        desc=f"Client {client_id} Epochs",
        position=1,
        leave=True
    )
    for epoch in epochs_pbar:
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        batch_pbar = tqdm(
            train_loader, desc=f"Epoch {epoch+1}", position=2, leave=False
        )
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
            client_id, round_idx * epochs + epoch, train_loss, train_acc, val_acc, lr
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = copy.deepcopy(model)

        scheduler.step()
        epochs_pbar.set_postfix(
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


def aggregate_models(models: List[nn.Module]) -> Dict[str, torch.Tensor]:
    """Aggregate model parameters with proper dtype handling."""
    state_dicts = [model.state_dict() for model in models]
    aggregated_dict = {}

    for key in state_dicts[0].keys():
        params = [state_dict[key] for state_dict in state_dicts]
        if params[0].dtype in [torch.int32, torch.int64, torch.long]:
            # For integer parameters (like batch norm running mean/var counts)
            aggregated_dict[key] = params[0].clone()  # Just keep the first one
        else:
            # For floating point parameters
            stacked = torch.stack([param.to(torch.float32) for param in params])
            avg_param = torch.mean(stacked, dim=0)
            # Convert back to original dtype
            aggregated_dict[key] = avg_param.to(params[0].dtype)

    return aggregated_dict


def federated_learning(
    num_clients: int = 5,
    rounds: int = 10,
    epochs: int = 5,
    batch_size: int = 64,
    lr: float = 0.001,
    save_dir: Union[str, PathLike] = "federated_models",
    num_workers: int = min(MIN_WORKERS, os.cpu_count() or 1),
    run_name: Optional[str] = None,
) -> nn.Module:
    """Run federated learning simulation with enhanced metrics tracking."""

    timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")
    if run_name is None:
        config_str = f"c{num_clients}_r{rounds}_e{epochs}_b{batch_size}_lr{lr}"
        run_name = f"{config_str}_{timestamp}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device} for run: {run_name}")

    metrics = MetricsTracker(num_clients, run_name=run_name)
    Path(save_dir).mkdir(exist_ok=True)
    best_model_path = os.path.join(save_dir, f"best_model_{timestamp}.pth")

    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=2),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ]
    )

    dataset = torchvision.datasets.CIFAR100(
        root=os.path.join(".", "data"),
        train=True,
        download=True,
        transform=transform_train,
    )
    test_dataset = torchvision.datasets.CIFAR100(
        root=os.path.join(".", "data"),
        train=False,
        download=False,  # Download only once
        transform=transform_test,
    )

    client_loaders = create_client_dataloaders(dataset, num_clients, batch_size)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size * 2,  # Larger batch size for evaluation
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True,
        prefetch_factor=2,
        pin_memory_device="cuda" if torch.cuda.is_available() else "",
    )

    global_model = ModernCNN(num_classes=100).to(device)
    best_accuracy = 0.0

    rounds_pbar = tqdm(range(rounds), desc="Federated Learning Rounds")
    for round_idx in rounds_pbar:
        client_models = []
        client_train_accs = []
        client_val_accs = []

        clients_pbar = tqdm(
            enumerate(client_loaders),
            total=len(client_loaders),
            desc="Training Clients",
            leave=False,
        )

        for client_idx, (train_loader, val_loader) in clients_pbar:
            client_model = copy.deepcopy(global_model)
            trained_model, train_acc, val_acc = train_client(
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

            if trained_model is not None:
                client_models.append(client_model)
                client_train_accs.append(train_acc)
                client_val_accs.append(val_acc)

            if client_models:
                aggregated_dict = aggregate_models(client_models)
                global_model.load_state_dict(aggregated_dict)

                test_accuracy = evaluate_model(global_model, test_loader, device)
                avg_client_acc = sum(client_val_accs) / len(client_val_accs)

                metrics.update_global_metrics(round_idx, test_accuracy, avg_client_acc)

                if test_accuracy > best_accuracy:
                    best_accuracy = test_accuracy
                    torch.save(
                        global_model.state_dict(),
                        best_model_path,
                    )

                rounds_pbar.set_postfix(
                    {
                        "test_acc": f"{test_accuracy:.2f}%",
                        "best_acc": f"{best_accuracy:.2f}%",
                        "client_avg": f"{avg_client_acc:.2f}%",
                    }
                )

    metrics.writer.close()

    logging.info(
        f"\nFederated Learning completed. Best test accuracy: {best_accuracy:.2f}%"
    )
    return global_model


if __name__ == "__main__":
    # Run with different configurations
    configs = [
        {"num_clients": 3, "rounds": 5, "epochs": 30, "batch_size": 128, "lr": 0.01},
        # Add more configurations as needed
    ]

    for i, config in enumerate(configs):
        run_name = f"experiment_{i+1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        model = federated_learning(**config, run_name=run_name)
