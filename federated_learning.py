import copy
import logging
import os
import shutil
from datetime import datetime
from os import PathLike
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from adabelief_pytorch import AdaBelief
from lion_pytorch import Lion
from torch.amp import autocast_mode
from torch.nn.parameter import Parameter
from torch.optim.lr_scheduler import (
    ChainedScheduler,
    CosineAnnealingWarmRestarts,
    OneCycleLR,
    ReduceLROnPlateau,
)
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm.auto import tqdm

from centralized_training import ModernCNN, NewModernCNN

DEVICE_TYPE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = torch.device(DEVICE_TYPE)
logging.info(f"Using device: {DEVICE_TYPE}")

MIN_WORKERS = 2
logging.basicConfig(format="[%(levelname)s] %(message)s", level=logging.INFO)
logging.addLevelName(
    logging.INFO, "\033[1;32m%s\033[1;0m" % logging.getLevelName(logging.INFO)
)


class MetricsTracker:
    def __init__(
        self,
        config: dict,
        log_dir: Union[str, PathLike] = os.path.join("runs", "cnn"),
        run_name: Optional[str] = None,
    ):
        """Initialize simplified metrics tracker."""
        # Create descriptive run name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if run_name is None:
            run_name = f"FL_C{config['num_clients']}_R{config['rounds']}_{timestamp}"

        self.log_dir = os.path.join(log_dir, run_name)
        if os.path.exists(self.log_dir):
            shutil.rmtree(self.log_dir)

        self.writer = SummaryWriter(self.log_dir)
        self.best_accuracy = 0.0

        # Log experiment configuration
        self.writer.add_text(
            "Experiment Configuration",
            f"Clients: {config['num_clients']}\n"
            f"Rounds: {config['rounds']}\n"
            f"Epochs: {config['epochs']}\n"
            f"Batch Size: {config['batch_size']}\n"
            f"Learning Rate: {config['lr']}\n"
            f"Optimizer: {config['optimizer']}",
        )

    def update_client_metrics(
        self,
        client_id: int,
        round_idx: int,
        train_acc: float,
        val_acc: float,
    ) -> None:
        """Track per-client accuracy metrics."""
        self.writer.add_scalars(
            f"Client {client_id}/Accuracy",
            {
                "Training": train_acc,
                "Validation": val_acc,
            },
            round_idx,
        )

    def update_global_metrics(
        self,
        round_idx: int,
        test_acc: float,
        avg_client_acc: float,
        avg_loss: float,
    ) -> None:
        """Track global model metrics."""
        self.writer.add_scalars(
            "Global Model/Performance",
            {
                "Test Accuracy": test_acc,
                "Average Client Accuracy": avg_client_acc,
                "Loss": avg_loss,
            },
            round_idx,
        )

        if test_acc > self.best_accuracy:
            self.best_accuracy = test_acc
            self.writer.add_scalar("Global Model/Best Accuracy", test_acc, round_idx)

    def close(self) -> None:
        """Close the writer."""
        self.writer.close()


def create_client_dataloaders(
    dataset: torchvision.datasets.VisionDataset,
    num_clients: int,
    batch_size: int,
    val_split: float = 0.25,
    cpu_count: int = os.cpu_count() or 1,
    iid: bool = True,
) -> List[Tuple[DataLoader, DataLoader]]:
    """Split dataset into client dataloaders ensuring proper class distribution."""
    num_workers = min(MIN_WORKERS, cpu_count // num_clients)
    total_size = len(dataset)

    if iid:
        indices = torch.randperm(total_size)
        # Split indices into client chunks
        client_size = total_size // num_clients
        client_indices = [
            indices[i : i + client_size] for i in range(0, total_size, client_size)
        ]
    else:
        # Non-IID: Distribute classes to clients
        if not isinstance(dataset, torchvision.datasets.CIFAR100):
            raise ValueError("Non-IID distribution not supported for this dataset.")

        labels = torch.tensor(dataset.targets)
        num_classes = 100
        classes_per_client = num_classes // num_clients

        # Initialize client indices
        client_indices = [[] for _ in range(num_clients)]

        # Distribute classes to clients
        for class_idx in range(num_classes):
            # Get indices for current class
            class_indices = torch.where(labels == class_idx)[0]
            # Determine which client should get this class
            client_idx = class_idx // classes_per_client
            if client_idx < num_clients:  # Ensure we don't exceed number of clients
                client_indices[client_idx].extend(class_indices.tolist())

        # Balance the number of samples per client
        min_size = min(len(indices) for indices in client_indices)
        client_indices = [
            torch.tensor(indices[:min_size]) for indices in client_indices
        ]

    dataloader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": True,
        "prefetch_factor": 2,
        "generator": torch.Generator().manual_seed(42),
    }

    client_loaders = []
    for client_idx, indices in enumerate(client_indices):
        # Split into train and validation maintaining class distribution
        dataset_size = len(indices)
        val_size = int(dataset_size * val_split)
        train_size = dataset_size - val_size

        # Shuffle indices before splitting
        perm = torch.randperm(dataset_size)
        indices = indices[perm]

        train_indices = indices[:train_size]
        val_indices = indices[train_size:]

        # Verify data distribution
        if not iid:
            train_labels = [dataset[int(i)][1] for i in train_indices]
            val_labels = [dataset[int(i)][1] for i in val_indices]
            unique_train = len(set(train_labels))
            unique_val = len(set(val_labels))
            assert (
                unique_train > 0
            ), f"Client {client_idx} has no classes in training set"
            assert (
                unique_val > 0
            ), f"Client {client_idx} has no classes in validation set"

        # Create datasets
        train_dataset = torch.utils.data.Subset(dataset, train_indices.tolist())
        val_dataset = torch.utils.data.Subset(dataset, val_indices.tolist())

        # Create dataloaders
        train_loader = DataLoader(
            train_dataset, shuffle=True, drop_last=True, **dataloader_kwargs
        )
        val_loader = DataLoader(val_dataset, shuffle=False, **dataloader_kwargs)

        client_loaders.append((train_loader, val_loader))

    return client_loaders


def get_optimizer(
    name: str, params: Iterator[Parameter], lr: float
) -> torch.optim.Optimizer:
    """Get optimizer by name with recommended settings."""
    name = name.lower()
    if name == "adabelief":
        # https://github.com/juntang-zhuang/Adabelief-Optimizer?tab=readme-ov-file#table-of-hyper-parameters
        return AdaBelief(
            params,
            lr=lr,
            eps=1e-8,
            betas=(0.9, 0.999),
            weight_decouple=False,
            rectify=False,
            fixed_decay=False,
            amsgrad=False,
            print_change_log=False,
        )
    elif name == "radam":
        # Rectified Adam
        return optim.RAdam(
            params, lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4
        )
    elif name == "adafactor":
        # Memory efficient optimizer
        return optim.Adafactor(
            params,
            lr=lr,
            # beta1=0.9,
            eps=(1e-30, 1e-3),
            # clip_threshold=1.0,
            beta2_decay=-0.8,
            weight_decay=1e-4,
        )
    elif name == "lion":
        # Google's Lion optimizer
        return Lion(params, lr=lr, betas=(0.9, 0.99), weight_decay=1e-4)
    elif name == "adamw":
        # Default AdamW
        return optim.AdamW(params, lr=lr, weight_decay=1e-4)
    else:
        raise ValueError(f"Unsupported optimizer: {name}")


def get_scheduler(
    name: str, optimizer: torch.optim.Optimizer, **kwargs
) -> OneCycleLR | CosineAnnealingWarmRestarts | ReduceLROnPlateau | ChainedScheduler:
    """Get learning rate scheduler by name."""
    name = name.lower()
    if name == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=kwargs.get("max_lr", 0.1),
            epochs=kwargs.get("epochs", 10),
            steps_per_epoch=kwargs.get("steps_per_epoch", 100),
            pct_start=0.3,
            div_factor=25.0,
        )
    elif name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=kwargs.get("t_0", 10),
            T_mult=kwargs.get("t_mult", 2),
            eta_min=1e-6,
        )
    elif name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
        )
    elif name == "warmup_cosine":
        return torch.optim.lr_scheduler.ChainedScheduler(
            [
                torch.optim.lr_scheduler.LinearLR(
                    optimizer, start_factor=0.01, end_factor=1.0, total_iters=5
                ),
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=kwargs.get("epochs", 10) - 5, eta_min=1e-6
                ),
            ]
        )
    else:
        raise ValueError(f"Unsupported scheduler: {name}")


def train_client(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    client_id: int,
    epochs: int,
    lr: float,
    metrics: MetricsTracker,
    round_idx: int,
    optimizer_name: str = "adamw",
    scheduler_name: str = "plateau",
) -> Tuple[Optional[nn.Module], float, float]:
    """Train a client model and return metrics."""
    criterion = nn.CrossEntropyLoss()
    optimizer: optim.Optimizer = get_optimizer(optimizer_name, model.parameters(), lr)
    scheduler = get_scheduler(scheduler_name, optimizer, epochs=epochs)
    assert isinstance(scheduler, ReduceLROnPlateau)
    best_val_acc: float = 0.0
    best_model: Optional[torch.nn.Module] = None
    train_acc = 0.0

    torch.backends.cudnn.benchmark = True

    epochs_pbar = tqdm(
        range(epochs),
        desc=f"Training Client {client_id+1} Epochs",
        position=2,
        leave=False,
    )
    for epoch in epochs_pbar:
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        batch_pbar = tqdm(
            train_loader, desc=f"Epoch {epoch+1}", position=3, leave=False
        )
        for inputs, labels in batch_pbar:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

            batch_pbar.set_postfix(
                {"loss": f"{loss.item():.3f}", "acc": f"{100.*correct/total:.2f}%"}
            )


        train_loss = running_loss / len(train_loader)
        train_acc = 100.0 * correct / total

        val_acc = evaluate_model(model, val_loader)
        global_step = round_idx * epochs + epoch
        metrics.update_client_metrics(client_id, global_step, train_acc, val_acc)

        scheduler.step(metrics=val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = copy.deepcopy(model)

        epochs_pbar.set_postfix(
            {
                "train_loss": f"{train_loss:.3f}",
                "train_acc": f"{train_acc:.2f}%",
                "val_acc": f"{val_acc:.2f}%",
            }
        )

    return best_model, train_acc, best_val_acc


def evaluate_model(model: nn.Module, dataloader: DataLoader) -> float:
    """Evaluate model accuracy."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

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
    iid: bool = True,
    optimizer: str = "adamw",
) -> nn.Module:
    """Run federated learning simulation with enhanced metrics tracking."""

    timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")
    if run_name is None:
        config_str = f"c{num_clients}_r{rounds}_e{epochs}_b{batch_size}_lr{lr}"
        run_name = f"{config_str}_{timestamp}"

    config: dict[str, int | float | str] = {
        "num_clients": num_clients,
        "rounds": rounds,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "optimizer": optimizer,
    }

    metrics = MetricsTracker(config, run_name=run_name)

    Path(save_dir).mkdir(exist_ok=True)
    best_model_path = os.path.join(save_dir, f"best_model_{timestamp}.pth")

    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]
            ),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ]
    )

    train_dataset = torchvision.datasets.CIFAR100(
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

    client_loaders = create_client_dataloaders(
        train_dataset, num_clients, batch_size, iid=iid
    )

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
    # TODO: choose the model to use
    # global_model = ModernCNN(num_classes=100).to(DEVICE)
    global_model = NewModernCNN(num_classes=100).to(DEVICE)
    best_accuracy = 0.0

    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    rounds_pbar = tqdm(range(rounds), desc="Rounds", leave=True, position=0)
    for round_idx in rounds_pbar:
        client_models = []
        client_train_accs = []
        client_val_accs = []
        round_losses = []

        clients_pbar = tqdm(
            enumerate(client_loaders),
            total=len(client_loaders),
            desc="Training Clients",
            leave=False,
            position=1,
        )

        for client_idx, (train_loader, val_loader) in clients_pbar:
            client_model = copy.deepcopy(global_model)
            trained_model, train_acc, val_acc = train_client(
                client_model,
                train_loader,
                val_loader,
                client_idx,
                epochs,
                lr,
                metrics,
                round_idx,
                optimizer_name=optimizer,
            )

            if trained_model is not None:
                client_models.append(client_model)
                client_train_accs.append(train_acc)
                client_val_accs.append(val_acc)

                # Calculate loss for global metrics
                criterion = nn.CrossEntropyLoss()
                client_model.eval()
                total_loss = 0.0
                n_batches = 0

                with torch.no_grad():
                    for inputs, labels in val_loader:
                        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                        outputs = client_model(inputs)
                        loss = criterion(outputs, labels)
                        total_loss += loss.item()
                        n_batches += 1

                avg_loss = total_loss / n_batches
                round_losses.append(avg_loss)

                clients_pbar.set_postfix(
                    {
                        "train_acc": f"{train_acc:.2f}%",
                        "val_acc": f"{val_acc:.2f}%",
                        "loss": f"{avg_loss:.3f}",
                    }
                )

            if client_models:
                with torch.no_grad():
                    aggregated_dict = aggregate_models(client_models)
                    global_model.load_state_dict(aggregated_dict)

                    test_accuracy = evaluate_model(global_model, test_loader)
                    avg_client_acc = sum(client_val_accs) / len(client_val_accs)
                    avg_loss = sum(round_losses) / len(round_losses)

                    metrics.update_global_metrics(
                        round_idx=round_idx,
                        test_acc=test_accuracy,
                        avg_client_acc=avg_client_acc,
                        avg_loss=avg_loss,
                    )

                    # Save best model when accuracy improves
                    if test_accuracy > best_accuracy:
                        best_accuracy = test_accuracy
                        torch.save(global_model.state_dict(), best_model_path)
                        # logging.info(f"New best model saved with accuracy: {best_accuracy:.2f}%")

                rounds_pbar.set_postfix(
                    {
                        "test_acc": f"{test_accuracy:.2f}%",
                        "best_acc": f"{best_accuracy:.2f}%",
                        "avg_client_acc": f"{avg_client_acc:.2f}%",
                        "loss": f"{avg_loss:.3f}",
                    }
                )

    # Save final model if different from best
    final_model_path = os.path.join(save_dir, f"final_model_{timestamp}.pth")
    torch.save(global_model.state_dict(), final_model_path)

    metrics.close()
    logging.info(
        f"\nFederated Learning completed. Best test accuracy: {best_accuracy:.2f}%"
    )
    logging.info(f"Best model saved at: {best_model_path}")
    return global_model


def archive_previous_runs(base_dir: Union[str, PathLike] = "runs") -> None:
    """Archive existing runs to a timestamped folder with proper error handling."""
    try:
        if os.path.exists(base_dir) and os.listdir(base_dir):
            # Create timestamp and archive directory path
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            archive_dir = os.path.join("old_runs", f"archived_{timestamp}")
            os.makedirs("old_runs", exist_ok=True)

            # Create a new directory for archiving
            os.makedirs(archive_dir, exist_ok=True)

            # Copy files instead of moving to avoid permission issues
            for item in os.listdir(base_dir):
                source = os.path.join(base_dir, item)
                dest = os.path.join(archive_dir, item)
                try:
                    if os.path.isdir(source):
                        shutil.copytree(source, dest)
                    else:
                        shutil.copy2(source, dest)
                except (shutil.Error, PermissionError) as e:
                    logging.warning(f"Error copying {item}: {e}")

            # Remove original directory after successful copy
            try:
                shutil.rmtree(base_dir)
            except (PermissionError, OSError) as e:
                logging.warning(f"Could not remove original directory: {e}")

            logging.info(f"Previous runs archived to: {archive_dir}")

        # Create fresh runs directory
        os.makedirs(base_dir, exist_ok=True)

    except Exception as e:
        logging.error(f"Error during archiving: {e}")
        # Ensure runs directory exists even if archiving fails
        os.makedirs(base_dir, exist_ok=True)


if __name__ == "__main__":
    archive_previous_runs()

    configs = [
        # {
        #     "num_clients": 2,
        #     "rounds": 3,
        #     "epochs": 10,
        #     "batch_size": 64,
        #     "lr": 0.001,
        #     "iid": True,
        #     "optimizer": "adamw",
        # },
        {
            "num_clients": 5,
            "rounds": 3,
            "epochs": 15,
            "batch_size": 64,
            "lr": 1e-3,
            "iid": True,
            "optimizer": "adabelief",
        },
    ]

    for i, config in enumerate(configs):
        run_name = f"experiment_{i+1}_{'iid' if config['iid'] else 'non_iid'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        model = federated_learning(**config, run_name=run_name)
