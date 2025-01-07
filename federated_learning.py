import copy
import logging
import os
import shutil
import sys
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

from centralized_training import ImprovedTraining, ModernCNN, NewModernCNN


# Custom formatter for colored output
class ColoredFormatter(logging.Formatter):
    """Custom formatter with colored output."""

    COLORS = {
        "DEBUG": "\033[1;34m",  # Bold Blue
        "INFO": "\033[1;32m",  # Bold Green
        "WARNING": "\033[1;33m",  # Bold Yellow
        "ERROR": "\033[1;31m",  # Bold Red
        "CRITICAL": "\033[1;35m",  # Bold Magenta
        "RESET": "\033[1;0m",  # Reset
    }

    def format(self, record):
        # Add color to the level name
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = (
                f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"
            )
        return super().format(record)


class DuplicateFilter:
    """Filter that ensures each log message is only printed once"""

    def __init__(self):
        self.msgs = set()

    def filter(self, record):
        rv = record.msg not in self.msgs
        self.msgs.add(record.msg)
        return rv


def setup_logging(level=logging.DEBUG):
    """Configure logging with proper handlers and formatting."""
    # Clear any existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    # Set logging level
    root_logger.setLevel(level)

    # Create console handler with custom formatter
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(ColoredFormatter("[%(levelname)s] %(message)s"))

    # Add duplicate filter
    duplicate_filter = DuplicateFilter()
    console_handler.addFilter(duplicate_filter)

    root_logger.addHandler(console_handler)


# Constants
DEVICE_TYPE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = torch.device(DEVICE_TYPE)
MIN_WORKERS = 2


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


def get_appropriate_batch_size(dataset_size: int, num_clients: int) -> int:
    """Calculate appropriate batch size based on dataset size and number of clients."""
    samples_per_client = dataset_size // num_clients
    suggested_batch_size = max(
        1, min(64, samples_per_client // 10)
    )  # At least 10 batches per client
    return suggested_batch_size


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

    logging.info(f"Total dataset size: {total_size}")
    logging.info(f"Number of clients: {num_clients}")

    suggested_batch_size = min(
        batch_size, get_appropriate_batch_size(total_size, num_clients)
    )
    if batch_size != suggested_batch_size:
        logging.warning(
            f"Batch size {batch_size} is too large for dataset size {total_size}. "
            f"Using suggested batch size: {suggested_batch_size}"
        )
        batch_size = suggested_batch_size

    if iid:
        indices = torch.randperm(total_size)
        # Calculate sizes ensuring each client gets enough data
        client_size = total_size // num_clients
        logging.debug(f"Samples per client: {client_size}")

        if client_size < batch_size * 2:  # Ensure at least 2 batches per client
            raise ValueError(
                f"Not enough samples per client ({client_size}) for batch size {batch_size}. "
                f"Please reduce batch size or number of clients."
            )

        client_indices = []
        for i in range(num_clients):
            start_idx = i * client_size
            end_idx = start_idx + client_size if i < num_clients - 1 else len(indices)
            client_indices.append(indices[start_idx:end_idx])
            logging.debug(f"Client {i} dataset size: {len(client_indices[-1])}")
    else:
        # Non-IID distribution logic
        if not isinstance(dataset, torchvision.datasets.CIFAR100):
            raise ValueError("Non-IID distribution only supported for CIFAR100")

        labels = torch.tensor(dataset.targets, dtype=torch.long)  # Explicit dtype
        num_classes = 100
        classes_per_client = num_classes // num_clients
        logging.debug(f"Classes per client: {classes_per_client}")

        client_indices = [[] for _ in range(num_clients)]
        for class_idx in range(num_classes):
            class_indices = torch.where(labels == class_idx)[0]
            client_idx = class_idx // classes_per_client
            if client_idx < num_clients:
                client_indices[client_idx].extend(class_indices.tolist())

    dataloader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": True if num_workers > 0 else False,
        "prefetch_factor": 2 if num_workers > 0 else None,
    }

    client_loaders = []
    for client_idx, indices in enumerate(client_indices):
        # Convert indices to tensor properly
        indices = torch.as_tensor(indices, dtype=torch.long)

        # Split into train and validation
        train_size = int((1 - val_split) * len(indices))
        val_size = len(indices) - train_size

        perm_indices = indices[torch.randperm(len(indices))]
        train_indices = perm_indices[:train_size]
        val_indices = perm_indices[train_size:]

        # Create datasets
        train_dataset = torch.utils.data.Subset(dataset, train_indices.tolist())
        val_dataset = torch.utils.data.Subset(dataset, val_indices.tolist())

        if len(train_dataset) == 0:
            raise ValueError(f"Empty training dataset for client {client_idx}")
        if len(val_dataset) == 0:
            raise ValueError(f"Empty validation dataset for client {client_idx}")

        # Create dataloaders
        train_loader = DataLoader(
            train_dataset, shuffle=True, drop_last=True, **dataloader_kwargs
        )
        val_loader = DataLoader(val_dataset, shuffle=False, **dataloader_kwargs)

        client_loaders.append((train_loader, val_loader))

        logging.info(
            f"Client {client_idx} - Train size: {len(train_dataset)}, "
            f"Val size: {len(val_dataset)}, Batch size: {batch_size}"
        )

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
) -> Tuple[Optional[nn.Module], float, float]:
    """Train a client model using improved training techniques."""

    best_val_acc: float = 0.0
    train_acc: float = 0.0
    best_model: Optional[torch.nn.Module] = None
    # Initialize improved training
    trainer = ImprovedTraining(
        model=model, trainloader=train_loader, device=DEVICE, epochs=epochs, lr=lr
    )

    torch.backends.cudnn.benchmark = True

    epochs_pbar = tqdm(
        range(epochs),
        desc="Epoch Progress",
        unit="epoch",
        leave=False,
        position=2,
        colour="green",
        dynamic_ncols=True,
        mininterval=0.2,
    )

    for epoch in epochs_pbar:
        # Train one epoch with improved methods
        train_loss, train_acc = trainer.train_epoch(pbar_position=3)

        # Evaluate on validation set
        val_acc = evaluate_model(model, val_loader)

        # Update metrics
        metrics.update_client_metrics(
            client_id, round_idx * epochs + epoch, train_acc, val_acc
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = copy.deepcopy(model)

        # Update progress bar
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

    rounds_pbar = tqdm(
        range(rounds),
        desc="Federated Learning Progress",
        unit="round",
        leave=True,
        position=0,
        colour="blue",
        dynamic_ncols=True,
        mininterval=0.5,
    )

    for round_idx in rounds_pbar:
        client_models = []
        client_train_accs = []
        client_val_accs = []
        round_losses = []

        clients_pbar = tqdm(
            enumerate(client_loaders),
            total=len(client_loaders),
            desc="Client Progress",
            unit="client",
            leave=False,
            position=1,
            colour="cyan",
            dynamic_ncols=True,
            mininterval=0.3,
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
    tqdm.write(
        f"\nFederated Learning completed. Best test accuracy: {best_accuracy:.2f}%"
    )
    tqdm.write(f"Best model saved at: {best_model_path}")
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
    setup_logging(logging.INFO)

    archive_previous_runs()

    configs = [
        {
            "num_clients": 5,
            "rounds": 15,
            "epochs": 3,
            "batch_size": 128,
            "lr": 1e-3,
            "iid": True,
            "optimizer": "adabelief",
        },
    ]

    logging.debug(f"Using device: {DEVICE_TYPE}")

    for i, config in enumerate(configs):
        run_name = f"experiment_{i+1}_{'iid' if config['iid'] else 'non_iid'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        try:
            model = federated_learning(**config, run_name=run_name)
        except Exception as e:
            logging.error(f"Training failed: {str(e)}")
            continue
