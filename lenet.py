"""
# AML Project 5 - Federated Learning - Track B

## Setup & Installation

### Downloading and importing Library
"""

import copy
import dataclasses
import json
import logging
import math
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler, StepLR
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision.datasets.cifar import CIFAR100
from tqdm import tqdm

"""
### Logging Setup
"""


class TqdmLoggingHandler(logging.Handler):
    def emit(self, record) -> None:
        try:
            msg = self.format(record)
            tqdm.write("\r\033[K" + msg)
            self.flush()
        except Exception:
            self.handleError(record)


class ColoredFormatter(logging.Formatter):
    COLORS = {
        "DEBUG": "\033[1;34m",
        "INFO": "\033[1;32m",
        "WARNING": "\033[1;33m",
        "ERROR": "\033[1;31m",
        "CRITICAL": "\033[1;35m",
        "RESET": "\033[0m",
    }

    def format(self, record):
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = (
                f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"
            )
        return super().format(record)


def setup_logging(level=logging.INFO):
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(level)

    tqdm_handler = TqdmLoggingHandler()
    formatter = ColoredFormatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    tqdm_handler.setFormatter(formatter)
    root_logger.addHandler(tqdm_handler)


setup_logging()

"""
## Config Setup
"""


@dataclass
class ExperimentSettings:
    """Settings for heterogeneity experiments."""

    # Core parameters
    K: int = 100  # Number of clients
    C: float = 0.1  # Client fraction

    # Non-IID configurations
    Nc: Tuple[Optional[int], ...] = (None, 1, 5, 10, 50)  # None = IID

    # Local steps configurations (J) with scaled rounds
    J_configs: Dict[int, int] = field(
        default_factory=lambda: {
            4: 2000,  # Base configuration
            8: 1000,  # Halved rounds
            16: 500,  # Quarter rounds
        }
    )

    def get_rounds(self, local_epochs: int) -> int:
        """Scale rounds based on local epochs."""
        base_j = min(self.J_configs.keys())
        base_rounds = self.J_configs[base_j]
        return int(base_rounds * (base_j / local_epochs))


@dataclass(frozen=True)
class BaseConfig:
    DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CPU_COUNT: int = os.cpu_count() or 1
    NUM_WORKERS: int = min(0, CPU_COUNT)
    PERSISTENT_WORKERS: bool = False
    PIN_MEMORY: bool = False
    PREFETCH_FACTOR: Optional[int] = None

    # Memory settings
    VIRTUAL_MEMORY_SIZE_MB: int = 16 * 1024 * 1024  # 16GB

    # Random seed
    SEED: int = 42

    # Paths
    ROOT_DIR: Path = Path.cwd()
    CONFIGS_DIR: Path = ROOT_DIR / "configs"
    DATA_DIR: Path = ROOT_DIR / "data"
    MODELS_DIR: Path = ROOT_DIR / "models"
    RESULTS_DIR: Path = ROOT_DIR / "results"
    RUNS_DIR: Path = ROOT_DIR / "runs"
    OLD_RUNS_DIR: Path = RUNS_DIR / "old_runs"

    # Training Parameters
    BATCH_SIZE: int = 64
    LEARNING_RATE: float = 0.01
    NUM_EPOCHS: int = 200
    MOMENTUM: float = 0.9
    WEIGHT_DECAY: float = 4e-4
    NUM_CLASSES: int = 100

    def serialize(self) -> dict:
        """Serialize essential config parameters."""
        return {
            "batch_size": self.BATCH_SIZE,
            "learning_rate": self.LEARNING_RATE,
            "num_epochs": self.NUM_EPOCHS,
            "momentum": self.MOMENTUM,
            "weight_decay": self.WEIGHT_DECAY,
            "num_classes": self.NUM_CLASSES,
        }

    @classmethod
    def deserialize(cls, data: dict) -> "BaseConfig":
        """Create config from serialized data."""
        return cls(
            BATCH_SIZE=data["batch_size"],
            LEARNING_RATE=data["learning_rate"],
            NUM_EPOCHS=data["num_epochs"],
            MOMENTUM=data["momentum"],
            WEIGHT_DECAY=data["weight_decay"],
            NUM_CLASSES=data["num_classes"],
        )

    def matches(self, other: Union[dict, "BaseConfig"]) -> bool:
        """Check if config matches current config."""
        if isinstance(other, dict):
            return (
                other.get("batch_size") == self.BATCH_SIZE
                and other.get("num_classes") == self.NUM_CLASSES
                and other.get("learning_rate") == self.LEARNING_RATE
            )
        else:
            return (
                other.BATCH_SIZE == self.BATCH_SIZE
                and other.NUM_CLASSES == self.NUM_CLASSES
                and other.LEARNING_RATE == self.LEARNING_RATE
            )


# Create directories
config = BaseConfig()
for dir_path in [
    config.DATA_DIR,
    config.MODELS_DIR,
    config.RESULTS_DIR,
    config.CONFIGS_DIR,
    config.RUNS_DIR,
    config.OLD_RUNS_DIR,
]:
    dir_path.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class FederatedConfig(BaseConfig):
    """Federated Learning specific configuration."""

    TWO_PHASE: bool = False
    NUM_CLIENTS: int = 100  # K
    PARTICIPATION_RATE: float = 0.1  # C
    LOCAL_EPOCHS: int = 4  # J
    NUM_ROUNDS: int = 2000
    CLASSES_PER_CLIENT: Optional[int] = None  # None for IID
    PARTICIPATION_MODE: str = "uniform"
    DIRICHLET_ALPHA: Optional[float] = None

    def __init__(self, *args, **kwargs):
        base_params = {
            k: v
            for k, v in kwargs.items()
            if k
            in [
                "BATCH_SIZE",
                "LEARNING_RATE",
                "NUM_EPOCHS",
                "MOMENTUM",
                "WEIGHT_DECAY",
                "NUM_CLASSES",
            ]
        }
        super().__init__(*args, **base_params)

    def get_epochs_for_training(self) -> int:
        """Get actual number of epochs to use in training."""
        return max(1, self.LOCAL_EPOCHS // 2) if self.TWO_PHASE else self.LOCAL_EPOCHS

    def matches(self, other: Union[dict, "FederatedConfig"]) -> bool:  # type: ignore
        """Check if config matches with detailed logging."""

        def log_mismatch(field: str, val1: Any, val2: Any) -> None:
            logging.warning(
                f"Config mismatch in {field}: {val1} != {val2} (saved != current)"
            )

        if isinstance(other, dict):
            # Base config comparison
            if not super().matches(other):
                logging.warning("Base config mismatch")
                return False

            # Compare federated parameters
            comparisons = [
                ("num_clients", other.get("num_clients"), self.NUM_CLIENTS),
                (
                    "participation_rate",
                    other.get("participation_rate"),
                    self.PARTICIPATION_RATE,
                ),
                ("local_epochs", other.get("local_epochs"), self.LOCAL_EPOCHS),
                (
                    "classes_per_client",
                    other.get("classes_per_client"),
                    self.CLASSES_PER_CLIENT,
                ),
                (
                    "participation_mode",
                    other.get("participation_mode"),
                    self.PARTICIPATION_MODE,
                ),
                ("dirichlet_alpha", other.get("dirichlet_alpha"), self.DIRICHLET_ALPHA),
                ("two_phase", other.get("two_phase", False), self.TWO_PHASE),
            ]

            for field, saved_val, current_val in comparisons:
                if saved_val != current_val:
                    log_mismatch(field, saved_val, current_val)
                    return False
            return True

        else:
            # Base config comparison
            if not super().matches(other):
                logging.debug("Base config mismatch")
                return False

            # Compare federated parameters
            comparisons = [
                ("num_clients", other.NUM_CLIENTS, self.NUM_CLIENTS),
                (
                    "participation_rate",
                    other.PARTICIPATION_RATE,
                    self.PARTICIPATION_RATE,
                ),
                ("local_epochs", other.LOCAL_EPOCHS, self.LOCAL_EPOCHS),
                (
                    "classes_per_client",
                    other.CLASSES_PER_CLIENT,
                    self.CLASSES_PER_CLIENT,
                ),
                (
                    "participation_mode",
                    other.PARTICIPATION_MODE,
                    self.PARTICIPATION_MODE,
                ),
                ("dirichlet_alpha", other.DIRICHLET_ALPHA, self.DIRICHLET_ALPHA),
                ("two_phase", other.TWO_PHASE, self.TWO_PHASE),
            ]

            for field, saved_val, current_val in comparisons:
                if saved_val != current_val:
                    log_mismatch(field, saved_val, current_val)
                    return False
            return True

    def serialize(self) -> dict:
        """Serialize with validation."""
        data = super().serialize()
        data.update(
            {
                "num_clients": self.NUM_CLIENTS,
                "participation_rate": self.PARTICIPATION_RATE,
                "local_epochs": self.LOCAL_EPOCHS,
                "num_rounds": self.NUM_ROUNDS,
                "classes_per_client": self.CLASSES_PER_CLIENT,
                "participation_mode": self.PARTICIPATION_MODE,
                "dirichlet_alpha": self.DIRICHLET_ALPHA,
                "two_phase": self.TWO_PHASE,
            }
        )
        return data

    @classmethod
    def deserialize(cls, data: dict) -> "FederatedConfig":
        """Deserialize with validation."""
        # Ensure all required fields are present
        required_fields = {
            "num_clients",
            "participation_rate",
            "local_epochs",
            "num_rounds",
            "classes_per_client",
            "participation_mode",
            "two_phase",
        }

        missing = required_fields - set(data.keys())
        if missing:
            logging.warning(f"Missing required fields in config: {missing}")
            # Set defaults for backward compatibility
            for field in missing:
                data[field] = getattr(cls, field.upper())

        return cls(
            BATCH_SIZE=data.get("batch_size", cls.BATCH_SIZE),
            LEARNING_RATE=data.get("learning_rate", cls.LEARNING_RATE),
            NUM_EPOCHS=data.get("num_epochs", cls.NUM_EPOCHS),
            MOMENTUM=data.get("momentum", cls.MOMENTUM),
            WEIGHT_DECAY=data.get("weight_decay", cls.WEIGHT_DECAY),
            NUM_CLASSES=data.get("num_classes", cls.NUM_CLASSES),
            NUM_CLIENTS=data.get("num_clients", cls.NUM_CLIENTS),
            PARTICIPATION_RATE=data.get("participation_rate", cls.PARTICIPATION_RATE),
            LOCAL_EPOCHS=data.get("local_epochs", cls.LOCAL_EPOCHS),
            NUM_ROUNDS=data.get("num_rounds", cls.NUM_ROUNDS),
            CLASSES_PER_CLIENT=data.get("classes_per_client", cls.CLASSES_PER_CLIENT),
            PARTICIPATION_MODE=data.get("participation_mode", cls.PARTICIPATION_MODE),
            DIRICHLET_ALPHA=data.get("dirichlet_alpha", cls.DIRICHLET_ALPHA),
            TWO_PHASE=data.get("two_phase", cls.TWO_PHASE),
        )


"""
## Model
"""


class LeNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.connected = nn.Sequential(
            nn.Linear(5 * 5 * 64, 384),
            nn.ReLU(),
            nn.Linear(384, 192),
            nn.ReLU(),
        )

        self.classifier = nn.Linear(192, config.NUM_CLASSES)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = x.view(x.size(0), -1)
        x = self.connected(x)
        x = self.classifier(x)
        return x


"""
## Metrics Logger

This class is used to save the model training in a way to be analyzed using tensorboard 
"""


class MetricsManager:
    """Enhanced metrics manager for experiment tracking, visualization and comparison."""

    def __init__(
        self,
        config: BaseConfig,
        model_name: str,
        training_type: Literal["centralized", "federated"],
        experiment_name: Optional[str] = None,
    ):
        self.config = config
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

        # Setup experiment directories
        self.experiment_group = training_type
        if experiment_name:
            self.experiment_group = f"{training_type}_{experiment_name.split('_')[0]}"

        self.experiment_dir = config.RUNS_DIR / self.experiment_group
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        # Setup run name and paths
        run_suffix = f"{experiment_name}_{timestamp}" if experiment_name else timestamp
        self.run_name = f"{model_name}_{run_suffix}"
        self.run_dir = self.experiment_dir / self.run_name

        # Setup metrics storage
        self.metrics_dir = self.run_dir / "metrics"
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        self.train_file = self.metrics_dir / "train_metrics.csv"
        self.val_file = self.metrics_dir / "val_metrics.csv"
        self.test_file = self.metrics_dir / "test_metrics.csv"

        # Initialize CSV files
        for file in [self.train_file, self.val_file, self.test_file]:
            if not file.exists():
                with open(file, "w", buffering=1) as f:
                    f.write("step,loss,accuracy\n")
                    f.flush()

        # Setup TensorBoard
        self.writer = SummaryWriter(self.run_dir)

        # In-memory metrics storage
        self.metrics = {
            "train": {"loss": [], "accuracy": [], "steps": []},
            "validation": {"loss": [], "accuracy": [], "steps": []},
            "test": {"loss": [], "accuracy": [], "steps": []},
        }

        # Store configuration
        self.save_config()

    def save_config(self) -> None:
        """Save experiment configuration."""
        config_dict = {
            "experiment_group": self.experiment_group,
            "run_name": self.run_name,
            "timestamp": datetime.now().isoformat(),
            "config": self.config.serialize(),
        }

        with open(self.run_dir / "config.json", "w") as f:
            json.dump(config_dict, f, indent=2)

    def save_metrics(self) -> None:
        """Safely save all metrics to disk."""
        try:
            # Ensure TensorBoard writes are flushed
            self.writer.flush()

            # Function to safely write metrics to CSV
            def write_metrics_to_csv(metrics_data: dict, file_path: Path) -> None:
                if not metrics_data["steps"]:
                    return

                df = pd.DataFrame(
                    {
                        "step": metrics_data["steps"],
                        "loss": metrics_data["loss"],
                        "accuracy": metrics_data["accuracy"],
                    }
                )

                # Write with proper index handling
                df.to_csv(file_path, index=False)

            # Save each split's metrics
            for split, file_path in [
                ("train", self.train_file),
                ("validation", self.val_file),
                ("test", self.test_file),
            ]:
                if self.metrics[split]["steps"]:  # Only save if we have data
                    write_metrics_to_csv(self.metrics[split], file_path)

        except Exception as e:
            logging.error(f"Error saving metrics: {e}")
            raise

    def log_metrics(
        self,
        split: Literal["train", "validation", "test"],
        loss: float,
        accuracy: float,
        step: int,
    ) -> None:
        """Log metrics for specified split."""
        # TensorBoard logging
        if not isinstance(loss, (int, float)) or not isinstance(accuracy, (int, float)):
            logging.error(f"Invalid metrics: loss={loss}, accuracy={accuracy}")
            return
        if math.isnan(loss) or math.isnan(accuracy):
            logging.error(f"NaN detected: loss={loss}, accuracy={accuracy}")
            return

        self.writer.add_scalars("metrics/loss", {split: loss}, step)
        self.writer.add_scalars("metrics/accuracy", {split: accuracy}, step)

        # Store in memory
        self.metrics[split]["loss"].append(loss)
        self.metrics[split]["accuracy"].append(accuracy)
        self.metrics[split]["steps"].append(step)

        # Save metrics to disk every 10 steps
        if len(self.metrics[split]["steps"]) % 10 == 0:
            self.save_metrics()

    def log_fl_metrics(
        self,
        round_idx: int,
        metrics: Dict[str, float],
        client_stats: Optional[Dict[str, int | float]] = None,
    ) -> None:
        """Log federated learning specific metrics."""
        val_loss = metrics.get("val_loss")
        val_accuracy = metrics.get("val_accuracy")
        test_loss = metrics.get("test_loss")
        test_accuracy = metrics.get("test_accuracy")

        if val_loss is not None and val_accuracy is not None:
            self.log_metrics("validation", val_loss, val_accuracy, round_idx)

        if test_loss is not None and test_accuracy is not None:
            self.log_metrics("test", test_loss, test_accuracy, round_idx)

        if client_stats:
            self.writer.add_scalars("federated/client_stats", client_stats, round_idx)

    def plot_learning_curves(self) -> None:
        """Generate learning curves plot."""
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        for split in ["train", "validation", "test"]:
            if len(self.metrics[split]["steps"]) > 0:
                ax1.plot(
                    self.metrics[split]["steps"],
                    self.metrics[split]["loss"],
                    label=f"{split} loss",
                )
                ax2.plot(
                    self.metrics[split]["steps"],
                    self.metrics[split]["accuracy"],
                    label=f"{split} accuracy",
                )

        ax1.set_title("Loss Curves")
        ax1.set_xlabel("Steps")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.grid(True)

        ax2.set_title("Accuracy Curves")
        ax2.set_xlabel("Steps")
        ax2.set_ylabel("Accuracy (%)")
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig(self.metrics_dir / "learning_curves.png", dpi=300)
        plt.close()

    def save_summary(self) -> None:
        """Save experiment summary statistics."""
        summary = {
            "experiment_group": self.experiment_group,
            "run_name": self.run_name,
            "final_metrics": {},
        }

        for split in ["train", "validation", "test"]:
            if len(self.metrics[split]["loss"]) > 0:
                summary["final_metrics"][split] = {
                    "final_loss": self.metrics[split]["loss"][-1],
                    "final_accuracy": self.metrics[split]["accuracy"][-1],
                    "best_accuracy": max(self.metrics[split]["accuracy"]),
                    "best_loss": min(self.metrics[split]["loss"]),
                }

        with open(self.metrics_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    @staticmethod
    def load_config(config_path: Path) -> Tuple[str, str, BaseConfig]:
        """Load experiment configuration."""
        with open(config_path) as f:
            config_dict = json.load(f)

        # Determine config type and deserialize
        config_data = config_dict["config"]
        if "num_clients" in config_data:
            config = FederatedConfig.deserialize(config_data)
        else:
            config = BaseConfig.deserialize(config_data)

        return config_dict["experiment_group"], config_dict["run_name"], config

    @staticmethod
    def compare_runs(base_dir: Path, experiment_group: str) -> None:
        """Compare multiple runs within an experiment group."""
        import matplotlib.pyplot as plt
        import pandas as pd

        exp_dir = base_dir / experiment_group
        if not exp_dir.exists():
            raise ValueError(f"No experiments found for group {experiment_group}")

        # Collect all run data
        summaries = []
        for run_dir in exp_dir.glob("*"):
            if not run_dir.is_dir():
                continue

            config_file = run_dir / "config.json"
            summary_file = run_dir / "metrics" / "summary.json"
            if summary_file.exists():
                with open(summary_file) as f:
                    summary = json.load(f)
                    if summaries:
                        # get only the config
                        summary["config"] = MetricsManager.load_config(config_file)[2]

                    summaries.append(summary)

        if not summaries:
            raise ValueError("No run data found")

        # Create comparison DataFrame
        comparison_data = []
        for summary in summaries:
            row = {"run": summary["run_name"]}
            for split in ["train", "validation", "test"]:
                if split in summary["final_metrics"]:
                    metrics = summary["final_metrics"][split]
                    row.update(
                        {
                            f"{split}_final_loss": metrics["final_loss"],
                            f"{split}_final_acc": metrics["final_accuracy"],
                            f"{split}_best_acc": metrics["best_accuracy"],
                        }
                    )
            comparison_data.append(row)

        df = pd.DataFrame(comparison_data)

        # Save comparison results
        results_dir = exp_dir / "comparisons"
        results_dir.mkdir(exist_ok=True)

        df.to_csv(results_dir / "comparison.csv", index=False)
        df.to_latex(
            results_dir / "comparison.tex",
            float_format="%.2f",
            index=False,
            caption=f"Comparison of {experiment_group} experiments",
            label=f"tab:{experiment_group}_comparison",
        )

        # Plot comparison
        plt.figure(figsize=(10, 6))
        plt.bar(df["run"], df["test_best_acc"])
        plt.xticks(rotation=45, ha="right")
        plt.title(f"{experiment_group} - Test Accuracy Comparison")
        plt.ylabel("Best Test Accuracy (%)")
        plt.tight_layout()
        plt.savefig(results_dir / "accuracy_comparison.png", dpi=300)
        plt.close()

    def close(self) -> None:
        """Close writer and save final artifacts."""
        try:
            # Save final metrics
            self.save_metrics()
            # Generate plots and summaries
            self.plot_learning_curves()
            self.save_summary()
        finally:
            self.writer.close()


"""
## Dataset Manager

Here the CIFAR100 is downloaded and the train, validation and test split are constructed to be used later
"""


class Cifar100DatasetManager:
    config: BaseConfig
    validation_split: float
    train_transform: transforms.Compose
    test_transform: transforms.Compose
    train_loader: DataLoader[CIFAR100]
    val_loader: DataLoader[CIFAR100]
    test_loader: DataLoader[CIFAR100]

    def __init__(self, config: BaseConfig, validation_split: float = 0.1) -> None:
        self.config = config
        self.validation_split = validation_split

        self.train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]
                ),
            ]
        )

        self.test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]
                ),
            ]
        )

        self.train_loader, self.val_loader, self.test_loader = self._prepare_data()

    def _prepare_data(
        self,
    ) -> Tuple[DataLoader[CIFAR100], DataLoader[CIFAR100], DataLoader[CIFAR100]]:
        full_trainset: CIFAR100 = CIFAR100(
            root=self.config.DATA_DIR,
            train=True,
            download=True,
            transform=self.train_transform,
        )

        train_size: int = int((1 - self.validation_split) * len(full_trainset))
        val_size: int = len(full_trainset) - train_size

        train_dataset, val_dataset = random_split(
            full_trainset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(self.config.SEED),
        )

        test_dataset: CIFAR100 = CIFAR100(
            root=self.config.DATA_DIR,
            train=False,
            download=False,
            transform=self.test_transform,
        )

        loader_kwargs = {
            "num_workers": self.config.NUM_WORKERS,
            "persistent_workers": self.config.PERSISTENT_WORKERS,
            "prefetch_factor": self.config.PREFETCH_FACTOR,
            "pin_memory": self.config.PIN_MEMORY,
        }

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            drop_last=True,
            **loader_kwargs,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            **loader_kwargs,
        )

        test_loader: DataLoader[CIFAR100] = DataLoader(
            test_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            **loader_kwargs,
        )

        return train_loader, val_loader, test_loader

    @property
    def train_dataset(self) -> Dataset[CIFAR100]:
        return self.train_loader.dataset

    @property
    def val_dataset(self) -> Dataset[CIFAR100]:
        return self.val_loader.dataset

    @property
    def test_dataset(self) -> Dataset[CIFAR100]:
        return self.test_loader.dataset


"""
## Centralized Trainer

This class is responsible to train and evaluate the model (here only considered for typing the LeNet model) in the traditional sense.
Local training with normal train and evaluate methods
"""


class CentralizedTrainer:
    model: LeNet
    config: BaseConfig
    device: torch.device
    metrics: MetricsManager

    def __init__(
        self, model: LeNet, config: BaseConfig, experiment_name: str = "baseline"
    ) -> None:
        self.model = model.to(config.DEVICE)
        self.config = config
        self.device = config.DEVICE
        self.device_type = str(config.DEVICE)
        self.metrics = MetricsManager(
            config=config,
            model_name=model.__class__.__name__.lower(),
            training_type="centralized",
            experiment_name=experiment_name,
        )
        self.checkpoint_dir = config.MODELS_DIR / "centralized"
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.checkpoint_path = self.checkpoint_dir / "centralized_baseline.pt"

    def save_checkpoint(
        self, epoch: int, best_val_loss: float, best_val_acc: float
    ) -> None:
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "best_val_loss": best_val_loss,
            "best_val_acc": best_val_acc,
            "config": self.config.serialize(),
        }
        torch.save(checkpoint, self.checkpoint_path)
        logging.info(f"Checkpoint saved: {self.checkpoint_path}")

    def load_checkpoint(self) -> Tuple[int, float, float]:
        if not self.checkpoint_path.exists():
            return 0, float("inf"), 0.0

        checkpoint = torch.load(self.checkpoint_path)

        # Validate using matches method
        if not self.config.matches(checkpoint["config"]):
            logging.warning("Config mismatch in checkpoint, starting fresh training")
            return 0, float("inf"), 0.0

        self.model.load_state_dict(checkpoint["model_state_dict"])
        logging.info(f"Resumed from checkpoint: {self.checkpoint_path}")
        return (
            checkpoint["epoch"],
            checkpoint["best_val_loss"],
            checkpoint["best_val_acc"],
        )

    def evaluate_model(
        self, model: LeNet, data_loader: DataLoader[CIFAR100]
    ) -> Tuple[float, float]:
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                total_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        avg_loss: float = total_loss / total
        accuracy: float = 100.0 * correct / total
        return avg_loss, accuracy

    def train(
        self,
        train_loader: DataLoader[CIFAR100],
        val_loader: DataLoader[CIFAR100],
        test_loader: DataLoader[CIFAR100],
        max_epochs: Optional[int] = None,
        max_patience: int = 10,
        scheduler_fn: Optional[LRScheduler] = None,
        manual_scheduler: bool = False,
    ) -> float:
        start_epoch, best_val_loss, best_val_acc = self.load_checkpoint()

        epoch = start_epoch
        epochs: int = max_epochs or self.config.NUM_EPOCHS

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.config.LEARNING_RATE,
            momentum=self.config.MOMENTUM,
            weight_decay=self.config.WEIGHT_DECAY,
        )

        scheduler = (
            scheduler_fn
            if manual_scheduler
            else CosineAnnealingLR(optimizer=optimizer, T_max=epochs)
        )

        best_model_state = None
        patience = max_patience
        patience_counter = 0
        train_acc = 0.0
        avg_train_loss = 0.0

        epoch_pbar = tqdm(
            range(start_epoch, epochs),
            initial=start_epoch,
            total=epochs,
            desc="Training",
            unit="epoch",
            leave=True,
            colour="green",
            position=0,
            dynamic_ncols=True,
        )

        try:
            for epoch in epoch_pbar:
                self.model.train()
                train_loss = 0
                correct = 0
                total = 0

                for batch_idx, (inputs, targets) in enumerate(train_loader):
                    inputs = inputs.to(self.device, non_blocking=True)
                    targets = targets.to(self.device, non_blocking=True)

                    optimizer.zero_grad(set_to_none=True)

                    with torch.amp.autocast_mode.autocast(device_type=self.device_type):
                        outputs = self.model(inputs)
                        loss = criterion(outputs, targets)

                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()

                    # Update metrics
                    train_acc = 100.0 * correct / total
                    avg_train_loss = train_loss / (batch_idx + 1)

                    global_step = epoch * len(train_loader) + batch_idx
                    self.metrics.log_metrics(
                        split="train",
                        loss=avg_train_loss,
                        accuracy=train_acc,
                        step=global_step,
                    )

                    del inputs, targets, outputs, loss
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                # Validation phase
                val_loss, val_acc = self.evaluate_model(self.model, val_loader)
                if scheduler is not None:
                    scheduler.step()

                self.metrics.log_metrics(
                    split="validation",
                    loss=val_loss,
                    accuracy=val_acc,
                    step=epoch,
                )

                epoch_pbar.set_postfix(
                    {
                        "ep": f"{epoch+1}/{max_epochs or self.config.NUM_EPOCHS}",
                        "tr_loss": f"{avg_train_loss:.3f}",
                        "tr_acc": f"{train_acc:.1f}%",
                        "val_loss": f"{val_loss:.3f}",
                        "val_acc": f"{val_acc:.1f}%",
                    },
                )

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_val_loss = val_loss
                    best_model_state = self.model.state_dict().copy()
                    self.save_checkpoint(epoch, best_val_loss, best_val_acc)
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    logging.info(f"Early stopping at epoch {epoch}")
                    break

            else:
                logging.info("Training completed!")
            # Final evaluation
            if best_model_state is not None:
                self.model.load_state_dict(best_model_state)

            test_loss, test_acc = self.evaluate_model(self.model, test_loader)
            self.metrics.log_metrics(
                split="test",
                loss=test_loss,
                accuracy=test_acc,
                step=epoch,
            )
            logging.info(
                f"Final Test Results - Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%"
            )

            return best_val_acc

        finally:
            self.metrics.close()


"""
## Federated Class

The core of this project.
After setting a baseline using the CentralizedTrainer in the Cell above, the following class is responsible to generate, train, aggregate and evaluate the distributed model on client, with different settings:

### Dataset modularity:
* iid distribution of the classes for the clients
* non_iid distribution
* custom number of classes per clients

### Clients Participation
* Uniform selection of the clients
* Dirichlet distribution to select the clients

### Others
This class also consider creating checkpoint after each generation if the global stats improves to resume training at a later time. This may be useful, for example, in the following cases:
1. Federated Learning takes a really long times, thus if the real scenario the amount of client that can participate is too low, the training can be paused and resumed when the availability is increased.
2. Academic reasons:
   1. The code takes 50h to complete, pause it if you need your pc for something else, and then resume it when it is free to compute other generations.
   2. Google Colab limit the amount of resources per day, when the GPU session end, wait until the resource returns available to resume with the latest best model found.
"""


class FederatedTrainer:
    def __init__(
        self,
        model: LeNet,
        train_dataset: Dataset[CIFAR100],
        val_loader: DataLoader[CIFAR100],
        test_loader: DataLoader[CIFAR100],
        config: FederatedConfig,
        experiment_name: Optional[str] = None,
    ) -> None:
        self.config = config
        self.global_model = model.to(config.DEVICE)
        self.device = config.DEVICE
        self.device_type = str(config.DEVICE)
        self.val_loader = val_loader
        self.test_loader = test_loader

        # Pre-create all client shards and dataloaders
        self.num_clients = config.NUM_CLIENTS
        self.client_loaders = self._setup_client_data(train_dataset)

        # Setup client selection
        if config.PARTICIPATION_MODE == "skewed":
            if config.DIRICHLET_ALPHA is None:
                raise ValueError("dirichlet_alpha required for skewed mode")
            self.selection_probs = np.random.dirichlet(
                [config.DIRICHLET_ALPHA] * config.NUM_CLIENTS
            )
        else:
            self.selection_probs = np.ones(config.NUM_CLIENTS) / config.NUM_CLIENTS

        experiment_group = "federated"

        self.metrics = MetricsManager(
            config=config,
            model_name=model.__class__.__name__.lower(),
            training_type=experiment_group,
            experiment_name=experiment_name,
        )

        # Pre-allocate client models
        self.client_models = [copy.deepcopy(model) for _ in range(config.NUM_CLIENTS)]
        self.checkpoint_dir = config.MODELS_DIR / "federated"
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.checkpoint_name = (
            f"{'iid' if config.CLASSES_PER_CLIENT is None else f'noniid_{config.CLASSES_PER_CLIENT}cls'}"
            f"_{config.PARTICIPATION_MODE}"
            f"_C{config.NUM_CLIENTS}"
            f"_P{config.PARTICIPATION_RATE}"
            f"_E{config.LOCAL_EPOCHS}"
            f"_{'two_phase' if config.TWO_PHASE else 'standard'}"
            f".pt"
        )
        self.checkpoint_path = self.checkpoint_dir / self.checkpoint_name

    def _setup_client_data(
        self, dataset: Dataset[CIFAR100]
    ) -> List[DataLoader[CIFAR100]]:
        shards: List[Subset[CIFAR100]] = (
            self._create_iid_shards(dataset)
            if self.config.CLASSES_PER_CLIENT is None
            else self._create_noniid_shards(dataset)
        )

        return [
            DataLoader(
                shard,
                batch_size=self.config.BATCH_SIZE,
                shuffle=True,
                num_workers=self.config.NUM_WORKERS,
                pin_memory=self.config.PIN_MEMORY,
                persistent_workers=self.config.PERSISTENT_WORKERS,
                prefetch_factor=self.config.PREFETCH_FACTOR,
                drop_last=True,
            )
            for shard in shards
        ]

    def _create_iid_shards(self, dataset: Dataset[CIFAR100]) -> List[Subset[CIFAR100]]:
        """Create IID data shards."""
        if len(dataset) == 0:
            raise ValueError("Empty dataset")

        indices = np.random.permutation(len(dataset))
        shard_size = len(dataset) // self.num_clients

        return [
            Subset(dataset, indices[i : i + shard_size])
            for i in range(0, len(indices), shard_size)
        ]

    def _create_noniid_shards(
        self, dataset: Dataset[CIFAR100]
    ) -> List[Subset[CIFAR100]]:
        """Create non-IID data shards using class distribution."""
        # Handle both Dataset and Subset cases
        if isinstance(dataset, Subset):
            # If dataset is a Subset, get targets from the original dataset
            targets = np.array(
                [dataset.dataset.targets[idx] for idx in dataset.indices]
            )
            original_indices = np.array(dataset.indices)
        else:
            # If dataset is the original dataset
            targets = np.array(dataset.targets)
            original_indices = np.arange(len(dataset))

        # Group indices by class
        class_indices = {
            label: np.where(targets == label)[0]
            for label in range(self.config.NUM_CLASSES)
        }

        # Convert relative indices back to original dataset indices
        class_indices = {
            label: original_indices[indices.astype(int)]
            for label, indices in class_indices.items()
        }

        client_indices = []
        num_classes = self.config.CLASSES_PER_CLIENT or self.config.NUM_CLASSES

        for _ in range(self.num_clients):
            indices = []
            # Select random classes for this client
            selected_classes = np.random.choice(
                list(class_indices.keys()),
                size=min(num_classes, len(class_indices)),
                replace=False,
            )

            # Add samples from each selected class
            for class_label in selected_classes:
                class_samples = np.random.choice(
                    class_indices[class_label],
                    size=len(class_indices[class_label]) // self.num_clients,
                    replace=False,
                )
                indices.extend(class_samples)

            client_indices.append(
                Subset(
                    dataset.dataset if isinstance(dataset, Subset) else dataset, indices
                )
            )

        return client_indices

    def _evaluate(self, loader: DataLoader[CIFAR100]) -> Tuple[float, float]:
        """Evaluate model on given data loader.
        # Returns
            Tuple of (loss, accuracy)
        """
        self.global_model.eval()
        total_loss = 0
        correct = 0
        total = 0
        criterion = nn.CrossEntropyLoss()

        with (
            torch.no_grad(),
            torch.amp.autocast_mode.autocast(device_type=self.device_type),
        ):
            for inputs, targets in loader:
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                outputs = self.global_model(inputs)
                loss = criterion(outputs, targets)

                total_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        return total_loss / total, 100.0 * correct / total

    def _aggregate_models(self, selected_clients: List[int] | npt.NDArray) -> None:
        """Aggregate models using weighted average based on dataset sizes."""
        with (
            torch.no_grad(),
            torch.amp.autocast_mode.autocast(device_type=self.device_type),
        ):
            # Calculate total samples across selected clients
            total_samples = sum(
                len(self.client_loaders[idx].dataset) for idx in selected_clients
            )

            # Initialize aggregated parameters
            for k, v in self.global_model.state_dict().items():
                weighted_sum = torch.zeros_like(v)
                for idx in selected_clients:
                    # Get client's weight based on dataset size
                    client_weight = (
                        len(self.client_loaders[idx].dataset) / total_samples
                    )
                    client_params = (
                        self.client_models[idx].state_dict()[k].to(self.device)
                    )
                    weighted_sum.add_(client_params * client_weight)

                # Update global model
                v.copy_(weighted_sum)

    def save_checkpoint(self, round_idx: int, best_val_loss: float) -> None:
        checkpoint = {
            "round": round_idx,
            "model_state_dict": self.global_model.state_dict(),
            "best_val_loss": best_val_loss,
            "config": self.config.serialize(),
        }
        torch.save(checkpoint, self.checkpoint_path)
        logging.info(f"Checkpoint saved: {self.checkpoint_path}")

    def load_checkpoint(self) -> Tuple[int, float]:
        if not self.checkpoint_path.exists():
            return 0, float("inf")

        checkpoint = torch.load(self.checkpoint_path)
        config_data = checkpoint["config"]

        # Deserialize config for comparison
        if isinstance(config_data, dict):
            saved_config = FederatedConfig.deserialize(config_data)
        else:
            saved_config = config_data

        # Validate using matches method
        if not self.config.matches(saved_config):
            logging.warning("Config mismatch in checkpoint, starting fresh training")
            return 0, float("inf")

        self.global_model.load_state_dict(checkpoint["model_state_dict"])
        logging.info(f"Resumed from checkpoint: {self.checkpoint_path}")
        return checkpoint["round"], checkpoint["best_val_loss"]

    def train_client(self, client_idx: int, model: LeNet) -> None:
        """Train a single client in-place."""
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        batch_count = 0

        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=self.config.LEARNING_RATE,
            momentum=self.config.MOMENTUM,
            weight_decay=self.config.WEIGHT_DECAY,
            nesterov=True,
        )
        criterion = nn.CrossEntropyLoss()
        scaler = torch.amp.grad_scaler.GradScaler(device=self.device_type)

        local_epochs = self.config.get_epochs_for_training()

        for epoch in range(local_epochs):
            for inputs, targets in self.client_loaders[client_idx]:
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast_mode.autocast(device_type=self.device_type):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                with torch.no_grad():
                    total_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
                    batch_count += 1

                    # Log metrics periodically
                    if batch_count % 10 == 0:
                        avg_loss = total_loss / batch_count
                        accuracy = 100.0 * correct / total

                        # Global step calculation
                        global_step = (
                            epoch * len(self.client_loaders[client_idx]) + batch_count
                        )

                        self.metrics.log_metrics(
                            split="train",
                            loss=avg_loss,
                            accuracy=accuracy,
                            step=global_step,
                        )

                del inputs, targets, outputs, loss
                torch.cuda.empty_cache()

    def _shuffle_and_redistribute_models(
        self, selected_clients: npt.NDArray
    ) -> Dict[int, int]:
        """Shuffle and redistribute models among selected clients.
        Returns mapping of client_id -> received_model_id"""
        shuffled_indices = selected_clients.copy()
        np.random.shuffle(shuffled_indices)
        return {
            client: shuffled_indices[i] for i, client in enumerate(selected_clients)
        }

    def train(self, max_patience: int = 50) -> None:
        # Load existing checkpoint if available
        start_round, best_val_loss = self.load_checkpoint()
        best_model_state = (
            self.global_model.state_dict().copy() if start_round > 0 else None
        )

        patience_counter = 0
        best_val_acc = 0.0

        if start_round > 0:
            logging.info(f"Resuming training from round {start_round}")

        round_idx = start_round
        rounds = self.config.NUM_ROUNDS

        round_pbar = tqdm(
            range(start_round, rounds),
            initial=start_round,
            total=rounds,
            desc="Training",
            unit="round",
            colour="green",
            dynamic_ncols=True,
        )

        try:
            for round_idx in round_pbar:
                # Select clients
                num_selected = max(
                    1, int(self.config.PARTICIPATION_RATE * self.config.NUM_CLIENTS)
                )
                selected_clients = np.random.choice(
                    self.config.NUM_CLIENTS,
                    size=num_selected,
                    replace=False,
                    p=self.selection_probs,
                )

                # Train selected clients in parallel
                for idx in selected_clients:
                    self.client_models[idx].load_state_dict(
                        self.global_model.state_dict()
                    )
                    self.train_client(idx, self.client_models[idx])

                # Evaluate after first phase

                val_loss_p1, val_acc_p1 = self._evaluate(self.val_loader)
                if self.config.TWO_PHASE:
                    # Phase 2: Shuffle and retrain
                    logging.debug("Phase 2: Training with shuffled models")
                    model_assignments = self._shuffle_and_redistribute_models(
                        selected_clients
                    )

                    # Train with shuffled models
                    for client_idx, model_idx in model_assignments.items():
                        self.client_models[client_idx].load_state_dict(
                            self.client_models[model_idx].state_dict()
                        )
                        self.train_client(client_idx, self.client_models[client_idx])

                # Aggregate models
                self._aggregate_models(selected_clients)

                # Evaluate
                val_loss, val_acc = self._evaluate(self.val_loader)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_val_acc = val_acc
                    best_model_state = self.global_model.state_dict().copy()
                    patience_counter = 0
                    self.save_checkpoint(round_idx, best_val_loss)
                else:
                    patience_counter += 1

                metrics_dict = {
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                    "best_val_loss": best_val_loss,
                    "best_val_acc": best_val_acc,
                }

                if self.config.TWO_PHASE:
                    metrics_dict.update(
                        {
                            "val_loss_phase1": val_loss_p1,
                            "val_accuracy_phase1": val_acc_p1,
                        }
                    )

                # Log metrics
                self.metrics.log_fl_metrics(
                    round_idx=round_idx,
                    metrics=metrics_dict,
                    client_stats={
                        "num_selected": len(selected_clients),
                        "participation_rate": len(selected_clients)
                        / self.config.NUM_CLIENTS,
                    },
                )

                # Update progress bar
                postfix_dict = {
                    "val_loss": f"{val_loss:.4f}",
                    "val_acc": f"{val_acc:.2f}%",
                    "best": f"{best_val_acc:.2f}%",
                }
                if self.config.TWO_PHASE:
                    postfix_dict.update({"p1_acc": f"{val_acc_p1:.2f}%"})
                round_pbar.set_postfix(postfix_dict, refresh=True)

                if patience_counter >= max_patience:
                    logging.info(f"Early stopping at round {round_idx}")
                    break

            # Final evaluation
            if best_model_state:
                self.global_model.load_state_dict(best_model_state)
            test_loss, test_acc = self._evaluate(self.test_loader)
            self.metrics.log_metrics(
                split="test",
                loss=test_loss,
                accuracy=test_acc,
                step=round_idx + 1,
            )

            self.save_checkpoint(round_idx + 1, best_val_loss)

        finally:
            self.metrics.close()


def create_comparison_plots(base_dir: Path) -> None:
    """Create comprehensive comparison plots for all experiments."""

    plots_dir = base_dir / "comparison_plots"
    plots_dir.mkdir(exist_ok=True)

    # Set style
    plt.style.use("default")  # Use matplotlib default style
    # Set color cycle
    plt.rcParams["axes.prop_cycle"] = plt.cycler(
        color=[
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
        ]
    )
    sns.set_palette("husl")

    def load_metrics(run_dir: Path) -> pd.DataFrame:
        """Load metrics from a run directory."""
        metrics_dir = run_dir / "metrics"
        metrics = {}

        # Load validation metrics
        val_file = metrics_dir / "val_metrics.csv"
        if val_file.exists():
            metrics["validation"] = pd.read_csv(val_file)

        # Load test metrics
        test_file = metrics_dir / "test_metrics.csv"
        if test_file.exists():
            metrics["test"] = pd.read_csv(test_file)

        return metrics

    def plot_baseline_comparison():
        """Compare centralized vs federated baselines."""
        plt.figure(figsize=(15, 6))

        # Plot validation accuracy
        plt.subplot(1, 2, 1)

        variants = {
            "Centralized": base_dir / "centralized" / "lenet_baseline",
            "FedAvg Standard": base_dir / "federated" / "lenet_baseline_standard",
            "FedAvg Two-Phase": base_dir / "federated" / "lenet_baseline_two_phase",
        }

        for label, path in variants.items():
            if path.exists():
                metrics = load_metrics(path)
                if "validation" in metrics:
                    plt.plot(
                        metrics["validation"]["step"],
                        metrics["validation"]["accuracy"],
                        label=label,
                        linewidth=2,
                    )

        plt.title("Validation Accuracy Comparison")
        plt.xlabel("Epochs/Rounds")
        plt.ylabel("Accuracy (%)")
        plt.legend()
        plt.grid(True)

        # Plot validation loss
        plt.subplot(1, 2, 1)

        variants = {
            "Centralized": base_dir / "centralized" / "lenet_baseline",
            "FedAvg Standard": base_dir / "federated" / "lenet_baseline_standard",
            "FedAvg Two-Phase": base_dir / "federated" / "lenet_baseline_two_phase",
        }

        for label, path in variants.items():
            if path.exists():
                metrics = load_metrics(path)
                if "validation" in metrics:
                    plt.plot(
                        metrics["validation"]["step"],
                        metrics["validation"]["loss"],
                        label=label,
                        linewidth=2,
                    )

        plt.title("Validation Loss: Centralized vs Federated")
        plt.xlabel("Epochs/Rounds")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(plots_dir / "baseline_comparison.png", dpi=300, bbox_inches="tight")
        plt.close()

    def plot_participation_comparison():
        """Compare different participation strategies."""
        participation_dir = base_dir / "federated_participation"
        if not participation_dir.exists():
            return

        plt.figure(figsize=(12, 6))

        for run_dir in participation_dir.glob("lenet_participation_*"):
            variant = run_dir.name.split("_")[-1]
            metrics = load_metrics(run_dir)

            if "validation" in metrics:
                plt.plot(
                    metrics["validation"]["step"],
                    metrics["validation"]["accuracy"],
                    label=f"Participation: {variant}",
                    linewidth=2,
                )

        plt.title("Impact of Participation Strategies")
        plt.xlabel("Rounds")
        plt.ylabel("Validation Accuracy (%)")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(
            plots_dir / "participation_comparison.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def plot_heterogeneity_comparison():
        """Compare different heterogeneity settings."""
        heterogeneity_dir = base_dir / "federated_heterogeneity"
        if not heterogeneity_dir.exists():
            return

        # Separate plots for different J values
        j_values = set()
        for run_dir in heterogeneity_dir.glob("lenet_heterogeneity_*"):
            # More robust parsing of J value
            parts = run_dir.name.split("_")
            for part in parts:
                if part.startswith("J") and part[1:].isdigit():
                    j = int(part[1:])
                    j_values.add(j)
                    break

        for j in sorted(j_values):
            plt.figure(figsize=(12, 6))

            # Group runs by training mode
            for mode in ["standard", "two_phase"]:
                # Match pattern for both standard and two-phase runs
                pattern = f"*J{j}*{mode}*"

                for run_dir in heterogeneity_dir.glob(pattern):
                    # Parse run configuration
                    name_parts = run_dir.name.split("_")
                    if "iid" in name_parts:
                        label = f"IID ({mode})"
                    else:
                        # Find the part containing "cls"
                        for part in name_parts:
                            if "cls" in part:
                                cls = part.replace("cls", "")
                                label = f"{cls} classes/client ({mode})"
                                break
                        else:
                            continue  # Skip if we can't parse the configuration

                    metrics = load_metrics(run_dir)
                    if "validation" in metrics:
                        plt.plot(
                            metrics["validation"]["step"],
                            metrics["validation"]["accuracy"],
                            label=label,
                            linewidth=2,
                        )

            plt.title(f"Impact of Data Heterogeneity (J={j})")
            plt.xlabel("Rounds")
            plt.ylabel("Validation Accuracy (%)")
            plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            plt.grid(True)

            plt.tight_layout()
            plt.savefig(
                plots_dir / f"heterogeneity_J{j}_comparison.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

    def plot_training_mode_comparison():
        """Compare standard vs two-phase training across experiments."""
        plt.figure(figsize=(15, 10))

        categories = ["Baseline", "Participation", "Heterogeneity"]
        standard_accs = []
        two_phase_accs = []

        for category in categories:
            if category == "Baseline":
                path = base_dir / "federated"
                standard = "lenet_baseline_standard"
                two_phase = "lenet_baseline_two_phase"
            else:
                path = base_dir / f"federated_{category.lower()}"
                standard = "*_standard"
                two_phase = "*_two_phase"

            # Get best accuracies for both modes
            best_standard = 0
            best_two_phase = 0

            for mode, pattern in [("standard", standard), ("two_phase", two_phase)]:
                for run_dir in path.glob(pattern):
                    metrics = load_metrics(run_dir)
                    if "validation" in metrics:
                        acc = metrics["validation"]["accuracy"].max()
                        if mode == "standard":
                            best_standard = max(best_standard, acc)
                        else:
                            best_two_phase = max(best_two_phase, acc)

            standard_accs.append(best_standard)
            two_phase_accs.append(best_two_phase)

        # Plot comparison
        x = np.arange(len(categories))
        width = 0.35

        plt.bar(x - width / 2, standard_accs, width, label="Standard FedAvg")
        plt.bar(x + width / 2, two_phase_accs, width, label="Two-Phase FedAvg")

        plt.xlabel("Experiment Category")
        plt.ylabel("Best Validation Accuracy (%)")
        plt.title("Standard vs Two-Phase Training Comparison")
        plt.xticks(x, categories)
        plt.legend()
        plt.grid(True, axis="y")

        # Add value labels
        for i, v in enumerate(standard_accs):
            plt.text(i - width / 2, v, f"{v:.1f}%", ha="center", va="bottom")
        for i, v in enumerate(two_phase_accs):
            plt.text(i + width / 2, v, f"{v:.1f}%", ha="center", va="bottom")

        plt.tight_layout()
        plt.savefig(
            plots_dir / "training_mode_comparison.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    # Generate all plots
    plot_baseline_comparison()
    plot_participation_comparison()
    plot_heterogeneity_comparison()
    plot_training_mode_comparison()

    # Create a final summary plot
    plt.figure(figsize=(15, 10))

    # Best results from each category
    categories = {
        "Centralized": base_dir / "centralized",
        "FedAvg Baseline": base_dir / "federated",
        "Participation": base_dir / "federated_participation",
        "Heterogeneity": base_dir / "federated_heterogeneity",
    }

    best_accuracies = {}

    for category, directory in categories.items():
        best_acc = 0
        if directory.exists():
            for run_dir in directory.glob("**/summary.json"):
                with open(run_dir) as f:
                    summary = json.load(f)
                    if (
                        "final_metrics" in summary
                        and "test" in summary["final_metrics"]
                    ):
                        acc = summary["final_metrics"]["test"]["best_accuracy"]
                        best_acc = max(best_acc, acc)
        best_accuracies[category] = best_acc

    # Plot best results
    plt.bar(best_accuracies.keys(), best_accuracies.values())
    plt.title("Best Test Accuracy Across Different Approaches")
    plt.ylabel("Test Accuracy (%)")
    plt.xticks(rotation=45)
    plt.grid(True, axis="y")

    for i, (category, acc) in enumerate(best_accuracies.items()):
        plt.text(i, acc, f"{acc:.1f}%", ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig(plots_dir / "final_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()


def compare_experiments(base_dir: Path) -> None:
    """Compare results across different experimental settings."""
    # Create comparison plots
    create_comparison_plots(base_dir)

    # Compare centralized vs federated baseline
    MetricsManager.compare_runs(base_dir, "centralized_baseline")
    MetricsManager.compare_runs(base_dir, "federated_baseline")

    # Compare different participation schemes
    MetricsManager.compare_runs(base_dir, "federated_participation")

    # Compare different local steps
    MetricsManager.compare_runs(base_dir, "federated_heterogeneity")


def cleanup_memory():
    """Aggressive memory cleanup"""
    import gc

    # Clear PyTorch cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.empty_cache()

    # Force garbage collection
    gc.collect(generation=2)

    if os.name == "nt":  # Windows
        import ctypes

        ctypes.windll.kernel32.SetProcessWorkingSetSize(-1, -1)


class ExperimentRunner:
    """Manages and runs all experiments sequentially with checkpointing."""

    def __init__(self, base_config: BaseConfig):
        self.config = base_config
        self.data = Cifar100DatasetManager(base_config)
        self.experiments_dir = base_config.RUNS_DIR / "experiments_status"
        self.experiments_dir.mkdir(exist_ok=True)
        self.status_file = self.experiments_dir / "completion_status.json"
        self.status = self._load_status()

    def _load_status(self) -> dict:
        """Load or initialize experiment status."""
        if self.status_file.exists():
            with open(self.status_file) as f:
                return json.load(f)
        return {
            "centralized_hyperparams": {},
            "centralized_baseline": False,
            "federated_baseline": {"standard": False, "two_phase": False},
            "participation_studies": {},
            "participation_studies_two_phase": {},
            "heterogeneity_study": {},
            "heterogeneity_study_two_phase": {},
        }

    def _save_status(self) -> None:
        """Save current experiment status."""
        with open(self.status_file, "w") as f:
            json.dump(self.status, f, indent=2)

    def mark_completed(
        self,
        experiment: str,
        variant: Optional[str] = None,
        metrics: Optional[dict] = None,
    ) -> None:
        """Mark an experiment as completed with optional metrics."""
        if variant:
            if experiment not in self.status:
                self.status[experiment] = {}
            self.status[experiment][variant] = {"completed": True, **(metrics or {})}
        else:
            self.status[experiment] = True
        self._save_status()

    def is_completed(self, experiment: str, variant: Optional[str] = None) -> bool:
        """Check if an experiment is completed."""
        if variant:
            return self.status.get(experiment, {}).get(variant, False)
        return self.status.get(experiment, False)

    def run_centralized_baseline(self) -> None:
        """Run centralized training with hyperparameter search."""
        # If baseline is already completed, skip everything
        if self.is_completed("centralized_baseline"):
            logging.info("Centralized baseline already completed, skipping...")
            return

        # Grid search parameters
        grid_search_epochs = 50
        lr_values = [0.1, 0.01, 0.001]
        schedulers = [
            ("cosine", lambda opt: CosineAnnealingLR(opt, T_max=grid_search_epochs)),
            ("step", lambda opt: StepLR(opt, step_size=30, gamma=0.1)),
        ]

        # Check if we need to do grid search
        need_grid_search = False
        for lr in lr_values:
            for scheduler_name, _ in schedulers:
                variant = f"lr{lr}_{scheduler_name}"
                if not self.is_completed("centralized_hyperparams", variant):
                    need_grid_search = True
                    break

        best_val_acc = 0
        best_config = None

        if need_grid_search:
            logging.info("Starting hyperparameter grid search...")
            # Grid search
            for lr in lr_values:
                for scheduler_name, scheduler_fn in schedulers:
                    variant = f"lr{lr}_{scheduler_name}"

                    if self.is_completed("centralized_hyperparams", variant):
                        logging.info(f"Skipping completed hyperparams: {variant}")
                        # Load saved accuracy
                        saved_acc = self.status["centralized_hyperparams"][variant].get(
                            "val_acc", 0
                        )
                        if saved_acc > best_val_acc:
                            best_val_acc = saved_acc
                            best_config = (lr, scheduler_name)
                        continue

                    logging.info(
                        f"Testing hyperparams: lr={lr}, scheduler={scheduler_name}"
                    )

                    config = dataclasses.replace(self.config, LEARNING_RATE=lr)
                    model = LeNet(config)
                    trainer = CentralizedTrainer(
                        model=model,
                        config=config,
                        experiment_name=f"grid_search_{variant}",
                    )

                    val_acc = trainer.train(
                        train_loader=self.data.train_loader,
                        val_loader=self.data.val_loader,
                        test_loader=self.data.test_loader,
                        max_epochs=grid_search_epochs,
                        max_patience=10,
                        scheduler_fn=scheduler_fn,
                    )

                    # Save results
                    self.mark_completed(
                        "centralized_hyperparams", variant, {"val_acc": val_acc}
                    )

                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        best_config = (lr, scheduler_name)

                    cleanup_memory()

        else:
            # Load best config from saved results
            logging.info("Loading best config from completed hyperparameter search...")
            for lr in lr_values:
                for scheduler_name, _ in schedulers:
                    variant = f"lr{lr}_{scheduler_name}"
                    if variant in self.status["centralized_hyperparams"]:
                        val_acc = self.status["centralized_hyperparams"][variant].get(
                            "val_acc", 0
                        )
                        if val_acc > best_val_acc:
                            best_val_acc = val_acc
                            best_config = (lr, scheduler_name)

        # Train final model with best hyperparameters
        if not self.is_completed("centralized_baseline") and best_config:
            logging.info(f"Training final model with best config: {best_config}")
            lr, scheduler_name = best_config
            variant = f"lr{lr}_{scheduler_name}"

            config = dataclasses.replace(self.config, LEARNING_RATE=lr)
            model = LeNet(config)
            trainer = CentralizedTrainer(
                model=model, config=config, experiment_name=f"baseline_{variant}"
            )

            scheduler_fn = next(s[1] for s in schedulers if s[0] == scheduler_name)
            trainer.train(
                train_loader=self.data.train_loader,
                val_loader=self.data.val_loader,
                test_loader=self.data.test_loader,
                scheduler_fn=scheduler_fn,
            )

            self.mark_completed("centralized_baseline")
            cleanup_memory()

    def run_federated_baseline(self) -> None:
        """Run federated learning baseline with both training modes."""
        for mode in ["standard", "two_phase"]:
            if self.is_completed("federated_baseline", mode):
                logging.info(f"Federated baseline ({mode}) already completed")
                continue

            logging.info(f"Running federated baseline (IID) with {mode} training")
            config = FederatedConfig(TWO_PHASE=(mode == "two_phase"))
            model = LeNet(config)
            trainer = FederatedTrainer(
                model=model,
                train_dataset=self.data.train_dataset,
                val_loader=self.data.val_loader,
                test_loader=self.data.test_loader,
                config=config,
                experiment_name=f"baseline_{mode}",
            )
            trainer.train()  # type: ignore
            self.mark_completed("federated_baseline", mode)
            cleanup_memory()

    def run_participation_studies(self) -> None:
        """Run participation scheme experiments with both training modes."""
        gamma_values = [0.1, 0.5, 1.0]

        for training_mode in ["standard", "two_phase"]:
            status_key = f"participation_studies{'_two_phase' if training_mode == 'two_phase' else ''}"

            for mode in ["uniform", "skewed"]:
                for gamma in gamma_values:
                    variant = f"{mode}_gamma{gamma}" if mode == "skewed" else "uniform"

                    if self.is_completed(status_key, variant):
                        logging.info(
                            f"Participation study {variant} ({training_mode}) already completed"
                        )
                        continue

                    logging.info(
                        f"Running participation study: {variant} with {training_mode} training"
                    )
                    config = FederatedConfig(
                        NUM_CLIENTS=100,
                        PARTICIPATION_RATE=0.1,
                        LOCAL_EPOCHS=4,
                        PARTICIPATION_MODE=mode,
                        DIRICHLET_ALPHA=gamma if mode == "skewed" else None,
                        TWO_PHASE=(training_mode == "two_phase"),
                    )
                    model = LeNet(config)
                    trainer = FederatedTrainer(
                        model=model,
                        train_dataset=self.data.train_dataset,
                        val_loader=self.data.val_loader,
                        test_loader=self.data.test_loader,
                        config=config,
                        experiment_name=f"participation_{variant}_{training_mode}",
                    )
                    trainer.train()
                    self.mark_completed(status_key, variant)
                    cleanup_memory()

    def run_heterogeneity_study(self, settings: ExperimentSettings) -> None:
        """Run comprehensive heterogeneity study with both training modes."""
        for training_mode in ["standard", "two_phase"]:
            status_key = f"heterogeneity_study{'_two_phase' if training_mode == 'two_phase' else ''}"

            for nc in settings.Nc:
                for j in settings.J_configs.keys():
                    variant = f"{'iid' if nc is None else f'noniid_{nc}cls'}_J{j}"

                    if self.is_completed(status_key, variant):
                        logging.info(
                            f"Skipping completed study: {variant} ({training_mode})"
                        )
                        continue

                    logging.info(
                        f"Running study: {variant} with {training_mode} training"
                    )
                    config = FederatedConfig(
                        NUM_CLIENTS=settings.K,
                        PARTICIPATION_RATE=settings.C,
                        LOCAL_EPOCHS=j,
                        NUM_ROUNDS=settings.get_rounds(j),
                        CLASSES_PER_CLIENT=nc,
                        TWO_PHASE=(training_mode == "two_phase"),
                    )

                    model = LeNet(config)
                    trainer = FederatedTrainer(
                        model=model,
                        train_dataset=self.data.train_dataset,
                        val_loader=self.data.val_loader,
                        test_loader=self.data.test_loader,
                        config=config,
                        experiment_name=f"heterogeneity_{variant}_{training_mode}",
                    )

                    trainer.train()
                    self.mark_completed(status_key, variant)
                    cleanup_memory()

    def run_all(self) -> None:
        """Run all experiments sequentially."""
        try:
            # Baselines
            self.run_centralized_baseline()
            self.run_federated_baseline()

            # Studies
            self.run_participation_studies()

            settings = ExperimentSettings(
                # K=100,  # Number of clients
                # C=0.1,  # Client fraction
                # Nc=(None, 1, 5, 10, 50),  # None = IID
                # J_configs={
                #     4: 2000,  # Base configuration
                #     8: 1000,  # Halved rounds
                #     16: 500,  # Quarter rounds
                # },
            )
            self.run_heterogeneity_study(settings)

            # Generate final comparisons
            compare_experiments(self.config.RUNS_DIR)

        except Exception as e:
            logging.error(f"Error during experiments: {e}")
            raise
        finally:
            cleanup_memory()


if __name__ == "__main__":
    # Set random seeds
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)

    # Run all experiments
    runner = ExperimentRunner(config)
    runner.run_all()
