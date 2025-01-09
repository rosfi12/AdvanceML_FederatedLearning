########################
# Imports and Setup
########################

import copy
import dataclasses
import itertools
import json
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm


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


########################
# Configuration
########################


@dataclass(frozen=True)
class BaseConfig:
    """Base configuration with common parameters."""

    # System
    DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CPU_COUNT: int = os.cpu_count() or 1
    NUM_WORKERS: int = min(2, CPU_COUNT)
    SEED: int = 42

    # Paths
    ROOT_DIR: Path = Path(__file__).parent
    DATA_DIR: Path = ROOT_DIR / "data"
    MODELS_DIR: Path = ROOT_DIR / "models"
    RESULTS_DIR: Path = ROOT_DIR / "results"
    CONFIGS_DIR: Path = ROOT_DIR / "configs"

    # Common Training Parameters
    BATCH_SIZE: int = 64
    LEARNING_RATE: float = 0.001
    NUM_EPOCHS: int = 200

    # Federated Learning
    NUM_CLIENTS: int = 5
    FL_ROUNDS: int = 20
    LOCAL_EPOCHS: int = 5

    def setup_directories(self):
        """Create necessary directories."""
        for dir_path in [
            self.DATA_DIR,
            self.MODELS_DIR,
            self.RESULTS_DIR,
            self.CONFIGS_DIR,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class LSTMConfig(BaseConfig):
    """LSTM specific configuration."""

    MODEL_TYPE: Literal["lstm"] = "lstm"
    HIDDEN_SIZE: int = 256
    EMBEDDING_SIZE: int = 8
    BATCH_SIZE: int = 2
    NUM_LAYERS: int = 2
    DROPOUT: float = 0.2
    BIDIRECTIONAL: bool = True
    ATTENTION_HEADS: int = 2


@dataclass(frozen=True)
class LeNetConfig(BaseConfig):
    """LeNet specific configuration."""

    MODEL_TYPE: Literal["lenet"] = "lenet"
    NUM_CLASSES: int = 100
    BATCH_SIZE: int = 64
    LEARNING_RATE: float = 0.01
    WEIGHT_DECAY: float = 4e-4
    MOMENTUM: float = 0.9


class ConfigFactory:
    """Factory for creating appropriate config objects."""

    @staticmethod
    def validate_config(config: Union[LeNetConfig, LSTMConfig]) -> None:
        """Validate configuration parameters."""
        if isinstance(config, LSTMConfig):
            if config.DROPOUT > 0 and config.NUM_LAYERS < 2:
                raise ValueError(
                    f"Invalid LSTM configuration: dropout={config.DROPOUT} requires "
                    f"num_layers > 1, but got num_layers={config.NUM_LAYERS}"
                )

    @staticmethod
    def create_config(model_type: str, **kwargs) -> Union[LeNetConfig, LSTMConfig]:
        """Create a config instance based on model type."""
        if model_type.lower() == "lenet":
            config = LeNetConfig(**kwargs)
        elif model_type.lower() == "lstm":
            config = LSTMConfig(**kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        ConfigFactory.validate_config(config)
        return config

    @staticmethod
    def create_from_dict(config_dict: dict) -> Union[LeNetConfig, LSTMConfig]:
        """Create a config instance from a dictionary."""
        model_type = config_dict.pop("model_type", None)
        if not model_type:
            raise ValueError("model_type must be specified in config dictionary")
        return ConfigFactory.create_config(model_type, **config_dict)


########################
# Hyperparameters
########################


@dataclass
class HyperParameters:
    """Base hyperparameters without system config."""

    learning_rate: float
    batch_size: int

    def to_dict(self) -> Dict:
        """Convert hyperparameters to dictionary."""
        return {k: v for k, v in dataclasses.asdict(self).items()}


@dataclass
class LeNetHyperParameters(HyperParameters):
    """LeNet-specific hyperparameters."""

    weight_decay: float = 4e-4
    momentum: float = 0.9


@dataclass
class LSTMHyperParameters(HyperParameters):
    """LSTM-specific hyperparameters."""

    hidden_size: int = 256
    num_layers: int = 2
    dropout: float = 0.2
    bidirectional: bool = True
    attention_heads: int = 2


def save_hyperparameters(
    hyperparams: HyperParameters,
    configs_dir: Path,
    model_type: str,
    prefix: str = "best",
):
    """Save hyperparameters to JSON file."""
    configs_dir.mkdir(parents=True, exist_ok=True)
    config_path = configs_dir / f"{model_type}_{prefix}_hyperparameters.json"

    with open(config_path, "w") as f:
        json.dump(hyperparams.to_dict(), f, indent=4)


def load_hyperparameters(
    configs_dir: Path, model_type: str
) -> Optional[HyperParameters]:
    """Load hyperparameters from JSON file."""
    config_path = configs_dir / f"{model_type}_best_hyperparameters.json"
    if config_path.exists():
        with open(config_path, "r") as f:
            params = json.load(f)
            if model_type == "lenet":
                return LeNetHyperParameters(**params)
            else:
                return LSTMHyperParameters(**params)
    return None


########################
# Utility Functions
########################


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: BaseConfig,
    epoch: int,
    val_loss: float,
    val_acc: float,
    is_best: bool = False,
) -> Path:
    """Save model checkpoint with metrics."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    model_type = model.__class__.__name__.lower()

    # Create checkpoint filename
    checkpoint_name = (
        f"{model_type}_epoch{epoch:03d}_"
        f"val_loss{val_loss:.4f}_"
        f"val_acc{val_acc:.2f}_{timestamp}"
        f"{'_best' if is_best else ''}.pth"
    )

    save_path = config.MODELS_DIR / checkpoint_name

    # Save checkpoint
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss,
        "val_acc": val_acc,
        "config": config,
    }

    torch.save(checkpoint, save_path)
    return save_path


########################
# Dataset Management
########################


class ShakespeareDataset(Dataset):
    def __init__(
        self,
        data_path,
        clients=None,
        seq_length=80,
        char_to_idx: Optional[Dict[str, int]] = None,
    ) -> None:
        """
        Initialize the dataset by loading and preprocessing the data.
        Args:
        - data_path: Path to the JSON file containing the dataset.
        - clients: List of client IDs to load data for (default: all clients).
        - seq_length: Sequence length for character-level data.
        """
        super(ShakespeareDataset).__init__()
        self.seq_length = seq_length
        self.data = []
        self.targets = []

        if char_to_idx is None:
            # Build vocabulary if not provided (training set)
            self.vocab = [
                c
                for c in " $&'(),-.0123456789:;>?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz}"
            ]
            self.char_to_idx = {char: idx for idx, char in enumerate(self.vocab)}
            self.idx_to_char = {idx: char for idx, char in enumerate(self.vocab)}
        else:
            # Use provided vocabulary (test set)
            self.char_to_idx = char_to_idx
            self.idx_to_char = {idx: char for char, idx in char_to_idx.items()}
            self.vocab = list(char_to_idx.keys())

        self.load_data(data_path=data_path, clients=clients)

    def load_data(self, data_path, clients):
        """
        Load and preprocess data for selected clients.
        """
        with open(data_path, "r") as f:
            raw_data = json.load(f)
            # Use selected clients or default to all users in the dataset.
            selected_clients = clients if clients else raw_data["users"]
            for client in selected_clients:
                # Concatenate all text data for this client.
                user_text = " ".join(raw_data["user_data"][client]["x"])
                self.process_text(user_text)

    def process_text(self, text):
        """
        Split text data into input-output sequences of seq_length.
        """
        for i in range(len(text) - self.seq_length):
            seq = text[i : i + self.seq_length]  # Input sequence.
            target = text[i + 1 : i + self.seq_length + 1]  # Target sequence.
            seq_indices = [self.char_to_idx.get(c, 0) for c in seq]
            target_indices = [self.char_to_idx.get(c, 0) for c in target]
            self.data.append(torch.tensor(seq_indices, dtype=torch.long))
            self.targets.append(torch.tensor(target_indices, dtype=torch.long))

    def __len__(self) -> int:
        """
        Return the number of sequences in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve the input-target pair at the specified index.
        """
        return self.data[idx], self.targets[idx]


class DataManager:
    """Handles data loading, splitting, and preparation for both text and image data."""

    def __init__(self, config: BaseConfig):
        self.config = config
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.char_to_idx: Dict[str, int] = {}
        self.idx_to_char: Dict[int, str] = {}
        self.vocab_size: int = 0
        self.dataset_type: Literal["text", "image"] = "text"
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

    def load_and_split_data(
        self,
        dataset_type: Literal["text", "image"] = "text",
        val_split: float = 0.1,
    ) -> None:
        """Load data and split into train/val/test sets."""
        self.dataset_type = dataset_type

        if dataset_type == "image":
            # CIFAR-100 transforms
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
                    transforms.Normalize(
                        mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]
                    ),
                ]
            )

            # Load CIFAR-100
            self.train_dataset = torchvision.datasets.CIFAR100(
                root=str(self.config.DATA_DIR),
                train=True,
                download=True,
                transform=transform_train,
            )

            self.test_dataset = torchvision.datasets.CIFAR100(
                root=str(self.config.DATA_DIR),
                train=False,
                download=True,
                transform=transform_test,
            )

            # Create validation split
            train_size = int((1 - val_split) * len(self.train_dataset))
            val_size = len(self.train_dataset) - train_size
            self.train_dataset, self.val_dataset = random_split(
                self.train_dataset, [train_size, val_size]
            )

        else:  # text dataset (Shakespeare)
            # Load Shakespeare text
            train_file = self.config.DATA_DIR / "train_data.json"
            test_file = self.config.DATA_DIR / "test_data.json"

            if not train_file.exists() or not test_file.exists():
                raise FileNotFoundError(
                    f"Shakespeare dataset files not found at {self.config.DATA_DIR}. "
                    "Please ensure both train_data.json and test_data.json exist."
                )

            # Create training dataset
            train_dataset = ShakespeareDataset(data_path=train_file, seq_length=100)

            # Create test dataset
            test_dataset = ShakespeareDataset(
                data_path=test_file,
                seq_length=100,
                char_to_idx=train_dataset.char_to_idx,  # Use training vocab
            )

            # Store vocabulary info from training set
            self.char_to_idx = train_dataset.char_to_idx
            self.idx_to_char = train_dataset.idx_to_char
            self.vocab_size = len(train_dataset.vocab)

            # Split training data into train/val
            train_size = int((1 - val_split) * len(train_dataset))
            val_size = len(train_dataset) - train_size

            self.train_dataset, self.val_dataset = random_split(
                train_dataset, [train_size, val_size]
            )
            self.test_dataset = test_dataset

        # Create DataLoaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=True,
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=True,
        )

        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=True,
        )

    def get_client_loaders(self) -> List[DataLoader]:
        """Split training data into client-specific DataLoaders for federated learning."""
        if self.train_dataset is None:
            raise ValueError("Dataset not loaded. Call load_and_split_data first.")

        # Calculate samples per client
        total_samples = len(self.train_dataset)
        samples_per_client = total_samples // self.config.NUM_CLIENTS

        client_loaders = []
        for i in range(self.config.NUM_CLIENTS):
            start_idx = i * samples_per_client
            end_idx = (
                start_idx + samples_per_client
                if i < self.config.NUM_CLIENTS - 1
                else total_samples
            )

            # Create subset of training data for this client
            indices = list(range(start_idx, end_idx))
            subset = torch.utils.data.Subset(self.train_dataset, indices)

            # Create DataLoader for this client
            client_loader = DataLoader(
                subset,
                batch_size=self.config.BATCH_SIZE,
                shuffle=True,
                num_workers=self.config.NUM_WORKERS,
                pin_memory=True,
            )
            client_loaders.append(client_loader)

        return client_loaders


########################
# Models
########################


class LeNet(nn.Module):
    """LeNet Model as described in the paper with two 5x5 conv layers and two FC layers."""

    def __init__(self, config: LeNetConfig):
        super(LeNet, self).__init__()

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

        # Calculate size after convolutions and pooling
        feature_size = 5 * 5 * 64

        # Fully connected layers
        self.connected = nn.Sequential(
            nn.Linear(feature_size, 384),
            nn.ReLU(),
            nn.Linear(384, 192),
            nn.ReLU(),
        )

        self.classifier = nn.Linear(192, config.NUM_CLASSES)

        # Activation
        self.relu = nn.ReLU()

    def forward(self, x):
        # First conv block
        x = self.conv_block1(x)
        x = self.conv_block2(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = self.connected(x)
        x = self.classifier(x)

        return x


class CharLSTM(nn.Module):
    def __init__(self, config: LSTMConfig, vocab_size: int) -> None:
        """
        Initialize the LSTM model.
        Args:
        - vocab_size: Number of unique characters in the dataset.
        - embedding_dim: Size of the character embedding.
        - hidden_dim: Number of LSTM hidden units.
        - num_layers: Number of LSTM layers.
        - dropout: Dropout rate for regularization.
        """
        super(CharLSTM, self).__init__()
        self.embedding = nn.Embedding(
            vocab_size, config.EMBEDDING_SIZE
        )  # Character embedding layer.
        self.lstm = nn.LSTM(
            config.EMBEDDING_SIZE,
            config.HIDDEN_SIZE,
            2,
            batch_first=True,
            dropout=config.DROPOUT if config.NUM_LAYERS > 1 else 0,
        )  # LSTM layers.
        self.fc = nn.Linear(config.HIDDEN_SIZE, vocab_size)

    def forward(self, x, hidden=None):
        """
        Forward pass of the model.
        Args:
        - x: Input batch (character indices).
        - hidden: Hidden state for LSTM (default: None, initialized internally).
        Returns:
        - Output logits and the updated hidden state.
        """
        x = self.embedding(x)  # Convert indices to embeddings.
        output, hidden = self.lstm(x, hidden)  # Process through LSTM layers.
        output = self.fc(output)  # Generate logits for each character.
        return output, hidden


class ModelFactory:
    """Factory for creating models."""

    @staticmethod
    def create_model(
        config: Union[LeNetConfig, LSTMConfig], vocab_size: int | None
    ) -> LeNet | CharLSTM:
        """Create a model instance based on config type."""
        if isinstance(config, LeNetConfig):
            return LeNet(config)
        elif isinstance(config, LSTMConfig):
            if vocab_size is None:
                raise ValueError("vocab_size is required for LSTM models")
            return CharLSTM(config=config, vocab_size=vocab_size)
        raise ValueError(f"Unsupported config type: {type(config)}")


########################
# Training Framework
########################


class HyperparameterTester:
    """Handles hyperparameter optimization."""

    def __init__(
        self, model_config: LeNetConfig | LSTMConfig, data_manager: DataManager
    ) -> None:
        self.config = model_config
        self.data_manager = data_manager
        self.device = model_config.DEVICE

    def grid_search(
        self, param_grid: Dict, n_folds: int = 5
    ) -> LeNetConfig | LSTMConfig:
        """Perform grid search with cross-validation."""
        if (
            self.data_manager.train_loader is None
            or self.data_manager.val_loader is None
        ):
            raise ValueError("Training and validation datasets not loaded")

        best_config = None
        best_val_loss = float("inf")

        # Generate parameter combinations
        param_combinations = [
            dict(zip(param_grid.keys(), v))
            for v in itertools.product(*param_grid.values())
        ]

        # Single progress bar for combinations
        with tqdm(total=len(param_combinations), desc="Grid Search Progress") as pbar:
            for params in param_combinations:
                tqdm.write(f"\nTesting parameters: {params}")

                try:
                    # Create new config for this parameter set
                    current_config = dataclasses.replace(
                        self.config, **{k.upper(): v for k, v in params.items()}
                    )

                    # Train and evaluate model
                    model = ModelFactory.create_model(
                        config=current_config, vocab_size=self.data_manager.vocab_size
                    ).to(self.device)

                    trainer = CentralizedTrainer(model, current_config)

                    # Quick training for validation
                    results = trainer.train(
                        train_loader=self.data_manager.train_loader,
                        val_loader=self.data_manager.val_loader,
                        max_epochs=20,
                    )

                    val_loss = results["history"][-1]["val_loss"]

                    # Update best config
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_config = current_config
                        tqdm.write(f"New best config found! Val Loss: {val_loss:.4f}")

                    # Update progress
                    pbar.set_postfix(
                        {"best_val_loss": f"{best_val_loss:.4f}", "params": str(params)}
                    )
                    pbar.update()

                except Exception as e:
                    tqdm.write(f"Error testing parameters {params}: {str(e)}")
                    continue

        if best_config is None:
            raise ValueError("No valid configuration found during grid search")

        return best_config


class CentralizedTrainer:
    """Handles centralized model training."""

    def __init__(self, model: LeNet | CharLSTM, config: LeNetConfig | LSTMConfig):
        self.model: LeNet | CharLSTM = model
        self.config: LeNetConfig | LSTMConfig = config
        self.device: torch.device = config.DEVICE
        self.evaluator = ModelEvaluator(config)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        max_epochs: int = BaseConfig().NUM_EPOCHS,
        progress_bar: Optional[tqdm] = None,
    ) -> Dict:
        """Train the model with progress tracking."""
        self.model = self.model.to(self.device)
        # TODO: add correct optimizer for LSTM
        if isinstance(self.config, LSTMConfig):
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.LEARNING_RATE,
                weight_decay=1e-4,
            )
        else:
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.config.LEARNING_RATE,
                momentum=self.config.MOMENTUM,
                weight_decay=self.config.WEIGHT_DECAY,
            )

        criterion = nn.CrossEntropyLoss()

        avg_loss = 0.0
        train_acc = 0.0

        # Training tracking
        best_val_loss = float("inf")
        best_model_state = None
        results_history = []
        best_model_path = None
        # patience_counter = 0
        epoch = 0

        epoch_pbar = tqdm(
            total=max_epochs,
            desc=f"Training {self.model.__class__.__name__}",
            position=0,
            leave=True,
            colour="blue",
            unit="epoch",
        )

        batch_pbar = tqdm(
            total=len(train_loader),
            desc="Processing batches",
            position=1,
            leave=False,
            colour="green",
            unit="batch",
        )

        try:
            for epoch in range(max_epochs):
                self.model.train()
                train_loss = 0.0
                correct = 0
                total = 0

                batch_pbar.reset(total=len(train_loader))
                batch_pbar.set_description(f"Epoch {epoch + 1}/{max_epochs}")
                # Training phase
                for batch_idx, (inputs, targets) in enumerate(train_loader):
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    optimizer.zero_grad()
                    outputs = self.model(inputs)

                    if isinstance(self.model, CharLSTM):
                        outputs = outputs.view(-1, outputs.size(-1))
                        targets = targets.view(-1)

                    loss = criterion(outputs, targets)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=1.0
                    )
                    optimizer.step()

                    # Update metrics
                    with torch.no_grad():
                        train_loss += loss.item()
                        _, predicted = outputs.max(1)
                        total += targets.size(0)
                        correct += predicted.eq(targets).sum().item()

                    # Update progress bar
                    train_acc = 100.0 * correct / total
                    avg_loss = train_loss / (batch_idx + 1)
                    batch_pbar.set_postfix(
                        {"loss": f"{avg_loss:.3f}", "acc": f"{train_acc:.2f}%"}
                    )
                    batch_pbar.update()

                # Validation phase
                val_loss, val_acc = self.evaluator.evaluate_model(
                    self.model, val_loader
                )

                epoch_pbar.set_postfix(
                    {
                        "train_loss": f"{avg_loss:.3f}",
                        "train_acc": f"{train_acc:.2f}%",
                        "val_acc": f"{val_acc:.2f}%",
                    }
                )
                epoch_pbar.update()

                # Update metrics
                metrics = {
                    "epoch": epoch + 1,
                    "train_loss": avg_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                }
                results_history.append(metrics)

                # Model checkpointing
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = self.model.state_dict().copy()
                    # patience_counter = 0
                    best_model_path = save_checkpoint(
                        model=self.model,
                        optimizer=optimizer,
                        config=self.config,
                        epoch=epoch + 1,
                        val_loss=val_loss,
                        val_acc=val_acc,
                        is_best=True,
                    )

                    tqdm.write(f"New best model saved: {best_model_path.name}")

                # else:
                # patience_counter += 1

                # # Early stopping
                # if patience_counter >= 10:
                #     tqdm.write("Early stopping triggered")
                #     break

        except Exception as e:
            tqdm.write(f"Training error: {str(e)}")
            raise

        finally:
            if progress_bar is None:
                epoch_pbar.close()
                batch_pbar.close()

            # Restore best model
            if best_model_state is not None:
                self.model.load_state_dict(best_model_state)

        return {
            "history": results_history,
            "best_val_loss": best_val_loss,
            "epochs_trained": epoch + 1,
            "best_model_path": best_model_path,
        }


class FederatedTrainer:
    """Handles federated learning training."""

    def __init__(
        self,
        model: LeNet | CharLSTM,
        config: BaseConfig,
    ):
        self.model = model
        self.config = config
        self.device = config.DEVICE
        self.criterion = nn.CrossEntropyLoss()

    def train_client(
        self,
        client_model: nn.Module,
        train_loader: DataLoader,
        client_id: int,
        inner_pbar: tqdm,
    ) -> nn.Module:
        """Train a client model for several local epochs."""
        optimizer = torch.optim.AdamW(
            client_model.parameters(), lr=self.config.LEARNING_RATE
        )

        for epoch in range(self.config.LOCAL_EPOCHS):
            client_model.train()
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                optimizer.zero_grad()
                outputs = client_model(inputs)

                if isinstance(client_model, CharLSTM):
                    outputs = outputs.view(-1, outputs.size(-1))
                    targets = targets.view(-1)

                loss = self.criterion(outputs, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(client_model.parameters(), max_norm=0.5)
                optimizer.step()

                # Update progress bar
                inner_pbar.set_postfix(
                    {
                        "client": client_id,
                        "epoch": f"{epoch + 1}/{self.config.LOCAL_EPOCHS}",
                        "loss": f"{loss.item():.3f}",
                    }
                )
                inner_pbar.update()

        return client_model

    def aggregate_models(self, client_models: List[LeNet | CharLSTM]) -> None:
        """Aggregate client models using FedAvg."""
        global_state_dict = self.model.state_dict()

        # Initialize accumulator for each parameter
        accumulated = {}
        for key in global_state_dict.keys():
            accumulated[key] = torch.zeros_like(global_state_dict[key])

        # Sum up the parameters
        num_clients = len(client_models)
        for client_model in client_models:
            client_state = client_model.state_dict()
            for key in global_state_dict.keys():
                accumulated[key] += client_state[key] / num_clients

        # Update global model
        self.model.load_state_dict(accumulated)

    def train(
        self,
        client_data: List[DataLoader],
        val_loader: DataLoader,
        participation: Literal["uniform", "skewed"],
    ) -> Dict:
        """Run federated learning training."""
        # Setup progress tracking
        evaluator = ModelEvaluator(self.config)
        evaluator.setup_progress_bars(
            total_steps=self.config.FL_ROUNDS, desc="Federated Learning Progress"
        )

        if evaluator.outer_pbar is None or evaluator.inner_pbar is None:
            raise ValueError("Progress bars not initialized")

        # Training history
        history = []
        best_val_loss = float("inf")
        best_model_state = None

        try:
            for round_idx in range(self.config.FL_ROUNDS):
                # Distribute model to clients
                client_models = [
                    copy.deepcopy(self.model).to(self.device)
                    for _ in range(len(client_data))
                ]

                # Reset inner progress bar for client training
                total_batches = sum(len(loader) for loader in client_data)
                evaluator.inner_pbar.reset(
                    total=total_batches * self.config.LOCAL_EPOCHS
                )
                evaluator.inner_pbar.set_description(
                    f"Round {round_idx + 1} Client Training"
                )

                # Train each client
                for client_id, (client_model, client_loader) in enumerate(
                    zip(client_models, client_data)
                ):
                    self.train_client(
                        client_model=client_model,
                        train_loader=client_loader,
                        client_id=client_id + 1,
                        inner_pbar=evaluator.inner_pbar,
                    )

                # Aggregate client models
                self.aggregate_models(client_models)

                # Evaluate global model
                val_loss, val_acc = evaluator.evaluate_model(self.model, val_loader)

                # Update metrics
                metrics = {
                    "round": round_idx + 1,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                }
                history.append(metrics)

                # Model checkpointing
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = copy.deepcopy(self.model.state_dict())

                # Update progress bar
                evaluator.outer_pbar.set_postfix(
                    {"val_loss": f"{val_loss:.3f}", "val_acc": f"{val_acc:.2f}%"}
                )
                evaluator.outer_pbar.update()

        finally:
            # Clean up progress bars
            evaluator.outer_pbar.close()
            evaluator.inner_pbar.close()

        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        return {
            "history": history,
            "best_val_loss": best_val_loss,
            "rounds_trained": self.config.FL_ROUNDS,
        }


########################
# Evaluation
########################


class ModelEvaluator:
    """Handles model evaluation and progress tracking."""

    def __init__(self, config: BaseConfig):
        self.config = config
        self.device = config.DEVICE
        self.criterion = nn.CrossEntropyLoss()

        # Create main progress bar for epochs/rounds
        self.outer_pbar = None
        self.inner_pbar = None

    def setup_progress_bars(self, total_steps: int, desc: str):
        """Initialize nested progress bars."""
        # Main progress bar (epochs/rounds)
        self.outer_pbar = tqdm(
            total=total_steps,
            desc=desc,
            position=0,
            leave=True,
            colour="blue",
            unit="step",
        )

        # Inner progress bar (batches)
        self.inner_pbar = tqdm(
            desc="Batch Progress", position=1, leave=False, colour="green", unit="batch"
        )

    def evaluate_epoch(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        mode: str = "centralized",
    ) -> Dict:
        """Evaluate one training epoch."""
        model.train()
        total_train_loss = 0
        correct_train = 0
        total_train = 0

        if self.inner_pbar is None or self.outer_pbar is None:
            raise ValueError("Progress bars not initialized. Call setup_progress_bars.")

        # Update inner progress bar for batches
        self.inner_pbar.reset(total=len(train_loader))
        self.inner_pbar.set_description(f"Epoch {epoch + 1} ({mode})")

        # Training phase
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            optimizer.zero_grad()
            outputs = model(inputs)

            if isinstance(model, CharLSTM):
                outputs = outputs.view(-1, outputs.size(-1))
                targets = targets.view(-1)

            loss = self.criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Update metrics
            total_train_loss += loss.item()
            _, predicted = outputs.max(1)
            total_train += targets.size(0)
            correct_train += predicted.eq(targets).sum().item()

            # Update inner progress bar
            train_acc = 100.0 * correct_train / total_train
            self.inner_pbar.set_postfix(
                {"loss": f"{loss.item():.3f}", "acc": f"{train_acc:.2f}%"}
            )
            self.inner_pbar.update()

        # Validation phase
        val_loss, val_acc = self.evaluate_model(model, val_loader)

        # Update outer progress bar
        avg_train_loss = total_train_loss / len(train_loader)
        train_acc = 100.0 * correct_train / total_train

        metrics = {
            "mode": mode,
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        }

        status_str = f"[{mode.upper()}] "
        status_str += f"Train Loss: {avg_train_loss:.3f} "
        status_str += f"Train Acc: {train_acc:.2f}% "
        status_str += f"Val Acc: {val_acc:.2f}%"

        self.outer_pbar.set_description(status_str)
        self.outer_pbar.update()

        return metrics

    def evaluate_model(
        self, model: nn.Module, data_loader: DataLoader
    ) -> Tuple[float, float]:
        """Evaluate model on given dataset."""
        model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)

                if isinstance(model, CharLSTM):
                    outputs = outputs.view(-1, outputs.size(-1))
                    targets = targets.view(-1)

                loss = self.criterion(outputs, targets)
                total_loss += loss.item()

                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        avg_loss = total_loss / len(data_loader)
        accuracy = 100.0 * correct / total

        return avg_loss, accuracy

    def save_results(self, results: Dict, model_type: str, mode: str):
        """Save evaluation results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        filename = f"{mode}_{model_type}_results_{timestamp}.json"

        with open(self.config.RESULTS_DIR / filename, "w") as f:
            json.dump(results, f, indent=4)


def train_single_model(
    model_type: Literal["lenet", "lstm"],
    dataset_type: Optional[Literal["image", "text"]] = None,
    config_override: Optional[Dict] = None,
) -> Tuple[nn.Module, Dict]:
    """Train a single model with specified configuration."""

    # Windows-specific multiprocessing setup
    if sys.platform == "win32":
        torch.multiprocessing.set_start_method("spawn", force=True)

    try:
        # Create configs
        base_config = BaseConfig()
        if sys.platform == "win32":
            base_config = dataclasses.replace(base_config, NUM_WORKERS=1)

        # Setup directories
        for dir_path in [
            base_config.DATA_DIR,
            base_config.MODELS_DIR,
            base_config.RESULTS_DIR,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Set seeds
        torch.manual_seed(base_config.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(base_config.SEED)

        # Determine dataset type
        if dataset_type is None:
            dataset_type = "image" if model_type == "lenet" else "text"

        # Initialize data manager
        data_manager = DataManager(base_config)
        data_manager.load_and_split_data(dataset_type=dataset_type)

        if (
            data_manager.train_loader is None
            or data_manager.val_loader is None
            or data_manager.test_loader is None
        ):
            raise ValueError("Data loaders not properly initialized")

        # Create model config
        model_config = ConfigFactory.create_config(model_type)
        if config_override:
            model_config = dataclasses.replace(
                model_config, **{k.upper(): v for k, v in config_override.items()}
            )

        # Create model
        vocab_size: int | None = (
            len(data_manager.char_to_idx) if model_type == "lstm" else None
        )
        model: LeNet | CharLSTM = ModelFactory.create_model(
            config=model_config, vocab_size=vocab_size
        )

        # Train model
        trainer = CentralizedTrainer(model, model_config)

        results = trainer.train(
            train_loader=data_manager.train_loader,
            val_loader=data_manager.val_loader,
            max_epochs=model_config.NUM_EPOCHS,
        )

        # Evaluate
        evaluator = ModelEvaluator(model_config)
        test_loss, test_acc = evaluator.evaluate_model(model, data_manager.test_loader)
        results["test_metrics"] = {"loss": test_loss, "accuracy": test_acc}

        return model, results

    except Exception as e:
        logging.error(f"Error during training: {str(e)}")
        raise

    finally:
        # Cleanup CUDA memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main(use_saved_config: bool = True) -> None:
    # Setup
    base_config = BaseConfig()
    base_config.setup_directories()

    torch.manual_seed(base_config.SEED)

    logging.debug("Starting AML project")
    logging.debug(f"PyTorch version: {torch.__version__}")
    logging.debug(f"Torchvision version: {torchvision.__version__}")
    logging.debug(f"Using device: {base_config.DEVICE}")

    # Initialize data manager
    data_manager = DataManager(base_config)

    # Train and evaluate models
    for model_type in ["lstm", "lenet"]:
        # Load appropriate dataset based on model type
        dataset_type: Literal["image", "text"] = (
            "image" if model_type == "lenet" else "text"
        )
        data_manager.load_and_split_data(dataset_type=dataset_type)
        if (
            data_manager.train_loader is None
            or data_manager.val_loader is None
            or data_manager.test_loader is None
        ):
            raise ValueError("Data not loaded. Call load_and_split_data first.")

        data_manager.load_and_split_data(dataset_type=dataset_type)

        # Create initial model config
        hyperparams = None
        if use_saved_config:
            hyperparams = load_hyperparameters(base_config.CONFIGS_DIR, model_type)
            if hyperparams:
                logging.info(f"Loaded saved hyperparameters for {model_type}")

        model_config = ConfigFactory.create_config(
            model_type, **(hyperparams.to_dict() if hyperparams else {})
        )

        # Hyperparameter optimization
        param_grid = {
            "learning_rate": [0.001, 0.01],
            "batch_size": [32, 64],
        }
        # Remove None values from param_grid
        param_grid = {k: v for k, v in param_grid.items() if v is not None}

        hp_tester = HyperparameterTester(model_config, data_manager)
        best_config = hp_tester.grid_search(param_grid)
        logging.info(f"Best config found: {best_config}")

        # Save best hyperparameters
        hyperparams = (
            LeNetHyperParameters if model_type == "lenet" else LSTMHyperParameters
        )(
            **{
                k.lower(): v
                for k, v in dataclasses.asdict(best_config).items()
                if k.lower() in HyperParameters.__dataclass_fields__
            }
        )
        save_hyperparameters(hyperparams, base_config.CONFIGS_DIR, model_type)

        # Create model with appropriate vocab_size
        vocab_size = len(data_manager.char_to_idx) if model_type == "lstm" else 0
        model = ModelFactory.create_model(
            config=best_config,
            vocab_size=vocab_size,
        )

        trainer = CentralizedTrainer(model, best_config)
        central_results = trainer.train(
            data_manager.train_loader, data_manager.val_loader
        )

        # Federated Training
        fed_trainer = FederatedTrainer(model, base_config)
        federated_results = fed_trainer.train(
            data_manager.get_client_loaders(),
            data_manager.val_loader,
            # TODO: for now this does nothing
            participation="uniform",
        )

        # Evaluation
        evaluator = ModelEvaluator(base_config)
        central_metrics = evaluator.evaluate_model(model, data_manager.test_loader)
        fed_metrics = evaluator.evaluate_model(
            fed_trainer.model, data_manager.test_loader
        )

        # Save results
        results = {
            "model_type": model_type,
            "centralized": central_metrics,
            "federated": fed_metrics,
            "training_history": {
                "centralized": central_results,
                "federated": federated_results,
            },
        }

        with open(base_config.RESULTS_DIR / f"{model_type}_results.json", "w") as f:
            json.dump(results, f, indent=4)


if __name__ == "__main__":
    if sys.platform == "win32":
        torch.multiprocessing.freeze_support()

    setup_logging()
    try:
        model, results = train_single_model("lenet")
        # main()

    except Exception as e:
        logging.error(f"Training failed: {str(e)}")
