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

    # Common Training Parameters
    BATCH_SIZE: int = 64
    LEARNING_RATE: float = 0.001
    NUM_EPOCHS: int = 20

    # Federated Learning
    NUM_CLIENTS: int = 5
    FL_ROUNDS: int = 20
    LOCAL_EPOCHS: int = 5

    def setup_directories(self):
        """Create necessary directories."""
        for dir_path in [self.DATA_DIR, self.MODELS_DIR, self.RESULTS_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class LSTMConfig(BaseConfig):
    """LSTM specific configuration."""

    MODEL_TYPE: Literal["lstm"] = "lstm"
    HIDDEN_SIZE: int = 256
    EMBEDDING_SIZE: int = 128
    NUM_LAYERS: int = 2
    DROPOUT: float = 0.2
    BIDIRECTIONAL: bool = True
    ATTENTION_HEADS: int = 2


@dataclass(frozen=True)
class CNNConfig(BaseConfig):
    """CNN specific configuration."""

    MODEL_TYPE: Literal["cnn"] = "cnn"
    NUM_CLASSES: int = 100
    BATCH_SIZE: int = 64
    LEARNING_RATE: float = 0.01
    WEIGHT_DECAY: float = 4e-4
    MOMENTUM: float = 0.9


class ConfigFactory:
    """Factory for creating appropriate config objects."""

    @staticmethod
    def validate_config(config: Union[CNNConfig, LSTMConfig]) -> None:
        """Validate configuration parameters."""
        if isinstance(config, LSTMConfig):
            if config.DROPOUT > 0 and config.NUM_LAYERS < 2:
                raise ValueError(
                    f"Invalid LSTM configuration: dropout={config.DROPOUT} requires "
                    f"num_layers > 1, but got num_layers={config.NUM_LAYERS}"
                )

    @staticmethod
    def create_config(model_type: str, **kwargs) -> Union[CNNConfig, LSTMConfig]:
        """Create a config instance based on model type."""
        if model_type.lower() == "cnn":
            config = CNNConfig(**kwargs)
        elif model_type.lower() == "lstm":
            config = LSTMConfig(**kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        ConfigFactory.validate_config(config)
        return config

    @staticmethod
    def create_from_dict(config_dict: dict) -> Union[CNNConfig, LSTMConfig]:
        """Create a config instance from a dictionary."""
        model_type = config_dict.pop("model_type", None)
        if not model_type:
            raise ValueError("model_type must be specified in config dictionary")
        return ConfigFactory.create_config(model_type, **config_dict)


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
    """Dataset class for Shakespeare text data."""

    def __init__(self, sequences: List[Tuple[str, str]], char_to_idx: Dict[str, int]):
        self.sequences = sequences
        self.char_to_idx = char_to_idx
        self.idx_to_char = {idx: char for char, idx in char_to_idx.items()}

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        input_seq, target_seq = self.sequences[index]

        # Convert characters to indices
        input_tensor = torch.tensor(
            [self.char_to_idx[c] for c in input_seq], dtype=torch.long
        )
        target_tensor = torch.tensor(
            [self.char_to_idx[c] for c in target_seq], dtype=torch.long
        )

        return input_tensor, target_tensor


class DataManager:
    """Handles data loading, splitting, and preparation for both text and image data."""

    def __init__(self, config: BaseConfig):
        self.config = config
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.char_to_idx = {}
        self.vocab_size: int = 0
        self.dataset_type: Literal["text"] | Literal["image"] = "text"
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

    def load_and_split_data(
        self,
        dataset_type: Literal["text"] | Literal["image"] = "text",
        val_split: float = 0.1,
        test_split: float = 0.1,
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
            shakespeare_file = self.config.DATA_DIR / "shakespeare.txt"

            # Download Shakespeare dataset if not exists
            if not shakespeare_file.exists():
                raise FileNotFoundError(
                    f"Shakespeare dataset not found. Please download and place it in the `{self.config.DATA_DIR}` as `shakespeare.txt`"
                )

            # Read and process text
            with open(shakespeare_file, "r", encoding="utf-8") as f:
                text = f.read()

            # Create character mappings
            chars = sorted(list(set(text)))
            self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
            self.vocab_size = len(chars)

            # Create sequences
            sequence_length = 100
            sequences = []
            for i in range(0, len(text) - sequence_length - 1, sequence_length):
                input_seq = text[i : i + sequence_length]
                target_seq = text[i + 1 : i + sequence_length + 1]
                sequences.append((input_seq, target_seq))

            # Create dataset
            full_dataset = ShakespeareDataset(sequences, self.char_to_idx)

            # Split dataset
            total_size = len(full_dataset)
            test_size = int(test_split * total_size)
            val_size = int(val_split * total_size)
            train_size = total_size - test_size - val_size

            self.train_dataset, self.val_dataset, self.test_dataset = random_split(
                full_dataset, [train_size, val_size, test_size]
            )

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


class CNN(nn.Module):
    """CNN Model as described in the paper with two 5x5 conv layers and two FC layers."""

    def __init__(self, config: CNNConfig):
        super().__init__()

        # First convolutional block
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Second convolutional block
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Calculate size after convolutions and pooling
        # Input: 32x32 → Conv1+Pool1: 16x16 → Conv2+Pool2: 8x8
        feature_size = 8 * 8 * 64

        # Fully connected layers
        self.fc1 = nn.Linear(feature_size, 384)
        self.fc2 = nn.Linear(384, 192)
        self.classifier = nn.Linear(192, config.NUM_CLASSES)

        # Activation
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # First conv block
        x = self.conv1(x)
        x = self.softmax(x)
        x = self.pool1(x)

        # Second conv block
        x = self.conv2(x)
        x = self.softmax(x)
        x = self.pool2(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = self.fc1(x)
        x = self.softmax(x)
        x = self.fc2(x)
        x = self.softmax(x)
        x = self.classifier(x)

        return x


class LSTM(nn.Module):
    """LSTM model for sequence processing."""

    def __init__(self, config: LSTMConfig, vocab_size: int) -> None:
        super().__init__()
        self.vocab_size: int = vocab_size
        self.embedding = nn.Embedding(vocab_size, config.HIDDEN_SIZE // 2)
        self.emb_dropout = nn.Dropout(config.DROPOUT)
        self.lstm_dropout = nn.Dropout(config.DROPOUT)
        self.layer_norm1 = nn.LayerNorm(config.HIDDEN_SIZE // 2)
        self.layer_norm2 = nn.LayerNorm(config.HIDDEN_SIZE * 2)

        self.lstm = nn.LSTM(
            config.HIDDEN_SIZE // 2,
            config.HIDDEN_SIZE,
            config.NUM_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=config.DROPOUT,
        )
        # Account for bidirectional
        self.attention = nn.MultiheadAttention(
            config.HIDDEN_SIZE * 2,
            num_heads=config.ATTENTION_HEADS,
            dropout=config.DROPOUT,
        )

        # Output layers
        self.fc1 = nn.Linear(config.HIDDEN_SIZE * 2, config.HIDDEN_SIZE)
        self.fc2 = nn.Linear(config.HIDDEN_SIZE, vocab_size)
        self.activation = nn.GELU()

    def forward(self, x, hidden=None):
        # x shape: [batch_size, seq_length]
        batch_size, seq_length = x.size()

        # Embedding layer with dropout
        embedded = self.embedding(x)  # [batch_size, seq_length, embedding_dim]
        embedded = self.layer_norm1(embedded)
        embedded = self.emb_dropout(embedded)

        # LSTM layers
        lstm_out, (hidden, cell) = self.lstm(embedded, hidden)
        # lstm_out shape: [batch_size, seq_length, hidden_dim * 2]
        # Apply attention mechanism
        attn_out, _ = self.attention(
            lstm_out.transpose(0, 1), lstm_out.transpose(0, 1), lstm_out.transpose(0, 1)
        )
        attn_out = attn_out.transpose(0, 1)

        # Residual connection and layer normalization
        lstm_out = self.layer_norm2(lstm_out + attn_out)
        lstm_out = self.lstm_dropout(lstm_out)

        # Fully connected layers
        out = self.activation(self.fc1(lstm_out))
        out = self.fc2(out)  # [batch_size, seq_length, vocab_size]

        return out

    def generate(
        self, initial_sequence, max_length=100, temperature=1.0, device="cuda"
    ):
        self.eval()
        with torch.no_grad():
            current_seq = initial_sequence.to(device)
            hidden = None
            generated = []

            for _ in range(max_length):
                # Get predictions
                output = self(current_seq, hidden)
                predictions = output[:, -1, :] / temperature

                # Sample from the distribution
                probs = torch.softmax(predictions, dim=-1)
                next_char = torch.multinomial(probs, 1)

                generated.append(next_char.item())
                current_seq = torch.cat([current_seq, next_char], dim=1)

        return generated


class ModelFactory:
    """Factory for creating models."""

    @staticmethod
    def create_model(
        config: Union[CNNConfig, LSTMConfig], vocab_size: int | None
    ) -> CNN | LSTM:
        """Create a model instance based on config type."""
        if isinstance(config, CNNConfig):
            return CNN(config)
        elif isinstance(config, LSTMConfig):
            if vocab_size is None:
                raise ValueError("vocab_size is required for LSTM models")
            return LSTM(config, vocab_size)
        raise ValueError(f"Unsupported config type: {type(config)}")


########################
# Training Framework
########################


class HyperparameterTester:
    """Handles hyperparameter optimization."""

    def __init__(self, config: BaseConfig, data_manager: DataManager) -> None:
        self.config = config
        self.data_manager = data_manager
        self.device = config.DEVICE

    def grid_search(self, param_grid: Dict, model_type: str, n_folds: int = 5) -> Dict:
        best_params = {}
        best_val_loss = float("inf")

        # Generate all combinations of parameters
        param_combinations = [
            dict(zip(param_grid.keys(), v))
            for v in itertools.product(*param_grid.values())
        ]

        if (
            self.data_manager.train_dataset is None
            or self.data_manager.train_loader is None
            or self.data_manager.val_dataset is None
            or self.data_manager.val_loader is None
        ):
            raise ValueError("Data not loaded. Call load_and_split_data first.")

        # Create progress bar for parameter combinations
        grid_search_pbar = tqdm(
            total=len(param_combinations),
            desc="Grid Search Progress",
            position=0,
            leave=True,
            colour="blue",
            unit="config",
        )

        # Create progress bar for folds
        fold_pbar = tqdm(
            total=n_folds,
            desc="Cross Validation",
            position=1,
            leave=False,
            colour="green",
            unit="fold",
        )

        # Create progress bar for batches
        batch_pbar = tqdm(
            desc="Training Batches",
            position=2,
            leave=False,
            colour="yellow",
            unit="batch",
        )

        try:
            for params in param_combinations:
                tqdm.write(f"\nTesting parameters: {params}")

                current_config = ConfigFactory.create_config(
                    model_type, **{k.upper(): v for k, v in params.items()}
                )

                fold_val_losses = []
                fold_pbar.reset()

                for fold in range(n_folds):
                    fold_pbar.set_description(f"Fold {fold + 1}/{n_folds}")

                    model = ModelFactory.create_model(
                        config=current_config, vocab_size=self.data_manager.vocab_size
                    ).to(self.device)

                    trainer = CentralizedTrainer(model, current_config)

                    # Update batch progress bar total
                    batch_pbar.reset(
                        total=len(self.data_manager.train_loader) * 5
                    )  # 5 epochs for CV

                    results = trainer.train(
                        train_loader=self.data_manager.train_loader,
                        val_loader=self.data_manager.val_loader,
                        max_epochs=5,
                        progress_bar=batch_pbar,
                    )

                    if isinstance(results, dict) and "history" in results:
                        val_loss = results["history"][-1]["val_loss"]
                        fold_val_losses.append(val_loss)
                        fold_pbar.set_postfix({"val_loss": f"{val_loss:.4f}"})
                    else:
                        raise ValueError(
                            "Trainer did not return expected metrics format"
                        )

                    fold_pbar.update()

                # Calculate average validation loss across folds
                avg_val_loss = sum(fold_val_losses) / len(fold_val_losses)

                # Update best parameters if better
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_params = params
                    tqdm.write(
                        f"New best parameters found: {best_params} (val_loss: {best_val_loss:.4f})"
                    )

                grid_search_pbar.set_postfix(
                    {
                        "best_val_loss": f"{best_val_loss:.4f}",
                        "best_params": str(best_params),
                    }
                )
                grid_search_pbar.update()

        except Exception as e:
            tqdm.write(f"Error during grid search: {str(e)}")
            raise

        finally:
            # Clean up progress bars
            grid_search_pbar.close()
            fold_pbar.close()
            batch_pbar.close()

        if not best_params:
            raise ValueError("No valid parameters found during grid search")

        return best_params


class CentralizedTrainer:
    """Handles centralized model training."""

    def __init__(self, model: CNN | LSTM, config: CNNConfig | LSTMConfig):
        self.model: CNN | LSTM = model
        self.config: CNNConfig | LSTMConfig = config
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
            logging.info(f"SGD optimizer with lr={self.config.LEARNING_RATE}")

        criterion = nn.CrossEntropyLoss()

        avg_loss = 0.0
        train_acc = 0.0

        # Training tracking
        best_val_loss = float("inf")
        best_model_state = None
        results_history = []
        best_model_path = None
        patience_counter = 0
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

                    if isinstance(self.model, LSTM):
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
                    patience_counter = 0
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

                else:
                    patience_counter += 1

                # Early stopping
                if patience_counter >= 10:
                    tqdm.write("Early stopping triggered")
                    break

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

    def __init__(self, model: CNN | LSTM, config: BaseConfig):
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

                if isinstance(client_model, LSTM):
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

    def aggregate_models(self, client_models: List[CNN | LSTM]) -> None:
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

    def train(self, client_data: List[DataLoader], val_loader: DataLoader) -> Dict:
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

            if isinstance(model, LSTM):
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

                if isinstance(model, LSTM):
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
    model_type: Literal["cnn", "lstm"],
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
            dataset_type = "image" if model_type == "cnn" else "text"

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
        model: CNN | LSTM = ModelFactory.create_model(
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


def main():
    # Setup

    base_config = BaseConfig()
    base_config.setup_directories()
    torch.manual_seed(base_config.SEED)

    logging.info("Starting AML project")
    logging.info(f"PyTorch version: {torch.__version__}")
    logging.info(f"Torchvision version: {torchvision.__version__}")
    logging.info(f"Using device: {base_config.DEVICE}")

    # Initialize components
    data_manager = DataManager(base_config)
    data_manager.load_and_split_data()
    # Hyperparameter optimization
    param_grid = {
        "learning_rate": [0.001, 0.01],
        "hidden_size": [128, 256],
        "num_layers": [2, 3],
    }

    hp_tester = HyperparameterTester(base_config, data_manager)
    hp_tester = HyperparameterTester(base_config, data_manager)
    best_params = hp_tester.grid_search(param_grid, model_type="lstm")

    # Update config with best parameters
    for param, value in best_params.items():
        setattr(base_config, param.upper(), value)

    # Train and evaluate models
    for model_type in ["cnn", "lstm"]:
        # Load appropriate dataset based on model type
        dataset_type = "image" if model_type == "cnn" else "text"
        data_manager.load_and_split_data(dataset_type=dataset_type)
        if (
            data_manager.train_loader is None
            or data_manager.val_loader is None
            or data_manager.test_loader is None
        ):
            raise ValueError("Data not loaded. Call load_and_split_data first.")

        # Create specific config for the model type
        config = ConfigFactory.create_config(model_type)

        # Create model with appropriate vocab_size
        vocab_size = len(data_manager.char_to_idx) if model_type == "lstm" else 0
        model = ModelFactory.create_model(
            config=config,
            vocab_size=vocab_size,
        )

        trainer = CentralizedTrainer(model, config)
        central_results = trainer.train(
            data_manager.train_loader, data_manager.val_loader
        )

        # Federated Training
        fed_trainer = FederatedTrainer(model, base_config)
        federated_results = fed_trainer.train(
            data_manager.get_client_loaders(), data_manager.val_loader
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
        model, results = train_single_model("cnn")
        tqdm.write(
            f"Training completed. Test accuracy: {results['test_metrics']['accuracy']:.2f}%"
        )
    except Exception as e:
        tqdm.write(f"Training failed: {str(e)}")
