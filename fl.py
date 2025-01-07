########################
# Imports and Setup
########################
import copy
import logging
import os
import shutil
import sys
from datetime import datetime
from os import PathLike
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import adabelief_pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision import datasets
from torchvision.models import resnet18
from tqdm import tqdm


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


########################
# Constants
########################
DEVICE_TYPE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = torch.device(DEVICE_TYPE)
RESULTS_DIR = Path("results")
MODELS_DIR = Path("models")
CONFIGS = {
    "dataset": "cifar100",
    "architecture": "modern_cnn",
    "batch_size": 128,
    "base_lr": 0.01,
    "epochs": 20,
    "fl_rounds": 200,
    "fl_epochs": 20,
    "client_configs": [
        {"num_clients": 2, "iid": True},
        {"num_clients": 5, "iid": True},
        {"num_clients": 10, "iid": True},
        {"num_clients": 5, "iid": False},
    ],
}

MIN_WORKERS = 2

########################
# Dataset Management
########################


class ShakespeareDataset(Dataset):
    """Dataset per gestire il caricamento dei dati di Shakespeare."""

    def __init__(self, sequences, char_to_idx):
        """
        sequences: lista di tuple (input_sequence, target_sequence)
        char_to_idx: dizionario per convertire caratteri in indici.
        """
        self.sequences = sequences
        self.char_to_idx = char_to_idx

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        input_seq, target_seq = self.sequences[index]
        input_tensor = torch.tensor(
            [self.char_to_idx[char] for char in input_seq], dtype=torch.long
        )
        target_tensor = torch.tensor(
            [self.char_to_idx[char] for char in target_seq], dtype=torch.long
        )
        return input_tensor, target_tensor


def get_cifar100_dataloader(batch_size=32, iid=True):
    """Load CIFAR-100 dataset with data augmentation."""
    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    dataset = datasets.CIFAR100(
        root="./data", train=True, download=True, transform=transform
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=iid, drop_last=True)


def get_shakespeare_dataloader(batch_size=32):
    """Simple text dataset loader without torchtext dependency."""
    try:
        # Load text file (placeholder - you'll need to provide the actual file)
        text_path = "./data/shakespeare.txt"
        with open(text_path, "r", encoding="utf-8") as f:
            text = f.read()

        # Simple character-level tokenization
        chars = sorted(list(set(text)))
        char_to_idx = {ch: i for i, ch in enumerate(chars)}

        # Create sequences
        sequence_length = 100
        sequences = []
        for i in range(0, len(text) - sequence_length, sequence_length):
            sequence = text[i : i + sequence_length]
            target = text[i + 1 : i + sequence_length + 1]

            # Convert to tensors
            input_tensor = torch.tensor(
                [char_to_idx[c] for c in sequence], dtype=torch.long
            )
            target_tensor = torch.tensor(
                [char_to_idx[c] for c in target], dtype=torch.long
            )

            sequences.append((input_tensor, target_tensor))

        dataset = ShakespeareDataset(sequences, char_to_idx)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    except FileNotFoundError:
        print("Shakespeare dataset file not found. Using dummy data instead.")
        # Return dummy data if file not found
        dummy_data = [
            (torch.randint(0, 100, (100,)), torch.randint(0, 100, (100,)))
            for _ in range(1000)
        ]
        dummy_char_to_idx = {chr(i): i for i in range(100)}  # Create dummy mapping
        dummy_dataset = ShakespeareDataset(dummy_data, dummy_char_to_idx)
        return DataLoader(dummy_dataset, batch_size=batch_size, shuffle=True)


def preprocess_shakespeare(file_path, sequence_length=100):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    chars = sorted(set(text))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for ch, i in char_to_idx.items()}

    sequences = []
    for i in range(len(text) - sequence_length):
        seq = text[i : i + sequence_length]
        target = text[i + 1 : i + sequence_length + 1]
        sequences.append((seq, target))

    return sequences, char_to_idx, idx_to_char


def split_dataset(dataset, num_clients):
    """Split a dataset into equal parts for federated learning."""
    partition_size = len(dataset) // num_clients
    lengths = [partition_size] * num_clients
    return random_split(dataset, lengths)


def save_in_models_folder(
    model: nn.Module,
    model_name: str,
    with_timestamp: bool = True,
    override: bool = False,
    feedback: bool = True,
) -> None:
    """Save the model state dict to a file."""
    os.makedirs("models", exist_ok=True)
    # Remove .pth extension if present at the end
    if model_name.endswith(".pth"):
        model_name = model_name[:-4]
    path = os.path.join("models", model_name)

    if with_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"{path}_{timestamp}.pth"
    else:
        path = f"{path}.pth"

    if not override and os.path.exists(path):
        index = 1
        base_path = path[:-4]
        while os.path.exists(path):
            path = f"{base_path}_{index}.pth"
            index += 1

    torch.save(model.state_dict(), path)
    if feedback:
        print(f"Global model saved as {path}")


def get_shakespeare_client_datasets(
    file_path="./data/shakespeare.txt", batch_size=32, num_clients=5, iid=True
):
    """Load the Shakespeare dataset and split it for clients."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        # Tokenizzazione caratteri
        chars = sorted(list(set(text)))
        char_to_idx = {ch: i for i, ch in enumerate(chars)}
        idx_to_char = {i: ch for ch, i in char_to_idx.items()}

        # Dividi il testo in sequenze
        sequence_length = 100
        sequences = []
        for i in range(0, len(text) - sequence_length, sequence_length):
            seq = text[i : i + sequence_length]
            target = text[i + 1 : i + sequence_length + 1]
            sequences.append((seq, target))

        # Suddividi in training e test (80% training, 20% test)
        split_idx = int(0.8 * len(sequences))
        train_data = sequences[:split_idx]
        test_data = sequences[split_idx:]

        # Suddividi i dati tra i client
        client_train_datasets = []
        partition_size = len(train_data) // num_clients
        for i in range(num_clients):
            client_train = train_data[i * partition_size : (i + 1) * partition_size]
            client_train_datasets.append(
                DataLoader(
                    ShakespeareDataset(client_train, char_to_idx),
                    batch_size=batch_size,
                    shuffle=True,
                )
            )

        # Crea un DataLoader per il test set
        test_loader = DataLoader(
            ShakespeareDataset(test_data, char_to_idx),
            batch_size=batch_size,
            shuffle=False,
        )

        return client_train_datasets, test_loader, idx_to_char
    except FileNotFoundError:
        print(f"File {file_path} not found!")
        return [], None, {}


########################
# Model Architectures
########################
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
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Predict the last character in the sequence
        return out


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


class NewModernCNN(nn.Module):
    """Improved CNN architecture with modern components."""

    class SEBlock(nn.Module):
        """Squeeze-and-Excitation block."""

        def __init__(self, channels, reduction=16):
            super().__init__()
            self.squeeze = nn.AdaptiveAvgPool2d(1)
            self.excitation = nn.Sequential(
                nn.Linear(channels, channels // reduction, bias=False),
                nn.SiLU(inplace=True),
                nn.Linear(channels // reduction, channels, bias=False),
                nn.Sigmoid(),
            )

        def forward(self, x):
            b, c, _, _ = x.size()
            squeeze = self.squeeze(x).view(b, c)
            excitation = self.excitation(squeeze).view(b, c, 1, 1)
            return x * excitation.expand_as(x)

    class ConvBlock(nn.Module):
        """Enhanced convolution block with residual connection."""

        def __init__(self, in_channels, out_channels, stride=1):
            super().__init__()
            self.conv1 = nn.Conv2d(
                in_channels, out_channels, 3, stride=stride, padding=1, bias=False
            )
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_channels)
            self.se = NewModernCNN.SEBlock(out_channels)
            self.act = nn.SiLU(inplace=True)

            # Skip connection
            self.shortcut = nn.Sequential()
            if stride != 1 or in_channels != out_channels:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_channels),
                )

        def forward(self, x):
            out = self.act(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out = self.se(out)
            out += self.shortcut(x)
            return self.act(out)

    def __init__(self, num_classes=100):
        super().__init__()

        # Initial convolution with larger kernel
        self.entry = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
        )

        # Progressive feature scaling
        self.stage1 = self.ConvBlock(32, 64, stride=2)
        self.stage2 = self.ConvBlock(64, 128, stride=2)
        self.stage3 = self.ConvBlock(128, 256, stride=2)

        # Global pooling and classifier
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(256, 1024),
            nn.BatchNorm1d(1024),
            nn.SiLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, num_classes),
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.entry(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def build_model(architecture="cnn", num_classes=100):
    """Builds and returns the specified model architecture."""
    if architecture.lower() == "cnn":
        return ModernCNN(num_classes=num_classes)
        return CNN(num_classes=num_classes)
    elif architecture.lower() == "resnet":
        return resnet18(pretrained=False, num_classes=num_classes)
    elif architecture.lower() == "lstm":
        return LSTM(
            vocab_size=num_classes, embedding_dim=512, hidden_dim=1024, num_layers=3
        )  # Use num_classes as vocab_size
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")


########################
# Training Utilities
########################
class ImprovedTraining:
    def __init__(
        self,
        model: nn.Module,
        trainloader: DataLoader,
        device: torch.device,
        epochs: int,
        num_classes=100,
        lr=1e-3,
        weight_decay=0.05,
        beta1=0.9,
        beta2=0.999,
    ) -> None:
        self.model: nn.Module = model
        self.trainloader = trainloader
        self.device: torch.device = device
        self.device_name: str = "cuda" if device.type == "cuda" else "cpu"
        self.scaler = torch.amp.grad_scaler.GradScaler(device=self.device_name)

        if len(trainloader) == 0:
            raise ValueError(
                "Training dataloader is empty. Please check your dataset and batch size."
            )

        # Improved optimizer configuration
        self.optimizer = adabelief_pytorch.AdaBelief(
            model.parameters(),
            lr=lr,
            eps=1e-8,
            betas=(0.9, 0.999),
            weight_decouple=False,
            rectify=False,
            fixed_decay=False,
            amsgrad=False,
            print_change_log=False,
        )

        # One Cycle scheduler with warmup
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=epochs,
        )

        # Mixup augmentation
        self.mixup = Mixup(
            mixup_alpha=0.8,
            cutmix_alpha=1.0,
            prob=0.5,
            switch_prob=0.5,
            mode="batch",
            num_classes=num_classes,
        )

    def train_epoch(self, pbar_position: int) -> tuple[float, float]:
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        epoch_pbar = tqdm(
            enumerate(self.trainloader),
            total=len(self.trainloader),
            position=pbar_position,
            desc="Training",
            unit="batch",
            leave=False,
            colour="yellow",
            dynamic_ncols=True,
        )

        for batch_idx, (inputs, targets) in epoch_pbar:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # Apply mixup augmentation
            inputs, targets_a, targets_b, lam = self.mixup(inputs, targets)

            # Use automatic mixed precision
            with torch.amp.autocast_mode.autocast(device_type=self.device_name):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets_a) * lam + self.criterion(
                    outputs, targets_b
                ) * (1 - lam)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

            # Update learning rate
            self.scheduler.step()

            with torch.no_grad():
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        return total_loss / len(self.trainloader), 100.0 * correct / total

    def criterion(self, outputs, targets):
        # Label smoothing cross entropy
        return nn.CrossEntropyLoss(label_smoothing=0.1)(outputs, targets)


class Mixup:
    def __init__(
        self,
        mixup_alpha=1.0,
        cutmix_alpha=1.0,
        prob=0.5,
        switch_prob=0.5,
        mode="batch",
        num_classes=100,
    ):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.mix_prob = prob
        self.switch_prob = switch_prob
        self.mode = mode
        self.num_classes = num_classes
        self.mixup_enabled = True

    def __call__(self, x, target):
        if not self.mixup_enabled or np.random.rand() > self.mix_prob:
            return x, target, target, 1.0

        if np.random.rand() < self.switch_prob:
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            # Mixup
            rand_index = torch.randperm(x.size()[0]).to(x.device)
            target_a = target
            target_b = target[rand_index]
            mixed_x = lam * x + (1 - lam) * x[rand_index, :]
            return mixed_x, target_a, target_b, lam
        else:
            # CutMix
            lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
            rand_index = torch.randperm(x.size()[0]).to(x.device)
            target_a = target
            target_b = target[rand_index]
            bbx1, bby1, bbx2, bby2 = self._rand_bbox(x.size(), lam)
            x[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
            return x, target_a, target_b, lam

    def _rand_bbox(self, size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2


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


########################
# Centralized Training
########################


def train_model(model, dataloader, criterion, optimizer, epoch, total_epochs):
    """Training loop for one epoch with progress bars"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    # Create progress bar for batches
    batch_pbar = tqdm(
        enumerate(dataloader),
        total=len(dataloader),
        desc=f"Training Epoch {epoch+1}/{total_epochs}",
        unit="batch",
        leave=False,
        position=1,
        colour="yellow",
        dynamic_ncols=True,
    )

    for batch_idx, (inputs, labels) in batch_pbar:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Statistics
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Update progress bar
        accuracy = 100.0 * correct / total
        batch_pbar.set_postfix(
            {"loss": f"{loss.item():.3f}", "acc": f"{accuracy:.2f}%"}
        )

    return total_loss / len(dataloader), 100.0 * correct / total


def centralized_training(
    dataset="cifar100", architecture="cnn", epochs=10, batch_size=32, lr=0.01
):
    """Centralized training with specified configuration and progress bars"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Create main progress bar for epochs
    epochs_pbar = tqdm(
        range(epochs),
        desc="Centralized Training Progress",
        unit="epoch",
        position=0,
        colour="blue",
        leave=True,
        dynamic_ncols=True,
    )

    # Data loading
    if dataset.lower() == "cifar100":
        transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        trainset = torchvision.datasets.CIFAR100(
            root="./data", train=True, download=True, transform=transform
        )
        trainloader = DataLoader(
            trainset, batch_size=batch_size, shuffle=True, num_workers=2
        )

        testset = torchvision.datasets.CIFAR100(
            root="./data", train=False, download=True, transform=transform
        )
        testloader = DataLoader(
            testset, batch_size=batch_size, shuffle=False, num_workers=2
        )
    else:
        raise ValueError(f"Dataset {dataset} not supported")

    # Model initialization
    model = build_model(architecture, num_classes=100).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_accuracy = 0.0

    # Training loop
    for epoch in epochs_pbar:
        train_loss, train_accuracy = train_model(
            model, trainloader, criterion, optimizer, epoch, epochs
        )

        # Evaluate on test set
        test_accuracy = evaluate_model(model, testloader)

        # Update schedulers
        scheduler.step()

        # Update progress bar
        epochs_pbar.set_postfix(
            {
                "train_loss": f"{train_loss:.3f}",
                "train_acc": f"{train_accuracy:.2f}%",
                "test_acc": f"{test_accuracy:.2f}%",
            }
        )

        # Save best model
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            torch.save(model.state_dict(), f"best_{architecture}_{dataset}.pth")

    # Save final model
    torch.save(model.state_dict(), f"final_{architecture}_{dataset}.pth")
    logging.info(f"Training completed! Best accuracy: {best_accuracy:.2f}%")

    return model


########################
# Federated Learning
########################
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
            f"Learning Rate: {config['lr']}\n",
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
        # val_size = len(indices) - train_size

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


def run_research_experiment():
    """Execute complete research experiment comparing centralized vs federated."""

    # 1. First run centralized training as baseline
    logging.info("Starting centralized training baseline...")
    centralized_model = centralized_training(
        dataset=CONFIGS["dataset"],
        architecture=CONFIGS["architecture"],
        epochs=CONFIGS["epochs"],
        batch_size=CONFIGS["batch_size"],
        lr=CONFIGS["base_lr"],
    )

    # 2. Run federated experiments with different client configurations
    for client_config in CONFIGS["client_configs"]:
        logging.info(f"\nStarting federated learning with {client_config}...")
        fl_model = federated_learning(
            num_clients=client_config["num_clients"],
            rounds=CONFIGS["fl_rounds"],
            epochs=CONFIGS["fl_epochs"],
            batch_size=CONFIGS["batch_size"],
            lr=CONFIGS["base_lr"],
            iid=client_config["iid"],
            run_name=f"fl_c{client_config['num_clients']}_{'iid' if client_config['iid'] else 'non_iid'}",
        )

        # Save results and models
        save_experiment_results(
            centralized_model=centralized_model,
            federated_model=fl_model,
            client_config=client_config,
        )


def save_experiment_results(centralized_model, federated_model, client_config):
    """Save models and results from experiment."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    experiment_name = f"exp_{timestamp}_c{client_config['num_clients']}_{'iid' if client_config['iid'] else 'non_iid'}"

    # Create directories
    RESULTS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)

    # Save models
    torch.save(
        centralized_model.state_dict(),
        MODELS_DIR / f"centralized_{experiment_name}.pth",
    )
    torch.save(
        federated_model.state_dict(), MODELS_DIR / f"federated_{experiment_name}.pth"
    )


########################
# Main Execution
########################
if __name__ == "__main__":
    setup_logging(logging.INFO)
    logging.debug(f"Using device: {DEVICE_TYPE}")

    run_research_experiment()
