import logging

import adabelief_pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.amp import autocast_mode
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from tqdm import tqdm

from utils import get_shakespeare_client_datasets


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
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=lr,
            epochs=epochs,
            steps_per_epoch=len(trainloader),
            pct_start=0.1,  # 10% warmup
            div_factor=25,
            final_div_factor=1000,
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
        trainloader, testloader, idx_to_char = get_shakespeare_client_datasets(
            batch_size=batch_size, num_clients=1
        )
        trainloader = trainloader[0]
        _num_classes = len(idx_to_char)
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
