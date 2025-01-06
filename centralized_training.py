import logging

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18

from utils import get_shakespeare_client_datasets


class CNN(nn.Module):
    def __init__(self, num_classes=100):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),  # Increased channels
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=512, num_layers=2):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, num_layers, batch_first=True, dropout=0.3
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Predict the last character in the sequence
        return out


class ImprovedLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, num_layers=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.emb_dropout = nn.Dropout(0.2)
        self.lstm_dropout = nn.Dropout(0.3)

        # Layer normalization for better training stability
        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim * 2)

        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2,
        )
        # Account for bidirectional
        self.attention = nn.MultiheadAttention(hidden_dim * 2, num_heads=4, dropout=0.1)

        # Output layers
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, vocab_size)
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

                # Optional: Stop if we generate an end token
                # if next_char.item() == end_token:
                #     break

        return generated


def build_model(architecture="cnn", num_classes=100):
    """Builds and returns the specified model architecture."""
    if architecture.lower() == "cnn":
        return CNN(num_classes=num_classes)
    elif architecture.lower() == "resnet":
        return resnet18(pretrained=False, num_classes=num_classes)
    elif architecture.lower() == "lstm":
        return ImprovedLSTM(
            vocab_size=num_classes, embedding_dim=64, hidden_dim=128, num_layers=1
        )  # Use num_classes as vocab_size
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")


def train_model(model, dataloader, criterion, optimizer, device, dataset):
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

        if dataset.lower() == "shakespeare":
            # Reshape outputs and labels for sequence prediction
            # outputs shape: [batch_size, seq_length, vocab_size]
            # labels shape: [batch_size, seq_length]
            batch_size, seq_length = labels.size()
            outputs = outputs.view(
                -1, outputs.size(-1)
            )  # [batch_size*seq_length, vocab_size]
            labels = labels.view(-1)  # [batch_size*seq_length]

        # Calculate loss
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        # Statistics
        total_loss += loss.item() * (
            inputs.size(0) if dataset.lower() == "cifar100" else labels.size(0)
        )
        if dataset.lower() == "cifar100":
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        if batch_idx % 100 == 0:
            logging.info(f"Batch: {batch_idx}, Loss: {loss.item():.3f}")

    avg_loss = total_loss / (
        len(dataloader.dataset)
        if dataset.lower() == "cifar100"
        else len(dataloader.dataset) * seq_length  # type: ignore
    )
    if dataset.lower() == "cifar100":
        accuracy = 100.0 * correct / total
        return avg_loss, accuracy
    return avg_loss, None


def evaluate_model(model, dataloader, device, dataset):
    """Evaluate the model on a given dataset."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            if dataset.lower() == "shakespeare":
                # Reshape outputs and labels for sequence prediction
                batch_size, seq_length = labels.size()
                outputs = outputs.view(-1, outputs.size(-1))
                labels = labels.view(-1)
                total += labels.size(0)

            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)

            if dataset.lower() == "cifar100":
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

    avg_loss = total_loss / total

    if dataset.lower() == "shakespeare":
        perplexity = torch.exp(torch.tensor(avg_loss))
        return avg_loss, perplexity
    else:
        accuracy = 100.0 * correct / total
        return avg_loss, accuracy


def centralized_training(
    dataset="cifar100", architecture="cnn", epochs=10, batch_size=16, lr=0.001
):
    """Centralized training with specified configuration"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Data loading
    if dataset.lower() == "cifar100":
        transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
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
            batch_size=batch_size, num_clients=5
        )
        num_classes = len(idx_to_char)
    else:
        raise ValueError(f"Dataset {dataset} not supported")

    # Model initialization
    model = build_model(
        architecture, num_classes=100 if dataset.lower() == "cifar100" else num_classes
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    for epoch in range(epochs):
        logging.info(f"Epoch {epoch + 1}/{epochs}")
        if dataset.lower() == "shakespeare":
            loss, accuracy = train_model(
                model, trainloader[0], criterion, optimizer, device, dataset
            )
        else:
            loss, accuracy = train_model(
                model, trainloader, criterion, optimizer, device, dataset
            )

        if dataset.lower() == "cifar100":
            logging.info(f"Train Loss: {loss:.4f}, Train Accuracy: {accuracy:.2f}%")
        else:
            logging.info(f"Train Loss: {loss:.4f}")

        scheduler.step(loss)

    # Evaluation
    eval_loss, eval_metric = evaluate_model(model, testloader, device, dataset)
    if dataset.lower() == "shakespeare":
        logging.info(f"Test Loss: {eval_loss:.4f}, Test Perplexity: {eval_metric:.4f}")
    else:
        logging.info(f"Test Loss: {eval_loss:.4f}, Test Accuracy: {eval_metric:.2f}%")

    # Save the trained model
    torch.save(model.state_dict(), f"{architecture}_{dataset}.pth")
    logging.info(f"Model saved as {architecture}_{dataset}.pth")

    return model


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,  # Livello del logging
        format="%(asctime)s - %(levelname)s - %(message)s",  # Formato del log
        handlers=[
            logging.StreamHandler(),  # Per vedere i log nel terminale
            logging.FileHandler(
                "training_logs.log", mode="w"
            ),  # Per salvare i log in un file
        ],
    )
    model = centralized_training(
        dataset="shakespeare", architecture="lstm", epochs=30, batch_size=64, lr=0.0005
    )
    logging.info("Training completed!")
