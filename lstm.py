# Import required libraries for dataset management, model building, training, and visualization.
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm  # For progress tracking.

# ====================
# Dataset Utilities
# ====================

# Class to handle the Shakespeare dataset in a way suitable for PyTorch.
class ShakespeareDataset(Dataset):
    def __init__(self, data_path, clients=None, seq_length=80):
        """
        Initialize the dataset by loading and preprocessing the data.
        Args:
        - data_path: Path to the JSON file containing the dataset.
        - clients: List of client IDs to load data for (default: all clients).
        - seq_length: Sequence length for character-level data.
        """
        self.seq_length = seq_length
        self.data = []  # Store input sequences.
        self.targets = []  # Store target sequences.
        self.build_vocab()  # Build the character-level vocabulary.
        self.load_data(data_path, clients)  # Load and preprocess the dataset.

    def build_vocab(self):
        """
        Define the vocabulary for character-level input and build mappings.
        """
        self.vocab = [c for c in " $&'(),-.0123456789:;>?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz}"]
        self.char2idx = {char: idx for idx, char in enumerate(self.vocab)}  # Character to index.
        self.idx2char = {idx: char for idx, char in enumerate(self.vocab)}  # Index to character.

    def load_data(self, data_path, clients):
        """
        Load and preprocess data for selected clients.
        """
        with open(data_path, 'r') as f:
            raw_data = json.load(f)
            # Use selected clients or default to all users in the dataset.
            selected_clients = clients if clients else raw_data['users']
            for client in selected_clients:
                # Concatenate all text data for this client.
                user_text = ' '.join(raw_data['user_data'][client]['x'])
                self.process_text(user_text)

    def process_text(self, text):
        """
        Split text data into input-output sequences of seq_length.
        """
        for i in range(len(text) - self.seq_length):
            seq = text[i:i + self.seq_length]  # Input sequence.
            target = text[i + 1:i + self.seq_length + 1]  # Target sequence.
            seq_indices = [self.char2idx.get(c, 0) for c in seq]
            target_indices = [self.char2idx.get(c, 0) for c in target]
            self.data.append(torch.tensor(seq_indices, dtype=torch.long))
            self.targets.append(torch.tensor(target_indices, dtype=torch.long))

    def __len__(self):
        """
        Return the number of sequences in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieve the input-target pair at the specified index.
        """
        return self.data[idx], self.targets[idx]


# ====================
# LSTM Model Definition
# ====================

# Define the character-level LSTM model for Shakespeare data.
class CharLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=8, hidden_dim=256, num_layers=2, dropout=0.2):
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
        self.embedding = nn.Embedding(vocab_size, embedding_dim)  # Character embedding layer.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)  # LSTM layers.
        self.fc = nn.Linear(hidden_dim, vocab_size)  # Output layer (vocab_size outputs).

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


# ====================
# Centralized Training
# ====================

# Define the centralized training pipeline.
def train_centralized(model, train_loader, test_loader, criterion, optimizer, scheduler, epochs, device):
    """
    Train the model on a centralized dataset.
    Args:
    - model: The LSTM model to train.
    - train_loader: DataLoader for training data.
    - test_loader: DataLoader for test data.
    - criterion: Loss function.
    - optimizer: Optimizer (SGD).
    - scheduler: Learning rate scheduler.
    - epochs: Number of training epochs.
    - device: Device to train on (CPU or GPU).
    Returns:
    - Training losses and accuracies, along with test loss and accuracy.
    """
    model.to(device)  # Move model to the device (CPU/GPU).
    model.train()  # Set the model to training mode.
    epoch_losses = []  # Store training loss for each epoch.
    epoch_accuracies = []  # Store training accuracy for each epoch.

    for epoch in range(epochs):
        total_loss = 0
        correct_predictions = 0
        total_samples = 0
        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")  # Track progress.

        for inputs, targets in progress:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()  # Clear previous gradients.
            outputs, _ = model(inputs)  # Forward pass.
            outputs = outputs.view(-1, outputs.size(-1))  # Reshape for loss computation.
            targets = targets.view(-1)  # Reshape for loss computation.
            loss = criterion(outputs, targets)  # Compute loss.
            loss.backward()  # Backpropagation.
            optimizer.step()  # Update weights.

            total_loss += loss.item()
            _, predictions = outputs.max(1)  # Get predictions.
            correct_predictions += (predictions == targets).sum().item()  # Count correct predictions.
            total_samples += targets.size(0)  # Update sample count.
            progress.set_postfix(loss=loss.item())  # Show current loss.

        scheduler.step()  # Update learning rate (scheduler).
        train_accuracy = correct_predictions / total_samples  # Compute accuracy.
        avg_loss = total_loss / len(train_loader)  # Compute average loss.
        epoch_losses.append(avg_loss)
        epoch_accuracies.append(train_accuracy)
        print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}, Accuracy: {train_accuracy:.4f}")

    # Evaluate on the test set.
    model.eval()
    test_loss, test_accuracy = evaluate_model(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
    return epoch_losses, epoch_accuracies, test_loss, test_accuracy


# Evaluate model performance on a dataset.
def evaluate_model(model, data_loader, criterion, device):
    """
    Evaluate the model on a given dataset.
    Args:
    - model: Trained model.
    - data_loader: DataLoader for the evaluation dataset.
    - criterion: Loss function.
    - device: Device to evaluate on (CPU/GPU).
    Returns:
    - Average loss and accuracy.
    """
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():  # Disable gradient computation for evaluation.
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, _ = model(inputs)
            outputs = outputs.view(-1, outputs.size(-1))
            targets = targets.view(-1)
            loss = criterion(outputs, targets)  # Compute loss.
            total_loss += loss.item()
            _, predictions = outputs.max(1)
            correct_predictions += (predictions == targets).sum().item()
            total_samples += targets.size(0)

    avg_loss = total_loss / len(data_loader)  # Compute average loss.
    accuracy = correct_predictions / total_samples  # Compute accuracy.
    return avg_loss, accuracy


# ====================
# Federated Training Utilities
# ====================

# Train a local model on a single client.
def train_local_model(model, data_loader, criterion, optimizer, local_steps, device):
    """
    Train the model locally on a client's dataset.
    Args:
    - model: LSTM model to train.
    - data_loader: DataLoader for the client's data.
    - criterion: Loss function.
    - optimizer: Optimizer.
    - local_steps: Number of local training steps.
    - device: Device to train on (CPU/GPU).
    """
    model.train()
    for _ in range(local_steps):
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs, _ = model(inputs)
            outputs = outputs.view(-1, outputs.size(-1))
            targets = targets.view(-1)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()


# Sample clients uniformly for a round of training.
def sample_clients_uniform(clients, fraction):
    """
    Sample a fraction of clients uniformly.
    Args:
    - clients: List of all clients.
    - fraction: Fraction of clients to sample.
    Returns:
    - A list of selected clients.
    """
    num_clients = len(clients)
    num_selected = max(1, int(fraction * num_clients))  # Compute number of selected clients.
    return np.random.choice(clients, num_selected, replace=False)  # Uniform random sampling.


# Sample clients skewed using Dirichlet distribution.
def sample_clients_skewed(clients, fraction, gamma):
    """
    Sample a fraction of clients based on Dirichlet distribution.
    Args:
    - clients: List of all clients.
    - fraction: Fraction of clients to sample.
    - gamma: Skewness parameter for Dirichlet distribution.
    Returns:
    - List of selected clients and their probabilities.
    """
    num_clients = len(clients)
    num_selected = max(1, int(fraction * num_clients))
    probabilities = np.random.dirichlet([gamma] * num_clients)  # Generate skewed probabilities.
    selected_indices = np.random.choice(range(num_clients), num_selected, replace=False, p=probabilities)
    return [clients[i] for i in selected_indices], probabilities


# Federated training with FedAvg.
def train_federated(data_path, global_model, criterion, rounds, clients, fraction, device, seq_length, local_steps, participation="uniform", gamma=0.5):
    """
    Train the global model using federated averaging (FedAvg).
    Args:
    - data_path: Path to dataset.
    - global_model: Global model to train.
    - criterion: Loss function.
    - rounds: Number of communication rounds.
    - clients: List of all clients.
    - fraction: Fraction of clients to select in each round.
    - device: Device to train on (CPU/GPU).
    - seq_length: Sequence length for local models.
    - local_steps: Number of local training steps per client.
    - participation: Participation scheme ('uniform' or 'skewed').
    - gamma: Skewness parameter for Dirichlet distribution (if 'skewed').
    Returns:
    - List of global losses and sampling distributions (if skewed).
    """
    global_model.to(device)
    global_losses = []  # Track global loss at each round.
    sampling_distributions = []  # Track sampling probabilities for skewed participation.

    for round_num in range(rounds):
        print(f"Round {round_num + 1}/{rounds}")
        local_weights = []
        if participation == "uniform":
            selected_clients = sample_clients_uniform(clients, fraction)  # Uniform sampling.
            sampling_distributions.append([1 / len(clients)] * len(clients))  # Uniform probabilities.
        elif participation == "skewed":
            selected_clients, probabilities = sample_clients_skewed(clients, fraction, gamma)  # Skewed sampling.
            sampling_distributions.append(probabilities)  # Store probabilities.

        # Train each selected client.
        for client in tqdm(selected_clients, desc=f"Round {round_num + 1}"):
            local_model = CharLSTM(global_model.embedding.num_embeddings).to(device)  # Create local copy.
            local_model.load_state_dict(global_model.state_dict())  # Load global model weights.
            optimizer = optim.SGD(local_model.parameters(), lr=0.8)  # Local optimizer.

            # Load client's dataset.
            client_dataset = ShakespeareDataset(data_path, clients=[client], seq_length=seq_length)
            client_loader = DataLoader(client_dataset, batch_size=32, shuffle=True)

            # Train local model.
            train_local_model(local_model, client_loader, criterion, optimizer, local_steps, device)

            # Save local weights for aggregation.
            local_weights.append({k: v.clone() for k, v in local_model.state_dict().items()})

        # Aggregate local weights into the global model.
        global_dict = global_model.state_dict()
        for k in global_dict.keys():
            global_dict[k] = torch.stack([w[k] for w in local_weights], dim=0).mean(dim=0)  # Weighted averaging.
        global_model.load_state_dict(global_dict)  # Update global model.

        # Evaluate global model.
        test_dataset = ShakespeareDataset(data_path)  # Load full dataset for evaluation.
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        test_loss, _ = evaluate_model(global_model, test_loader, criterion, device)
        global_losses.append(test_loss)  # Track global loss.
        print(f"Round {round_num + 1}, Global Loss: {test_loss:.4f}")

    return global_losses, sampling_distributions


# ====================
# Main Execution
# ====================

def main():
    # Dataset and training configurations.
    data_path = './data/shakespeare/train_data.json'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available.
    epochs = 10  # Number of epochs for centralized training.
    rounds = 50  # Number of federated communication rounds.
    fraction = 0.1  # Fraction of clients to select each round.
    seq_length = 80  # Sequence length for LSTM inputs.
    local_steps = 4  # Number of local training steps.
    gamma = 0.5  # Skewness parameter for Dirichlet sampling.

    # Centralized Dataset Preparation.
    centralized_dataset = ShakespeareDataset(data_path, seq_length=seq_length)
    train_size = int(0.8 * len(centralized_dataset))  # 80% of data for training.
    val_size = len(centralized_dataset) - train_size  # 20% of data for validation.
    train_dataset, test_dataset = torch.utils.data.random_split(centralized_dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    vocab_size = len(centralized_dataset.vocab)

    # ====================
    # Centralized Training
    # ====================
    print("Starting centralized training...")
    model = CharLSTM(vocab_size)  # Initialize LSTM model.
    criterion = nn.CrossEntropyLoss()  # Loss function.
    optimizer = optim.SGD(model.parameters(), lr=0.8)  # Optimizer.
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)  # Learning rate scheduler.

    # Train and evaluate centralized model.
    centralized_losses, centralized_accuracies, test_loss, test_accuracy = train_centralized(
        model, train_loader, test_loader, criterion, optimizer, scheduler, epochs, device
    )

    # Plot centralized training performance.
    plt.figure()
    plt.plot(range(1, len(centralized_losses) + 1), centralized_losses, label="Centralized Loss")
    plt.plot(range(1, len(centralized_accuracies) + 1), centralized_accuracies, label="Centralized Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()
    plt.title("Centralized Training Performance")
    plt.show()

    # ====================
    # Federated Training
    # ====================
    print("Starting federated training (Uniform Participation)...")
    with open(data_path, 'r') as f:
        raw_data = json.load(f)
        clients = raw_data['users']  # Get all client IDs.

    global_model = CharLSTM(vocab_size)  # Initialize global LSTM model.
    uniform_losses, _ = train_federated(
        data_path, global_model, criterion, rounds, clients, fraction, device, seq_length, local_steps, "uniform"
    )

    print("Starting federated training (Skewed Participation)...")
    global_model = CharLSTM(vocab_size)  # Reset global model for skewed participation.
    skewed_losses, sampling_distributions = train_federated(
        data_path, global_model, criterion, rounds, clients, fraction, device, seq_length, local_steps, "skewed", gamma
    )

    # Plot federated training performance.
    plt.figure()
    plt.plot(range(1, len(uniform_losses) + 1), uniform_losses, label="Uniform Participation")
    plt.plot(range(1, len(skewed_losses) + 1), skewed_losses, label="Skewed Participation")
    plt.xlabel("Round")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Federated Training Performance")
    plt.show()

    # Plot sampling distributions for skewed participation.
    if len(sampling_distributions) > 0:
        plt.figure()
        plt.bar(range(len(sampling_distributions[-1])), sampling_distributions[-1])
        plt.title("Sampling Distribution (Skewed, Last Round)")
        plt.xlabel("Client")
        plt.ylabel("Probability")
        plt.show()


if __name__ == "__main__":
    main()
