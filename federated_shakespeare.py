# Import necessary libraries
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import CharLSTM and ShakespeareDataset from lstm.py
from lstm import CharLSTM, ShakespeareDataset

# Define the custom train_federated function to meet project requirements
def train_federated(data_path, global_model, criterion, rounds, clients, fraction, device, seq_length, local_steps, participation="uniform", gamma=0.5):
    """
    Custom implementation of Federated Learning training.

    Args:
        data_path (str): Path to the dataset.
        global_model (nn.Module): Global LSTM model for training.
        criterion (nn.Module): Loss function.
        rounds (int): Number of communication rounds.
        clients (list): List of client IDs.
        fraction (float): Fraction of clients to select in each round.
        device (torch.device): Training device (CPU/GPU).
        seq_length (int): Sequence length for character input.
        local_steps (int): Number of local training steps per client.
        participation (str): Participation strategy ("uniform" or "skewed").
        gamma (float): Skewness parameter for skewed participation.

    Returns:
        dict: Dictionary containing metrics such as global loss and accuracy.
    """
    global_model.to(device)
    metrics = {
        "global_loss": [],
        "accuracy": [],
        "client_distributions": [] if participation == "skewed" else None
    }

    for round_num in range(rounds):
        print(f"Starting Round {round_num + 1}/{rounds}...")

        # Client selection based on participation strategy
        if participation == "uniform":
            selected_clients = sample_clients_uniform(clients, fraction)
        elif participation == "skewed":
            selected_clients, probabilities = sample_clients_skewed(clients, fraction, gamma)
            metrics["client_distributions"].append(probabilities)
        else:
            raise ValueError("Invalid participation strategy.")

        # Local updates collection
        local_weights = []
        for client_id in tqdm(selected_clients, desc=f"Training Local Models"):
            # Load client-specific data
            client_dataset = ShakespeareDataset(data_path, clients=[client_id], seq_length=seq_length)
            data_loader = DataLoader(client_dataset, batch_size=32, shuffle=True)

            # Initialize local model
            local_model = CharLSTM(global_model.embedding.num_embeddings).to(device)
            local_model.load_state_dict(global_model.state_dict())

            # Define local optimizer
            optimizer = torch.optim.Adam(local_model.parameters(), lr=0.005)

            # Perform local training
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)
            for _ in range(local_steps):
                for inputs, targets in data_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    optimizer.zero_grad()
                    outputs, _ = local_model(inputs)
                    outputs = outputs.view(-1, outputs.size(-1))
                    targets = targets.view(-1)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

            # Append local model weights
            local_weights.append({k: v.clone() for k, v in local_model.state_dict().items()})

        # Federated aggregation using FedAvg
        global_weights = global_model.state_dict()
        for key in global_weights.keys():
            global_weights[key] = torch.stack([w[key] for w in local_weights], dim=0).mean(dim=0)
        global_model.load_state_dict(global_weights)

        # Evaluate global model on test data
        test_dataset = ShakespeareDataset(data_path)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        global_loss, global_accuracy = evaluate_model(global_model, test_loader, criterion, device)

        # Log metrics
        metrics["global_loss"].append(global_loss)
        metrics["accuracy"].append(global_accuracy)

        print(f"Round {round_num + 1}: Global Loss: {global_loss:.4f}, Accuracy: {global_accuracy:.4f}")

    return metrics

# Define client sampling functions
def sample_clients_uniform(clients, fraction):
    num_clients = len(clients)
    num_selected = max(1, int(fraction * num_clients))
    return np.random.choice(clients, num_selected, replace=False)


def sample_clients_skewed(clients, fraction, gamma):
    num_clients = len(clients)
    num_selected = max(1, int(fraction * num_clients))
    probabilities = np.random.dirichlet([gamma] * num_clients)
    selected_indices = np.random.choice(range(num_clients), num_selected, p=probabilities, replace=False)
    return [clients[i] for i in selected_indices], probabilities

# Evaluate the global model
def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, _ = model(inputs)
            outputs = outputs.view(-1, outputs.size(-1))
            targets = targets.view(-1)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct_predictions += (predicted == targets).sum().item()
            total_samples += targets.size(0)

    return total_loss / len(data_loader), correct_predictions / total_samples

# Ensure the dataset is prepared
data_path = './data/shakespeare/train_data.json'
if not os.path.exists(data_path):
    from generate_data_shakespeare import generate_mock_shakespeare_dataset
    generate_mock_shakespeare_dataset()

# Configurations
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocab_size = len(ShakespeareDataset(data_path).vocab)
rounds = 50  # Number of federated rounds
fraction = 0.1  # Fraction of clients to select each round
seq_length = 80  # Sequence length for LSTM
local_steps = 5  # Local training steps
batch_size = 32

# Loss function and global model
criterion = nn.CrossEntropyLoss()
global_model = CharLSTM(vocab_size).to(device)

# Load client information
with open(data_path, 'r') as f:
    raw_data = json.load(f)
    clients = raw_data['users']  # List of client IDs

# Federated Training: Uniform Participation
print("Starting federated training with uniform participation...")
metrics_uniform = train_federated(
    data_path=data_path,
    global_model=global_model,
    criterion=criterion,
    rounds=rounds,
    clients=clients,
    fraction=fraction,
    device=device,
    seq_length=seq_length,
    local_steps=local_steps,
    participation="uniform"
)

# Federated Training: Skewed Participation
print("Starting federated training with skewed participation...")
global_model = CharLSTM(vocab_size).to(device)  # Reset the global model
metrics_skewed = train_federated(
    data_path=data_path,
    global_model=global_model,
    criterion=criterion,
    rounds=rounds,
    clients=clients,
    fraction=fraction,
    device=device,
    seq_length=seq_length,
    local_steps=local_steps,
    participation="skewed",
    gamma=0.5
)

# Visualization
plt.figure()
plt.plot(range(1, len(metrics_uniform["global_loss"]) + 1), metrics_uniform["global_loss"], label="Uniform Participation")
plt.plot(range(1, len(metrics_skewed["global_loss"]) + 1), metrics_skewed["global_loss"], label="Skewed Participation")
plt.xlabel("Round")
plt.ylabel("Loss")
plt.legend()
plt.title("Federated Training Performance")
plt.show()

# Plot sampling distributions (if skewed)
if metrics_skewed.get("client_distributions"):
    plt.figure()
    plt.bar(range(len(metrics_skewed["client_distributions"][-1])), metrics_skewed["client_distributions"][-1])
    plt.title("Sampling Distribution (Skewed, Last Round)")
    plt.xlabel("Client")
    plt.ylabel("Probability")
    plt.show()
