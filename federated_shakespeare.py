# Import necessary libraries
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import utilities from the existing implementation
from lstm import CharLSTM, ShakespeareDataset, train_federated

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
uniform_losses, _ = train_federated(
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
skewed_losses, sampling_distributions = train_federated(
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
    gamma=0.5  # Skewness parameter
)

# Visualization
plt.figure()
plt.plot(range(1, len(uniform_losses) + 1), uniform_losses, label="Uniform Participation")
plt.plot(range(1, len(skewed_losses) + 1), skewed_losses, label="Skewed Participation")
plt.xlabel("Round")
plt.ylabel("Loss")
plt.legend()
plt.title("Federated Training Performance")
plt.show()

# Plot sampling distributions (if skewed)
if len(sampling_distributions) > 0:
    plt.figure()
    plt.bar(range(len(sampling_distributions[-1])), sampling_distributions[-1])
    plt.title("Sampling Distribution (Skewed, Last Round)")
    plt.xlabel("Client")
    plt.ylabel("Probability")
    plt.show()
