# Import required libraries for dataset management, model building, training, and visualization.
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torch.backends import cudnn
import matplotlib.pyplot as plt
from collections import defaultdict
import random
import kagglehub
import shutil
import glob
import re
from collections import Counter
from collections import defaultdict
import random
from tqdm import tqdm  # For progress tracking.


# ====================
# Dataset Prsing and Preprocessing
# ====================

# Regular expressions for parsing Shakespeare text
CHARACTER_LINE_RE = re.compile(r'^  ([a-zA-Z][a-zA-Z ]*)\. (.*)')
CONTINUATION_LINE_RE = re.compile(r'^    (.*)')
COE_CHARACTER_LINE_RE = re.compile(r'^([a-zA-Z][a-zA-Z ]*)\. (.*)')
COE_CONTINUATION_LINE_RE = re.compile(r'^(.*)')

CHARACTER_RE = re.compile(r'^  ([a-zA-Z][a-zA-Z ]*)\. (.*)')  # Matches character lines
CONT_RE = re.compile(r'^    (.*)')  # Matches continuation lines
COE_CHARACTER_RE = re.compile(r'^([a-zA-Z][a-zA-Z ]*)\. (.*)')  # Special regex for Comedy of Errors
COE_CONT_RE = re.compile(r'^(.*)')  # Continuation for Comedy of Errors

# path = kagglehub.dataset_download("kewagbln/shakespeareonline")
# print("Path to dataset files:", path)
# DATA_PATH = os.path.join(path, "shakespeare.txt")
# OUTPUT_DIR = "processed_data/"


# Get current script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Download dataset
path = kagglehub.dataset_download("kewagbln/shakespeareonline")

# Debug: print downloaded files
print(f"Downloaded path: {path}")
print("Files in downloaded path:")
for file in glob.glob(os.path.join(path, "*")):
    print(f" - {file}")

# Set up paths relative to script location
DATA_PATH = os.path.join(SCRIPT_DIR, "shakespeare.txt")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "processed_data")

# Create directories if they don't exist
os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Find and copy Shakespeare text file
shakespeare_file = None
for file in glob.glob(os.path.join(path, "*.txt")):
    shakespeare_file = file
    break

if shakespeare_file:
    shutil.copy2(shakespeare_file, DATA_PATH)
    print(f"Dataset saved to: {DATA_PATH}")
else:
    raise FileNotFoundError(f"Could not find Shakespeare text file in {path}")




def parse_shakespeare(filepath, train_split=0.9):
    """
    Parses Shakespeare's text into training and testing datasets.
    """
    with open(filepath, "r") as file:
        raw_text = file.read()

    plays_data, _ = process_plays(raw_text)
    _, training_set, testing_set = split_train_test_data(plays_data, 0.9)
    return training_set, testing_set

def process_plays(shakespeare_full):
    """
    Processes the Shakespeare text into individual plays and characters' dialogues.
    Handles special cases for "The Comedy of Errors".
    """
    plays = []
    slines = shakespeare_full.splitlines(True)[1:]  # Skip the first line (title/header)
    current_character = None
    comedy_of_errors = False

    for i, line in enumerate(slines):
        # Detect play titles and initialize character dictionary
        if "by William Shakespeare" in line:
            current_character = None
            characters = defaultdict(list)
            title = slines[i - 2].strip() if slines[i - 2].strip() else slines[i - 3].strip()
            comedy_of_errors = title == "THE COMEDY OF ERRORS"
            plays.append((title, characters))
            continue

        # Match character lines or continuation lines
        match = _match_character_regex(line, comedy_of_errors)
        if match:
            character, snippet = match.group(1).upper(), match.group(2)
            if not (comedy_of_errors and character.startswith("ACT ")):
                characters[character].append(snippet)
                current_character = character
        elif current_character:
            match = _match_continuation_regex(line, comedy_of_errors)
            if match:
                characters[current_character].append(match.group(1))

    # Filter out plays with insufficient dialogue data
    return [play for play in plays if len(play[1]) > 1], []

def _match_character_regex(line, comedy_of_errors=False):
    """Matches character dialogues, with special handling for 'The Comedy of Errors'."""
    return COE_CHARACTER_RE.match(line) if comedy_of_errors else CHARACTER_RE.match(line)

def _match_continuation_regex(line, comedy_of_errors=False):
    """Matches continuation lines of dialogues."""
    return COE_CONT_RE.match(line) if comedy_of_errors else CONT_RE.match(line)

def extract_play_title(lines, index):
    """
    Extracts the title of the play from the lines of the text.
    """
    for i in range(index - 1, -1, -1):
        if lines[i].strip():
            return lines[i].strip()
    return "UNKNOWN"

def detect_character_line(line, comedy_of_errors):
    """
    Matches a line of character dialogue.
    """
    return COE_CHARACTER_LINE_RE.match(line) if comedy_of_errors else CHARACTER_LINE_RE.match(line)

def detect_continuation_line(line, comedy_of_errors):
    """
    Matches a continuation line of dialogue.
    """
    return COE_CONTINUATION_LINE_RE.match(line) if comedy_of_errors else CONTINUATION_LINE_RE.match(line)

def split_train_test_data(plays, test_fraction):
    """
    Splits the plays into training and testing datasets by character dialogues.
    """
    all_train_examples = defaultdict(list)
    all_test_examples = defaultdict(list)

    def add_examples(example_dict, example_tuple_list):
        """Adds examples to the respective dataset dictionary."""
        for play, character, sound_bite in example_tuple_list:
            example_dict[f"{play}_{character}".replace(" ", "_")].append(sound_bite)

    for play, characters in plays:
        for character, sound_bites in characters.items():
            examples = [(play, character, sound_bite) for sound_bite in sound_bites]
            if len(examples) <= 2:
                continue

            # Calculate the number of test samples
            num_test = max(1, int(len(examples) * test_fraction))
            num_test = min(num_test, len(examples) - 1)  # Ensure at least one training example

            # Split into train and test sets
            train_examples = examples[:-num_test]
            test_examples = examples[-num_test:]

            add_examples(all_train_examples, train_examples)
            add_examples(all_test_examples, test_examples)

    return {}, all_train_examples, all_test_examples


# ====================
# Dataset Utilities
# ====================


# def char_to_index(char, vocab_size=128):
#     """
#     Maps a character to an index based on the vocabulary size.
#     """
#     return ord(char) % vocab_size

# def string_to_indices(text, vocab_size=128):
#     """Convert string/list to character indices"""
#     if isinstance(text, list):
#         indices = []
#         for item in text:
#             if isinstance(item, str):
#                 indices.extend([ord(char) % vocab_size for char in item])
#             elif isinstance(item, (int, float)):
#                 indices.append(int(item) % vocab_size)
#     elif isinstance(text, str):
#         indices = [ord(char) % vocab_size for char in text]
#     elif isinstance(text, (int, float)):
#         indices = [int(text) % vocab_size]
#     else:
#         raise TypeError(f"Unsupported input type: {type(text)}")
#     return indices

# def prepare_input_sequences(raw_input_data, sequence_length, vocab_size):
#     """Process input data into fixed-length sequences"""
#     if not isinstance(raw_input_data, (list, str, int, float)):
#         raise TypeError(f"Unsupported input type: {type(raw_input_data)}")
        
#     if isinstance(raw_input_data, (int, float, str)):
#         raw_input_data = [raw_input_data]
        
#     input_sequences = [string_to_indices(string, vocab_size) for string in raw_input_data]
#     padded_sequences = [
#         seq[:sequence_length] + [0] * (sequence_length - len(seq)) 
#         for seq in input_sequences
#     ]
#     return padded_sequences

# def prepare_target_sequences(raw_target_data, sequence_length, vocab_size):
#     """
#     Processes raw target data into padded sequences for the model output.
#     Shifts sequences by one position to the right for training and applies padding.
#     """
#     target_sequences = [string_to_indices(string, vocab_size) for string in raw_target_data]
#     shifted_sequences = [
#         seq[1:sequence_length + 1] + [0] * (sequence_length - len(seq[1:sequence_length + 1]))
#         for seq in target_sequences
#     ]
#     return torch.tensor(shifted_sequences, dtype=torch.long)

# def generate_batches(dialogue_data, batch_size, sequence_length, vocab_size):
#     """
#     Creates batches of input and target data for training.
#     Dialogues are split into uniformly sized batches.
#     """
#     input_batches = []
#     target_batches = []
#     all_dialogues = list(dialogue_data)
#     random.shuffle(all_dialogues.values())  # Randomize the order of dialogues

#     current_batch = []
#     for dialogue in all_dialogues:
#         current_batch.append(dialogue)
#         if len(current_batch) == batch_size:
#             inputs = prepare_input_sequences(current_batch, sequence_length, vocab_size)
#             targets = prepare_target_sequences(current_batch, sequence_length, vocab_size)
#             input_batches.append(inputs)
#             target_batches.append(targets)
#             current_batch = []

#     # Handle remaining dialogues that don't fill a full batch
#     if current_batch:
#         inputs = prepare_input_sequences(current_batch, sequence_length, vocab_size)
#         targets = prepare_target_sequences(current_batch, sequence_length, vocab_size)
#         input_batches.append(inputs)
#         target_batches.append(targets)

#     return input_batches, target_batches

def letter_to_vec(c, n_vocab=128):
    """Converts a single character to a vector index based on the vocabulary size."""
    return ord(c) % n_vocab

def word_to_indices(word, n_vocab=128):
    """
    Converts a word or list of words into a list of indices.
    Each character is mapped to an index based on the vocabulary size.
    """
    if isinstance(word, list):  # If input is a list of words
        res = []
        for stringa in word:
            res.extend([ord(c) % n_vocab for c in stringa])  # Convert each word to indices
        return res
    else:  # If input is a single word
        return [ord(c) % n_vocab for c in word]

def process_x(raw_x_batch, seq_len, n_vocab):
    """
    Processes raw input data into padded sequences of indices.
    Ensures all sequences are of uniform length.
    """
    x_batch = [word_to_indices(word, n_vocab) for word in raw_x_batch]
    x_batch = [x[:seq_len] + [0] * (seq_len - len(x)) for x in x_batch]
    return torch.tensor(x_batch, dtype=torch.long)

def process_y(raw_y_batch, seq_len, n_vocab):
    """
    Processes raw target data into padded sequences of indices.
    Shifts the sequence by one character to the right.
    y[1:seq_len + 1] takes the input data, right shift of an
    element and uses the next element of the sequence to fill
    and at the end (with [0]) final padding (zeros) are (eventually)
    added to reach the desired sequence length.
    """
    y_batch = [word_to_indices(word, n_vocab) for word in raw_y_batch]
    y_batch = [y[1:seq_len + 1] + [0] * (seq_len - len(y[1:seq_len + 1])) for y in y_batch]  # Shifting and final padding
    return torch.tensor(y_batch, dtype=torch.long)

def create_batches(data, batch_size, seq_len, n_vocab):
    """
    Creates batches of input and target data from dialogues.
    Each batch contains sequences of uniform length.
    """
    x_batches = []
    y_batches = []
    dialogues = list(data.values())
    random.shuffle(dialogues)  # Shuffle to ensure randomness in batches

    batch = []
    for dialogue in dialogues:
        batch.append(dialogue)
        if len(batch) == batch_size:
            x_batch = process_x(batch, seq_len, n_vocab)
            y_batch = process_y(batch, seq_len, n_vocab)
            x_batches.append(x_batch)
            y_batches.append(y_batch)
            batch = []

    # Add the last batch if it's not full
    if batch:
        x_batch = process_x(batch, seq_len, n_vocab)
        y_batch = process_y(batch, seq_len, n_vocab)
        x_batches.append(x_batch)
        y_batches.append(y_batch)

    return x_batches, y_batches

# Class to handle the Shakespeare dataset in a way suitable for PyTorch.
class ShakespeareDataset(Dataset):
    def __init__(self, text, clients=None, seq_length=80, n_vocab=90):
        """
        Initialize the dataset by loading and preprocessing the data.
        Args:
        - data_path: Path to the JSON file containing the dataset.
        - clients: List of client IDs to load data for (default: all clients).
        - seq_length: Sequence length for character-level data.
        """
        self.seq_length = seq_length  # Sequence length for the model
        self.n_vocab = n_vocab  # Vocabulary size

    # def load_data(self, data_path, clients):
    #     """
    #     Load and preprocess data for selected clients.
    #     """

    #     # Download latest version of the shakespeare dataset and save the path
    #     path = kagglehub.dataset_download("kewagbln/shakespeareonline")
    #     print("Path to dataset files:", path)
    #     DATA_PATH = os.path.join(path, data_path)
    #     OUTPUT_DIR = "processed_data/"

    # def process_text(self, text):
    #     """
    #     Split text data into input-output sequences of seq_length.
    #     """
    #     for i in range(len(text) - self.seq_length):
    #         seq = text[i:i + self.seq_length]  # Input sequence.
    #         target = text[i + 1:i + self.seq_length + 1]  # Target sequence.
    #         seq_indices = [self.char2idx.get(c, 0) for c in seq]
    #         target_indices = [self.char2idx.get(c, 0) for c in target]
    #         self.data.append(torch.tensor(seq_indices, dtype=torch.long))
    #         self.targets.append(torch.tensor(target_indices, dtype=torch.long))

        # Create character mappings
        
        self.data = list(text.values())  # Convert the dictionary values to a list
            

    def __len__(self):
        """
        Return the number of sequences in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieve the input-target pair at the specified index.
        """
        diag = self.data[idx]
        x = process_x(diag, self.seq_length, self.n_vocab)
        y = process_y(diag, self.seq_length, self.n_vocab)
        return x[0], y[0]


# ====================
# LSTM Model Definition
# ====================

# Define the character-level LSTM model for Shakespeare data.
class CharLSTM(nn.Module):
    def __init__(self, n_vocab=90, embedding_dim=8, hidden_dim=256, seq_length=80, batch_size=4, lr=0.2, num_layers=2, dropout=0.2):
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
        self.seq_length = seq_length
        self.n_vocab = n_vocab
        self.embedding_size = embedding_dim
        self.lstm_hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.lr = lr
        self.embedding = nn.Embedding(n_vocab, embedding_dim)  # Character embedding layer.
        self.lstm_first = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, dropout=dropout)  # LSTM first layer
        self.lstm_second = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, dropout=dropout)  # LSTM second layer.
        self.fc = nn.Linear(hidden_dim, n_vocab)  # Output layer (vocab_size outputs).
        self.softmax = nn.Softmax(dim=-1)  # Softmax activation for output probabilities.
        

    def forward(self, x, hidden=None):
        """
        Forward pass of the model.
        Args:
        - x: Input batch (character indices).
        - hidden: Hidden state for LSTM (default: None, initialized internally).
        Returns:
        - Output logits and the updated hidden state.
        """
        # First layer for Embedding
        x = self.embedding(x)  # Convert indices to embeddings.
        # Second layer for First LSTM
        output, hidden = self.lstm_first(x, hidden)  # Process through first LSTM layer.
        # Third layer for Second LSTM
        output, hidden = self.lstm_second(x, hidden)  # Process through second LSTM layer.
        # Fourth layer for Fully Connected Layer
        output = self.fc(output)  # Generate logits for each character.

        ## WHY NO x = self.softmax(x) ?????
        return output, hidden

    def hidden(self, batch_size):
        """Initializes hidden and cell states for the LSTM."""
        weight = next(self.parameters())
        return (weight.new_zeros(self.num_layers, batch_size, self.lstm_hidden_dim),
                weight.new_zeros(self.num_layers, batch_size, self.lstm_hidden_dim))


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
            # Initialize hidden state
            state = model.hidden(inputs.size(0)) 
            state = (state[0].to(device), state[1].to(device)) 
            outputs, _ = model(inputs)
            outputs = outputs.view(-1, model.n_vocab)
            targets = targets.view(-1)
            loss = criterion(outputs, targets)  # Compute loss.
            total_loss += loss.item()
            _, predictions = outputs.max(1)
            correct_predictions += (predictions == targets).sum().item()
            total_samples += targets.size(0)

    avg_loss = total_loss / len(data_loader)  # Compute average loss.
    accuracy = (correct_predictions / total_samples ) * 100  # Compute accuracy.
    return avg_loss, accuracy


# ====================
# Federated Training Utilities
# ====================

class Client:
    def __init__(self, data_loader, client, model, device):
        self.data = data_loader
        self.client = client
        self.model = model
        self.device = device
    
    # Train a local model on a single client.
    def train_local_model(self, data_loader, criterion, optimizer, local_steps, device):
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


        self.model.train()

        # Optimize runtime
        cudnn.benchmark = True
        
        loss_tot = 0
        correct_prediction = 0
        for _ in range(local_steps):
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                state = self.model.hidden(inputs.size(0))
                state = (state[0].to(device), state[1].to(device)) 
                outputs, _ = self.model(inputs)
                outputs = outputs.view(-1, outputs.size(-1))
                targets = targets.view(-1)
                loss = criterion(outputs, targets)
                loss.backward()
                # Clip gradients to avoid exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                optimizer.step()

                accuracy, loss = evaluate_model(self.model, data_loader, criterion, device)

        return self.model.state_dict(), accuracy, loss

class Server:
    def __init__(self, test_data, val_data, model, device):
        self.test_data = test_data
        self.val_data = val_data
        self.clients = None
        self.global_model = model
        self.device = device
        self.losses_round = []
        self.accuracies_round = []
        self.client_selected = []
        self.test_losses = []
        self.test_accuracies = []


    # Federated training with FedAvg.
    def train_federated(self, train_loader, val_loader, test_loader, criterion, rounds, num_classes, num_clients, fraction, device, lr, momentum, batch_size, wd, seq_length, C=0.1, local_steps=4, participation="uniform", gamma=None):
        """
        Train the global model using federated averaging (FedAvg).
        Args:
        - self -> containing global_model: Global model to train.
        - data_path: Path to dataset.
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

        self.global_model.to(device)

        sampling_distributions = []  # Track sampling probabilities for skewed participation.

        train_losses = []
        train_accuracies = []
        validation_losses = []
        validation_accuracies = []
        
        shards = create_sharding(train_loader.dataset, num_clients, num_classes) #each shard represent the training data for one client
        client_sizes = [len(shard) for shard in shards]

        for round_num in range(rounds):
            client_states = []
            client_losses = []
            client_accuracies = []
            print(f"Round {round_num + 1}/{rounds}")
            local_weights = []
            if participation == "uniform":
                selected_clients = sample_clients_uniform(num_clients, fraction)  # Uniform sampling.
                sampling_distributions.append([1 / num_clients] * num_clients) # Uniform probabilities.
            elif participation == "skewed":
                selected_clients, probabilities = sample_clients_skewed(num_clients, fraction, gamma)  # Skewed sampling.
                sampling_distributions.append(probabilities)  # Store probabilities.

            # Train each selected client.
            for client in tqdm(selected_clients, desc=f"Round {round_num + 1}"):
                local_model = CharLSTM(self.global_model.embedding.num_embeddings).to(device)  # Create local copy.
                local_model.load_state_dict(self.global_model.state_dict())  # Load global model weights.
                local_model.train()
                optimizer = optim.SGD(local_model.parameters(), lr, momentum, 0, wd)  # Local optimizer.

                # Load client's dataset.
               # client_dataset = ShakespeareDataset(train_data, clients=[client], seq_length=seq_length)
                client_loader = DataLoader(shards[client], batch_size, shuffle=True)

                client = Client(client_loader, client, local_model, self.device)

                # Train local model.
                client_local_state, client_accuracy, client_loss = client.train_local_model(client_loader, criterion, optimizer, local_steps, device)
                client_states.append(client_local_state)
                client_losses.append(client_loss)
                client_accuracies.append(client_accuracy)


                # Save local weights for aggregation.
                local_weights.append({k: v.clone() for k, v in local_model.state_dict().items()})

            # Aggregate local weights into the global model with FedAvg 
            global_dict = self.global_model.state_dict()

            tot = sum(client_sizes)
            for k in global_dict.keys():
                global_dict[k] = torch.stack([w[k] for w in local_weights], dim=0).mean(dim=0)  # Weighted averaging.

            loss_tot = 0
            accuracy_tot = 0

            for state, size, loss, accuracy in zip(client_states, client_sizes, client_losses, client_accuracies):
                for k in global_dict.keys():
                    global_dict[k] += (state[k] * size / tot)
                loss_tot += loss * size
                accuracy_tot += accuracy * size

            global_loss = loss_tot / tot
            global_accuracy = accuracy_tot / tot

            self.global_model.load_state_dict(global_dict)  # Update global model.

            train_accuracies.append(global_accuracy)
            train_losses.append(global_loss)
            
            # Validation
            val_loss, val_accuracy = evaluate_model(self.global_model, val_loader, criterion, device)
            validation_losses.append(val_loss)
            validation_accuracies.append(val_accuracy)
            print(f"Round {round_num + 1}, Validation Loss: {val_loss:.4f}")

            # Evaluate global model.
            # test_loss, _ = evaluate_model(global_model, test_loader, criterion, device)
            # global_losses.append(test_loss)  # Track global loss.
            # print(f"Round {round_num + 1}, Global Loss: {test_loss:.4f}")


        return self.global_model, train_accuracies, train_losses, validation_accuracies, validation_losses, sampling_distributions

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
    num_clients = clients
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
    num_clients = clients
    num_selected = max(1, int(fraction * num_clients))
    probabilities = np.random.dirichlet([gamma] * num_clients)  # Generate skewed probabilities.
    selected_indices = np.random.choice(range(num_clients), num_selected, replace=False, p=probabilities)
    return [clients[i] for i in selected_indices], probabilities


# ====================
# Sharding for iid and non-iid splits
# ====================
def create_sharding(data, num_clients, num_classes = 90, num_labels_per_client=None):
    """
    Create data shards for iid and non-iid splits.
    Args:
    - data: Dataset (list of sequences).
    - num_clients: Number of clients.
    - iid: Boolean indicating iid or non-iid distribution.
    - num_labels_per_client: Number of labels per client (for non-iid only).
    Returns:
    - A dictionary of client datasets with 'data' and 'labels' keys.
    """
    # indices = np.random.permutation(len(data))
    # shard_size = len(data) // num_clients
    # remainder = len(data) % num_clients

    # client_data = []
    # init = 0

    # if num_clients == num_classes: # iid
    #     for i in range(num_clients):
    #         end = init + shard_size + (1 if i < remainder else 0)
    #         client_data.append(Subset(data, indices[init:end]))
    #         init = end
    # else:
    #     # Non-iid sharding
    #     # Count of each class in the dataset
    #     target_counts = Counter(target for _, target in data)

    #     # Calculate per client class allocation
    #     class_per_client = np.random.choice(list(target_counts.keys()), size=num_classes, replace=False)
    #     class_idx = {class_: np.where([target == class_ for _, target in data])[0] for class_ in class_per_client}

    #     # Assign class indices evenly to clients
    #     for i in range(num_clients):
    #         client_indices = np.array([], dtype=int)
    #         for class_ in class_per_client:
    #             num_samples = len(class_idx[class_]) // num_clients + (1 if i < remainder else 0)
    #             client_indices = np.concatenate((client_indices, class_idx[class_][:num_samples]))
    #             class_idx[class_] = np.delete(class_idx[class_], np.arange(num_samples))

    #         client_data.append(Subset(data, indices=client_indices))

    # return client_data
    client_data = []
    remainder = len(data) % num_clients

    if num_clients == num_classes:
        # IID sharding logic
        indices = np.random.permutation(len(data))
        batch_size = len(data) // num_clients
        for i in range(num_clients):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size + (1 if i < remainder else 0)
            client_data.append(Subset(data, indices[start_idx:end_idx]))
    else:
        # Non-IID sharding
        targets = np.array([target for _, target in data])
        unique_targets = np.unique(targets)
        
        # Ensure num_classes doesn't exceed available classes
        num_classes = min(num_classes, len(unique_targets))
        
        # Select classes for each client
        class_per_client = np.random.choice(
            unique_targets, 
            size=num_classes, 
            replace=False
        )
        
        # Create class indices dictionary
        class_idx = {
            c: np.where(targets == c)[0] 
            for c in class_per_client
        }
        
        # Distribute data to clients
        for i in range(num_clients):
            client_indices = []
            for class_ in class_per_client:
                idx = class_idx[class_]
                if len(idx) > 0:
                    samples = len(idx) // num_clients + (1 if i < remainder else 0)
                    client_indices.extend(idx[:samples])
                    class_idx[class_] = idx[samples:]
            
            client_data.append(Subset(data, client_indices))

    return client_data

# ====================
# Simulate Shakespeare IID and Non-IID
# ====================
def simulate_shakespeare(data_path, seq_length, num_clients, iid=True, num_labels_per_client=None):
    """
    Simulate iid and non-iid splits for Shakespeare.
    Args:
    - data_path: Path to the dataset.
    - seq_length: Sequence length for LSTM inputs.
    - num_clients: Number of clients.
    - iid: Boolean indicating iid or non-iid distribution.
    - num_labels_per_client: Number of labels per client (for non-iid only).
    Returns:
    - Sharded client datasets.
    """
    dataset = ShakespeareDataset(data_path, seq_length=seq_length)
    data, labels = [], []
    for i in range(len(dataset)):
        d, t = dataset[i]
        data.append(d)
        labels.append(t[0].item())  # Take the first character as "label"
    return create_sharding(data, labels, num_clients, iid, num_labels_per_client)

# ====================
# Plot Data Distributions for IID
# ====================
def plot_data_distribution(client_data, title="Data Distribution"):
    """
    Plot data distribution for clients.
    Args:
    - client_data: Dictionary of client datasets.
    - title: Plot title.
    """
    label_counts = defaultdict(int)
    for client_id, client_dataset in client_data.items():
        for label in client_dataset["labels"]:
            label_counts[label] += 1

    plt.figure(figsize=(10, 5))
    plt.bar(label_counts.keys(), label_counts.values())
    plt.xlabel("Labels")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.show()


# ====================
# Main Execution
# ====================

def main():
    # Dataset and training configurations
    data_path = "shakespeare.txt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available
    epochs = 20  # Number of epochs for centralized training
    rounds = 2000  # Number of federated communication rounds
    fraction = 0.1  # Fraction of clients to select each round
    seq_length = 80  # Sequence length for LSTM inputs
    local_steps = 4  # Number of local training steps
    gamma = 0.05  # Skewness parameter for Dirichlet sampling
    num_clients = 100  # Total number of clients
    #num_labels_per_client_list = [1, 5, 10, 50]  # For non-IID experiments
    #local_steps_list = [4, 8, 16]  # Varying local steps
    batch_size = 4 # For local training
    n_vocab = 90 # Character number in vobulary (ASCII)
    learning_rate = 1e-2
    embedding_size = 8
    hidden_dim = 256
    train_split = 0.9
    momentum = 0.9
    weight_decay = 1e-4
    C = 0.1

    # Load data
    train_data, test_data = parse_shakespeare(data_path, train_split)

    # Centralized Dataset Preparation
    train_dataset = ShakespeareDataset(train_data, seq_length=seq_length, n_vocab=n_vocab)
    test_dataset = ShakespeareDataset(test_data, seq_length=seq_length, n_vocab=n_vocab)
    train_size = int(0.9 * len(train_dataset))  # 90% of data for training
    val_size = len(train_dataset) - train_size  # 10% of data for validation
    train_dataset, validation_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    
    # ====================
    # Federated Training
    # ====================

    print("Starting federated training (Uniform Participation)...")
    # with open(data_path, 'r') as f:
    #     raw_data = json.load(f)
    #     clients = raw_data['users']  # Get all client IDs

    learning_rates = np.logspace(-3, -1, 3)
    weight_decays = np.logspace(-4, -1, 4)
    rounds = 100
    best_accuracy = 0
    best_lr = 0
    best_wd = 0

    for wd in weight_decays:
        for lr in learning_rates:
            print(f"Weight decay: {wd} and Learning rate: {lr}")

    
            global_model = CharLSTM(n_vocab, embedding_size, hidden_dim, seq_length, batch_size, lr, num_layers=2) # Initialize global LSTM model
            global_model = global_model.to(device)
            server = Server(test_data, val_loader, global_model, device)
            #criterion = nn.NLLLoss() # DA CONTROLLARE QUALE LOSS SI USA QUA SE QUESTA O QUELLA CROSSENTROPY
            criterion = nn.CrossEntropyLoss()  # Loss function
            optimizer = optim.SGD(global_model.parameters(), learning_rate, momentum, 0, weight_decay)  # Optimizer
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)  # Learning rate scheduler
            global_model, validation_accuracies, validation_losses, train_accuracies, train_losses, client_sel_count = server.train_federated(
                train_loader, val_loader, test_loader, criterion, rounds, n_vocab, num_clients, fraction, device, lr, momentum, batch_size, wd, seq_length, C, local_steps, "uniform"
            )
            print(f"Validation accuracy: {validation_accuracies[-1]}")
            accuracy_max = max(validation_accuracies)
            if accuracy_max > best_accuracy:
                best_accuracy = accuracy_max
                best_lr = lr
                best_wd = wd

            print("Starting federated training (Skewed Participation)...")
            global_model = CharLSTM(n_vocab, embedding_size, hidden_dim, seq_length, batch_size, lr, num_layers=2)  # Reset global model for skewed participation
            #criterion = nn.NLLLoss() # DA CONTROLLARE QUALE LOSS SI USA QUA SE QUESTA O QUELLA CROSSENTROPY
            criterion = nn.CrossEntropyLoss()  # Loss function
            optimizer = optim.SGD(global_model.parameters(), learning_rate, momentum, 0, weight_decay)  # Optimizer
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)  # Learning rate scheduler
            global_model, validation_accuracies, validation_losses, train_accuracies, train_losses, client_sel_count = server.train_federated(
                train_loader, val_loader, test_loader, criterion, rounds, n_vocab, num_clients, fraction, device, lr, momentum, batch_size, wd, seq_length, C, local_steps, "skewed"
            )
            print(f"Validation accuracy: {validation_accuracies[-1]}")
            accuracy_max = max(validation_accuracies)
            if accuracy_max > best_accuracy:
                best_accuracy = accuracy_max
                best_lr = lr
                best_wd = wd

    print(f"Best parameters: lr = {best_lr} wd = {best_wd} with Accuracy = {best_accuracy}")
    # # Plot federated training performance
    # plt.figure()
    # plt.plot(range(1, len(uniform_losses) + 1), uniform_losses, label="Uniform Participation")
    # plt.plot(range(1, len(skewed_losses) + 1), skewed_losses, label="Skewed Participation")
    # plt.xlabel("Round")
    # plt.ylabel("Loss")
    # plt.legend()
    # plt.title("Federated Training Performance")
    # plt.show()

    # # Plot sampling distributions for skewed participation
    # if len(sampling_distributions) > 0:
    #     plt.figure()
    #     plt.bar(range(len(sampling_distributions[-1])), sampling_distributions[-1])
    #     plt.title("Sampling Distribution (Skewed, Last Round)")
    #     plt.xlabel("Client")
    #     plt.ylabel("Probability")
    #     plt.show()

#     # ====================
#     # IID and Non-IID Sharding Experiments
#     # ====================
#     print("Starting IID and Non-IID experiments...")
#     for num_labels_per_client in num_labels_per_client_list:
#         for local_steps in local_steps_list:
#             # Generate non-IID sharded data
#             non_iid_shards = create_sharding(
#                 centralized_dataset, centralized_dataset.targets, num_clients, iid=False, num_labels_per_client=num_labels_per_client
#             )

#             # Train global model with non-IID shards
#             print(f"Training with Non-IID Sharding: {num_labels_per_client} labels per client")
#             global_model = CharLSTM(vocab_size)
#             non_iid_losses, _ = train_federated(
#                 data_path, global_model, criterion, rounds, non_iid_shards, fraction, device, seq_length, local_steps
#             )

#             # Plot performance
#             plt.figure()
#             plt.plot(range(1, len(non_iid_losses) + 1), non_iid_losses, label=f"Non-IID ({num_labels_per_client} labels)")
#             plt.xlabel("Round")
#             plt.ylabel("Loss")
#             plt.legend()
#             plt.title(f"Non-IID Sharding ({num_labels_per_client} labels), Local Steps: {local_steps}")
#             plt.show()

#     print("All experiments completed!")

if __name__ == "__main__":
    main()
