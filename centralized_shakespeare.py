# Import required libraries for dataset management, model building, training, and visualization.
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from collections import defaultdict
import random
import kagglehub
import shutil
import glob
import re
from collections import defaultdict
import random
import torch
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
        return (torch.zeros(self.num_layers, batch_size),
            torch.zeros(self.num_layers, batch_size))

# ====================
# Centralized Training
# ====================

# Define the centralized training pipeline.
def train_centralized(model, train_data, test_data, val_data, criterion, optimizer, scheduler, epochs, device):
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
    epoch_train_losses = []  # Store training loss for each epoch.
    epoch_train_accuracies = []  # Store training accuracy for each epoch.
    epoch_validation_losses = []  # Store validation loss for each epoch.
    epoch_validation_accuracies = []  # Store validation accuracy for each epoch.
    epoch_test_losses = []  # Store test loss for each epoch.
    epoch_test_accuracies = []  # Store test accuracy for each epoch.

    for epoch in range(epochs):
        total_loss = 0
        correct_predictions = 0
        total_samples = 0

        #input_batches, target_batches = create_batches(train_data, model.batch_size, model.seq_length, model.n_vocab )
        progress = tqdm(train_data, desc=f"Epoch {epoch + 1}/{epochs}")  # Track progress.

        for inputs, targets in progress:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()  # Clear previous gradients.
            outputs, _ = model(inputs)  # Forward pass.
            outputs = outputs.view(-1, model.n_vocab)  # Reshape for loss computation.
            targets = targets.view(-1)  # Reshape for loss computation.
            loss = criterion(outputs, targets)  # Compute loss.
            loss.backward()  # Backpropagation.
            optimizer.step()  # Update weights.

            total_loss += loss.item()
            _, predictions = outputs.max(1)  # Get predictions.
            correct_predictions += (predictions == targets).sum().item()  # Count correct predictions.
            total_samples += targets.size(0)  # Update sample count.
            progress.set_postfix(loss=loss.item())  # Show current loss.

        train_accuracy = correct_predictions / total_samples  # Compute accuracy.
        avg_loss = total_loss / len(train_data)  # Compute average loss.
        epoch_train_losses.append(avg_loss)
        epoch_train_accuracies.append(train_accuracy)
        print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}, Accuracy: {train_accuracy:.4f}")

    scheduler.step()  # Update learning rate (scheduler).

    # Evaluate on the validation set.
    model.eval()
    val_loss, val_accuracy = evaluate_model(model, val_data, criterion, device)
    epoch_validation_losses.append(val_loss)
    epoch_validation_accuracies.append(val_accuracy)
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    # Evaluate on the test set.
    test_loss, test_accuracy = evaluate_model(model, test_data, criterion, device)
    epoch_test_losses.append(test_loss)
    epoch_test_accuracies.append(test_accuracy)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")


    # Final evaluation on test set
    test_loss, test_accuracy = evaluate_model(model, test_data, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    return epoch_train_losses, epoch_train_accuracies, epoch_validation_losses, epoch_validation_accuracies, epoch_test_losses, epoch_test_accuracies


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
# Main Execution
# ====================

def main():
    # Dataset and training configurations
    data_path = "shakespeare.txt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available
    epochs = 20  # Number of epochs for centralized training
    seq_length = 80  # Sequence length for LSTM inputs
    batch_size = 64
    n_vocab = 90 # Character number in vobulary (ASCII)
    learning_rate = [0.01, 0.005, 0.001]
    embedding_size = 8
    hidden_dim = 256
    train_split = 0.9
    momentum = 0.9
    weight_decay = [1e-3, 1e-4, 1e-5]

    # Load data
    train_data, test_data = parse_shakespeare(data_path, train_split)

    # Centralized Dataset Preparation
    train_dataset = ShakespeareDataset(train_data, seq_length=seq_length, n_vocab=n_vocab)
    test_dataset = ShakespeareDataset(test_data, seq_length=seq_length, n_vocab=n_vocab)
    train_size = int(0.9 * len(train_dataset))  # 80% of data for training
    val_size = len(train_dataset) - train_size  # 20% of data for validation
    train_dataset, validation_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # ====================
    # Centralized Training
    # ====================
    print("Starting centralized training...")

    # Saving best result
    best_result = {
        "hyperparameters": None,
        "val_accuracy": 0.0,
        "val_loss": float('inf'),
        "test_loss": float('inf'),
        "test_accuracy": 0.0
    }

    for wd in weight_decay:
        for lr in learning_rate:
            print(f"Learning Rate = {lr} - Weight Decay = {wd}")

            model = CharLSTM(n_vocab, embedding_size, hidden_dim, seq_length, batch_size, lr)  # Initialize LSTM model
            criterion = nn.CrossEntropyLoss()  # Loss function
            optimizer = optim.SGD(model.parameters(), lr, momentum, 0, wd)  # Optimizer
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)  # Learning rate scheduler


            # Train and evaluate centralized model
            train_losses, train_accuracies, validation_losses, validation_accuracies, test_losses, test_accuracies = train_centralized(
                model, train_loader, test_loader, val_loader, criterion, optimizer, scheduler, epochs, device
            )

            if validation_accuracies[-1] > best_result["val_accuracy"]:
                best_result["hyperparameters"] = f"LR={lr}, WD={wd}"
                best_result["val_accuracy"] = validation_accuracies[-1]
                best_result["val_loss"] = validation_losses[-1]
                best_result["test_loss"] = test_losses[-1]
                best_result["test_accuracy"] = test_accuracies[-1]
                print(f"Update best result -> Val Accuracy: {validation_accuracies[-1]:.4f}, Val Loss: {validation_losses[-1]:.4f}, Test Accuracy: {test_accuracies[-1]:.4f}, Test Loss: {test_losses[-1]:.4f}")

        

    # Plot centralized validation performance
    plt.figure()
    plt.plot(range(1, len(validation_losses) + 1), validation_losses, label="Validation Loss")
    plt.plot(range(1, len(validation_accuracies) + 1), validation_accuracies, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()
    plt.title("Centralized Validation Performance")
    plt.show()


    # Plot centralized test performance
    plt.figure()
    plt.plot(range(1, len(test_losses) + 1), test_losses, label="Test Loss")
    plt.plot(range(1, len(test_accuracies) + 1), test_accuracies, label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()
    plt.title("Centralized Test Performance")
    plt.show()

    # Print best parameters found
    print(f"Best parameters:\n{best_result} ")

if __name__ == "__main__":
    main()
