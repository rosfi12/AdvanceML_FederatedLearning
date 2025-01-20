import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import matplotlib.pyplot as plt
import collections
from collections import defaultdict
from json import JSONEncoder
import random
import kagglehub
import shutil
import glob
import re
from tqdm import tqdm  # For progress tracking.

# Regular expressions for parsing Shakespeare text
CHARACTER_RE = re.compile(r'^  ([a-zA-Z][a-zA-Z ]*)\. (.*)')  # Matches character lines
CONT_RE = re.compile(r'^    (.*)')  # Matches continuation lines
COE_CHARACTER_RE = re.compile(r'^([a-zA-Z][a-zA-Z ]*)\. (.*)')  # Special regex for Comedy of Errors
COE_CONT_RE = re.compile(r'^(.*)')  # Continuation for Comedy of Errors


# Get current script directory
SCRIPT_DIR = os.getcwd()

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


def __txt_to_data(txt_dir, seq_length=80):
    """Parses text file in given directory into data for next-character model.

    Args:
        txt_dir: path to text file
        seq_length: length of strings in X
    """
    raw_text = ""
    with open(txt_dir,'r') as inf:
        raw_text = inf.read()
    raw_text = raw_text.replace('\n', ' ')
    raw_text = re.sub(r"   *", r' ', raw_text)
    dataX = []
    dataY = []
    for i in range(0, len(raw_text) - seq_length, 1):
        seq_in = raw_text[i:i + seq_length]
        seq_out = raw_text[i + seq_length]
        dataX.append(seq_in)
        dataY.append(seq_out)
    return dataX, dataY

def parse_data_in(data_dir, users_and_plays_path, raw=False):
    '''
    returns dictionary with keys: users, num_samples, user_data
    raw := bool representing whether to include raw text in all_data
    if raw is True, then user_data key
    removes users with no data
    '''
    with open(users_and_plays_path, 'r') as inf:
        users_and_plays = json.load(inf)
    files = os.listdir(data_dir)
    users = []
    hierarchies = []
    num_samples = []
    user_data = {}
    for f in files:
        user = f[:-4]
        passage = ''
        filename = os.path.join(data_dir, f)
        with open(filename, 'r') as inf:
            passage = inf.read()
        dataX, dataY = __txt_to_data(filename)
        if(len(dataX) > 0):
            users.append(user)
            if raw:
                user_data[user] = {'raw': passage}
            else:
                user_data[user] = {}
            user_data[user]['x'] = dataX
            user_data[user]['y'] = dataY
            hierarchies.append(users_and_plays[user])
            num_samples.append(len(dataY))
    all_data = {}
    all_data['users'] = users
    all_data['hierarchies'] = hierarchies
    all_data['num_samples'] = num_samples
    all_data['user_data'] = user_data
    return all_data

def parse_shakespeare(filepath, train_split=0.8):
    """
    Parses Shakespeare's text into training and testing datasets.
    """
    with open(filepath, "r") as file:
        raw_text = file.read()

    plays_data, _ = process_plays(raw_text)
    _, training_set, testing_set = split_train_test_data(plays_data, 1.0 - train_split)

    total_train = sum(len(lines) for lines in training_set.values())
    total_test = sum(len(lines) for lines in testing_set.values())
    print(f"Training examples: {total_train}")
    print(f"Testing examples: {total_test}")
    
    assert total_train > total_test, "Training set should be larger than test set"

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
    return COE_CHARACTER_RE.match(line) if comedy_of_errors else CHARACTER_RE.match(line)

def detect_continuation_line(line, comedy_of_errors):
    """
    Matches a continuation line of dialogue.
    """
    return COE_CONT_RE.match(line) if comedy_of_errors else CONT_RE.match(line)

def _split_into_plays(shakespeare_full):
    """Splits the full data by play."""
    # List of tuples (play_name, dict from character to list of lines)
    plays = []
    discarded_lines = []  # Track discarded lines.
    slines = shakespeare_full.splitlines(True)[1:]

    # skip contents, the sonnets, and all's well that ends well
    author_count = 0
    start_i = 0
    for i, l in enumerate(slines):
        if 'by William Shakespeare' in l:
            author_count += 1
        if author_count == 2:
            start_i = i - 5
            break
    slines = slines[start_i:]

    current_character = None
    comedy_of_errors = False
    for i, line in enumerate(slines):
        # This marks the end of the plays in the file.
        if i > 124195 - start_i:
            break
        # This is a pretty good heuristic for detecting the start of a new play:
        if 'by William Shakespeare' in line:
            current_character = None
            characters = collections.defaultdict(list)
            # The title will be 2, 3, 4, 5, 6, or 7 lines above "by William Shakespeare".
            if slines[i - 2].strip():
                title = slines[i - 2]
            elif slines[i - 3].strip():
                title = slines[i - 3]
            elif slines[i - 4].strip():
                title = slines[i - 4]
            elif slines[i - 5].strip():
                title = slines[i - 5]
            elif slines[i - 6].strip():
                title = slines[i - 6]
            else:
                title = slines[i - 7]
            title = title.strip()

            assert title, (
                'Parsing error on line %d. Expecting title 2 or 3 lines above.' %
                i)
            comedy_of_errors = (title == 'THE COMEDY OF ERRORS')
            # Degenerate plays are removed at the end of the method.
            plays.append((title, characters))
            continue
        match = _match_character_regex(line, comedy_of_errors)
        if match:
            character, snippet = match.group(1), match.group(2)
            # Some character names are written with multiple casings, e.g., SIR_Toby
            # and SIR_TOBY. To normalize the character names, we uppercase each name.
            # Note that this was not done in the original preprocessing and is a
            # recent fix.
            character = character.upper()
            if not (comedy_of_errors and character.startswith('ACT ')):
                characters[character].append(snippet)
                current_character = character
                continue
            else:
                current_character = None
                continue
        elif current_character:
            match = _match_continuation_regex(line, comedy_of_errors)
            if match:
                if comedy_of_errors and match.group(1).startswith('<'):
                    current_character = None
                    continue
                else:
                    characters[current_character].append(match.group(1))
                    continue
        # Didn't consume the line.
        line = line.strip()
        if line and i > 2646:
            # Before 2646 are the sonnets, which we expect to discard.
            discarded_lines.append('%d:%s' % (i, line))
    # Remove degenerate "plays".
    return [play for play in plays if len(play[1]) > 1], discarded_lines


def _remove_nonalphanumerics(filename):
    return re.sub('\\W+', '_', filename)

def play_and_character(play, character):
    return _remove_nonalphanumerics((play + '_' + character).replace(' ', '_'))

def split_train_test_data(plays, test_fraction=0.2):
    """
    Splits the plays into training and testing datasets by character dialogues.
    """
    skipped_characters = 0
    all_train_examples = collections.defaultdict(list)
    all_test_examples = collections.defaultdict(list)

    def add_examples(example_dict, example_tuple_list):
        for play, character, sound_bite in example_tuple_list:
            example_dict[play_and_character(
                play, character)].append(sound_bite)

    users_and_plays = {}
    for play, characters in plays:
        curr_characters = list(characters.keys())
        for c in curr_characters:
            users_and_plays[play_and_character(play, c)] = play
        for character, sound_bites in characters.items():
            examples = [(play, character, sound_bite)
                        for sound_bite in sound_bites]
            if len(examples) <= 2:
                skipped_characters += 1
                # Skip characters with fewer than 2 lines since we need at least one
                # train and one test line.
                continue
            train_examples = examples
            if test_fraction > 0:
                num_test = max(int(len(examples) * test_fraction), 1)
                train_examples = examples[:-num_test]
                test_examples = examples[-num_test:]
                
                assert len(test_examples) == num_test
                assert len(train_examples) >= len(test_examples)

                add_examples(all_test_examples, test_examples)
                add_examples(all_train_examples, train_examples)

    return users_and_plays, all_train_examples, all_test_examples


def _write_data_by_character(examples, output_directory):
    """Writes a collection of data files by play & character."""
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    for character_name, sound_bites in examples.items():
        filename = os.path.join(output_directory, character_name + '.txt')
        with open(filename, 'w') as output:
            for sound_bite in sound_bites:
                output.write(sound_bite + '\n')


def letter_to_vec(c, n_vocab=90):
    """Converts a single character to a vector index based on the vocabulary size."""
    return ord(c) % n_vocab

def word_to_indices(word, n_vocab=90):
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


class NumpyTensorEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.ndarray, torch.Tensor)):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return super().default(obj)

def save_results_federated(model, train_accuracies, train_losses, test_accuracy, test_loss, client_selection, filename):
    """
    Save federated learning results in both .pth and .json formats.
    Handles PyTorch tensors and NumPy arrays serialization.
    """
    try:
        # Create output directory
        subfolder_path = os.path.join(OUTPUT_DIR, "Federated")
        os.makedirs(subfolder_path, exist_ok=True)
        
        # Define file paths
        filepath_pth = os.path.join(subfolder_path, f"{filename}.pth")
        filepath_json = os.path.join(subfolder_path, f"{filename}.json")
        
        # Prepare results dictionary
        results = {
            'model_state': model.state_dict(),
            'train_accuracies': train_accuracies,
            'train_losses': train_losses,
            'test_accuracy': test_accuracy,
            'test_loss': test_loss,
            'client_count': client_selection
        }
        
        # Save model checkpoint
        torch.save(results, filepath_pth)
        
        # Save JSON metrics with custom encoder
        with open(filepath_json, 'w') as json_file:
            json.dump(results, json_file, indent=4, cls=NumpyTensorEncoder)
            
        print(f"Results saved successfully to {subfolder_path}")
        
    except Exception as e:
        print(f"Error saving results: {str(e)}")
        raise

    def plot_results_federated(train_losses, train_accuracies, filename):   
        # Plot federated training performance
        subfolder_path = os.path.join(OUTPUT_DIR, "Federated")
        os.makedirs(subfolder_path, exist_ok=True)

        file_path = os.path.join(subfolder_path, filename)

        # Create epochs list
        epochs = list(range(1, len(train_losses) + 1))
        
        # Create subplot figure
        plt.figure(figsize=(15, 6))
        
        # Plot Training Loss
        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_losses, label='Train Loss', color='blue')
        plt.xlabel('Rounds', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Federated Training Loss', fontsize=14)
        plt.legend()
        plt.grid(True)
        
        # Plot Training Accuracy 
        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_accuracies, label='Train Accuracy', color='blue')
        plt.xlabel('Rounds', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title('Federated Training Accuracy', fontsize=14)
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{file_path}.png")
        plt.close()


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


# Define the character-level LSTM model for Shakespeare data.
class CharLSTM(nn.Module):
    def __init__(self, n_vocab=90, embedding_dim=8, hidden_dim=256, seq_length=80, num_layers=2):
        """
        Initialize the LSTM model.
        Args:
        - n_vocab: Number of unique characters in the dataset.
        - embedding_dim: Size of the character embedding.
        - hidden_dim: Number of LSTM hidden units.
        - num_layers: Number of LSTM layers.
        - seq_length: Length of input sequences.
        """
        super(CharLSTM, self).__init__()
        self.seq_length = seq_length
        self.n_vocab = n_vocab
        self.embedding_size = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Character embedding layer: Maps indices to dense vectors.
        self.embedding = nn.Embedding(n_vocab, embedding_dim)  # Character embedding layer.
        
        # LSTM layers
        self.lstm_first = nn.LSTM(embedding_dim, hidden_dim, num_layers=1, batch_first=True)  # LSTM first layer
        self.lstm_second = nn.LSTM(embedding_dim, hidden_dim, num_layers=1, batch_first=True)  # LSTM second layer.
        
        # Fully connected layer: Maps LSTM output to vocabulary size.
        self.fc = nn.Linear(hidden_dim, n_vocab)  # Output layer (vocab_size outputs).

    def forward(self, x, hidden=None):
        """
        Forward pass of the model.
        Args:
        - x: Input batch (character indices).
        - hidden: Hidden state for LSTM (default: None, initialized internally).
        Returns:
        - Output logits and the updated hidden state.
        """
        # Embedding layer: Convert indices to embeddings.
        x = self.embedding(x)  
        # First LSTM
        output, _ = self.lstm_first(x, hidden)  # Process through first LSTM layer.
        # Second LSTM
        output, hidden = self.lstm_second(x, hidden)  # Process through second LSTM layer.
        # Fully connected layer: Generate logits for each character.
        output = self.fc(output)

        # Note: Softmax is not applied here because CrossEntropyLoss in PyTorch
        # combines the softmax operation with the computation of the loss. 
        # Adding softmax here would be redundant and could introduce numerical instability.
        return output, hidden

    def init_hidden(self, batch_size):
        """
        Initializes hidden and cell states for the LSTM.
        Args:
        - batch_size: Number of sequences in the batch.
        Returns:
        - A tuple of zero-initialized hidden and cell states.
        """
        return (torch.zeros(self.num_layers, batch_size, self.hidden_dim),
            torch.zeros(self.num_layers, batch_size, self.hidden_dim))

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
    
    total_loss = 0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():  # Disable gradient computation for evaluation.
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            # Initialize hidden state
            state = model.init_hidden(inputs.size(0)) 
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
    selected = np.random.choice(range(num_clients), num_selected, replace=False)
    return selected.tolist()  # Convert to list for consistent indexing



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
    return selected_indices.tolist(), probabilities


# ====================
# Sharding for iid and non-iid splits
# ====================
def create_sharding(data, num_clients, num_classes=90):
    """Create data shards ensuring correct client count"""
    if len(data) == 0:
        raise ValueError("Empty dataset")

    client_data = []
    indices = np.random.permutation(len(data))
    batch_size = max(1, len(data) // num_clients)  # Ensure at least 1 sample per client
    remainder = len(data) % num_clients

    # IID sharding
    if num_clients == num_classes:
        start_idx = 0
        for i in range(num_clients):
            end_idx = start_idx + batch_size + (1 if i < remainder else 0)
            shard = Subset(data, indices[start_idx:end_idx])
            client_data.append(shard)
            start_idx = end_idx
    
    # Non-IID sharding
    else:
        targets = [t[0].item() for _, t in data]
        unique_targets = np.unique(targets)
        samples_per_client = max(1, len(data) // num_clients)
        
        # Create initial shards
        for i in range(num_clients):
            start_idx = i * samples_per_client
            end_idx = start_idx + samples_per_client + (1 if i < remainder else 0)
            client_data.append(Subset(data, indices[start_idx:end_idx]))

    assert len(client_data) == num_clients, f"Created {len(client_data)} shards, expected {num_clients}"
    return client_data

class Client:
    def __init__(self, data_loader, id_client, model, device):
        self.data = data_loader
        self.id_client = id_client
        self.model = model.to(device)
        self.device = device
    

    def train_local_model(self, data_loader, criterion, optimizer, local_steps, device):
        """Train model locally with memory optimization"""
        self.model.train()
        # accumulation_steps = 4
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        try:
            for local_step in range(local_steps):
                batch_loss = 0
                optimizer.zero_grad()
                
                for i, (inputs, targets) in enumerate(data_loader):
                    # Move data to device
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    
                    optimizer.zero_grad()

                    # Initialize hidden state
                    state = self.model.init_hidden(inputs.size(0))
                    state = tuple(s.to(device) for s in state)
                    
                    # Forward pass with memory efficiency
                    outputs, state = self.model(inputs, state)
                    outputs = outputs.view(-1, outputs.size(-1))  # Flatten predictions
                    targets = targets.view(-1) # Flatten targets to match
                    # loss = criterion(outputs, targets) / accumulation_steps
                    # Fix loss calculation
                    # outputs = outputs.contiguous().view(-1, 90)
                    # targets = targets.contiguous().view(-1)
                    
                    loss = criterion(outputs, targets)
                    # batch_loss += loss.item()

                    # Backward pass
                    loss.backward()
                    
                    # Gradient accumulation
                    # if (i + 1) % accumulation_steps == 0:
                    #     torch.nn.utils.clip_grad_norm_(
                    #         self.model.parameters(), 
                    #         max_norm=5.0
                    #     )
                    optimizer.step()
                    
                    # Update metrics
                    # total_loss += loss.item() * inputs.size(0)
                    # _, predictions = outputs.max(1)
                    # correct_predictions += (predictions == targets).sum().item()
                    # total_samples += targets.size(0)

                    # Update metrics - key fix here
                    total_loss += loss.item() * targets.size(0)  # Weight by batch
                    _, predictions = outputs.max(1)  # Get predicted characters
                    correct_predictions += (predictions == targets).sum().item()  # Compare at character level
                    total_samples += targets.size(0)  # Count characters, not sequences
                    
                    # Clear memory
                    # del inputs, targets, outputs, loss, state
                    # if torch.cuda.is_available():
                    #     torch.cuda.empty_cache()
                
                # total_loss += batch_loss / len(data_loader)
                
            # Compute final metrics
            avg_loss = total_loss / total_samples
            accuracy = (correct_predictions / total_samples) * 100
            
            print(f"Client {self.id_client}: Loss={avg_loss:.4f}, Acc={accuracy:.4f}")
            return self.model.state_dict(), avg_loss, accuracy
            
        except RuntimeError as e:
            print(f"Error training client {self.id_client}: {str(e)}")
            return None, float('inf'), 0.0


from copy import deepcopy
class Server:
    def __init__(self, test_data, val_data, global_model, device):
        self.test_data = test_data
        self.val_data = val_data
        self.clients = None
        self.global_model = global_model
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
        - num_clients: Number of all clients.
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
        client_sel_count = np.zeros(num_clients)
        best_model = None
        best_loss = float('inf')
        
        shards = create_sharding(train_loader.dataset, num_clients, num_classes) #each shard represent the training data for one client
        assert len(shards) == num_clients, f"Expected {num_clients} shards, got {len(shards)}"
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
            # for id_client in tqdm(selected_clients, desc=f"Round {round_num + 1}"):
            for id_client in selected_clients:
                client_sel_count[id_client] += 1
                local_model = CharLSTM(self.global_model.embedding.num_embeddings).to(device)  # Create local copy.
                local_model.load_state_dict(self.global_model.state_dict())  # Load global model weights.
                local_model.train()
                # local_model = deepcopy(self.global_model)
                optimizer = optim.SGD(local_model.parameters(), lr=lr, momentum=momentum, weight_decay=wd)
                
                # Load client's dataset.
                client_loader = DataLoader(shards[id_client], batch_size, shuffle=True)

                client = Client(client_loader, id_client, local_model, self.device)

                # Train local model.
                client_local_state, client_loss, client_accuracy = client.train_local_model(client_loader, criterion, optimizer, local_steps, device)
                client_states.append(client_local_state)
                client_losses.append(client_loss)
                client_accuracies.append(client_accuracy)


                # Save local weights for aggregation.
                # local_weights.append({k: v.clone() for k, v in local_model.state_dict().items()})

            # Aggregate local weights into the global model with FedAvg 
            # global_dict = deepcopy(self.global_model.state_dict())

            # tot = sum(client_sizes)
            # for k in global_dict:
            #     global_dict[k] = torch.zeros_like(global_dict[k]) # Weight initialitation to zero

            # loss_tot = 0
            # accuracy_tot = 0

            # for state, size, loss, accuracy in zip(client_states, client_sizes, client_losses, client_accuracies):
            #     for k in global_dict:
            #         global_dict[k] += (state[k] * size / tot)
            #     loss_tot += loss * size
            #     accuracy_tot += accuracy * size

            # global_loss = loss_tot / tot
            # global_accuracy = accuracy_tot / tot

            # FedAvg aggregation
            if client_states:
                global_dict = deepcopy(self.global_model.state_dict())
                tot = sum(client_sizes)
                # global_dict = client_states[0].copy()
                
                # Initialize weights
                for k in global_dict:
                    global_dict[k] = torch.zeros_like(global_dict[k])

                # Metrics
                loss_tot = 0
                accuracy_tot = 0
                
                for state, size, loss, accuracy in zip(client_states, client_sizes, 
                                                     client_losses, client_accuracies):
                    weight = size / tot
                    for k in global_dict:
                        global_dict[k] += (state[k] * weight)
                        
                    # Weight metrics by client size
                    loss_tot += loss * weight
                    accuracy_tot += accuracy * weight

                # Update global model and metrics
                self.global_model.load_state_dict(global_dict)
                train_losses.append(loss_tot)
                train_accuracies.append(accuracy_tot)
                
                print(f"Round {round_num} - Global Loss: {loss_tot:.4f}, Accuracy: {accuracy_tot:.2f}%")

                # self.global_model.load_state_dict(global_dict)  # Update global model.
                
                # train_accuracies.append(global_accuracy)
                # train_losses.append(global_loss)
                
                if loss_tot < best_loss:
                    best_loss = loss_tot
                    best_model = deepcopy(global_dict)

            self.global_model.load_state_dict(best_model)
        
        return self.global_model, train_accuracies, train_losses, validation_accuracies, validation_losses, sampling_distributions


def main():
    # Dataset and training configurations
    data_path = "shakespeare.txt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available
    epochs = 20  # Number of epochs for centralized training
    fraction = 0.1  # Fraction of clients to select each round
    seq_length = 80  # Sequence length for LSTM inputs   
    batch_size = 64 # For local training
    n_vocab = 90 # Character number in vobulary (ASCII)
    embedding_size = 8
    hidden_dim = 256
    train_split = 0.8
    # momentum = 0 # TODO ask to TA if is correct 0
    momentum = 0.9
    learning_rate = 0.01
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
    

    # EXPERIMENTS

    local_steps = [4, 8, 16] #what is called J -> # Number of local training steps
    # Scale the number of rounds inversely with J to maintain a constant computational budget
    num_rounds = {4: 200, 8: 100, 16: 50} # Number of federated communication rounds


    # The first FL baseline
    print("FIRST FL BASELINE")

    num_clients = 100
    num_classes = 100 #iid
    C = 0.1
    local_steps = 4

    rounds = num_rounds[local_steps]

    global_model = CharLSTM(n_vocab, embedding_size, hidden_dim, seq_length, num_layers=2) # Initialize global LSTM model
    server = Server(test_data, val_loader, global_model, device)
    criterion = nn.CrossEntropyLoss()  # Loss function
    optimizer = optim.SGD(global_model.parameters(), learning_rate, momentum, weight_decay)  # Optimizer
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)  # Learning rate scheduler
    global_model, train_accuracies, train_losses, validation_accuracies, validation_losses, client_sel_count = server.train_federated(
        train_loader, val_loader, test_loader, criterion, rounds, num_classes, num_clients, fraction, device, learning_rate, momentum, 
        batch_size, weight_decay, seq_length, C, local_steps, "uniform")

    # Test
    test_loss, test_accuracy = evaluate_model(global_model, test_loader, criterion, device) 
    print(f"Local steps={local_steps} -> Test Accuracy: {test_accuracy}")

    filename = f"First_baseline_Num_classes_{num_classes}_local_steps_{local_steps}"
    save_results_federated(global_model, train_accuracies, train_losses, test_accuracy, test_loss, client_sel_count, filename)
    plot_results_federated(train_losses, train_accuracies, filename)


    # The impact of client participation
    print("THE IMPACT OF CLIENT PARTECIPATION")

    num_clients = 100
    num_classes = 100 #iid
    C = 0.1

    local_steps = 4

    rounds = num_rounds[local_steps]

    print("Uniform partecipation")

    global_model = CharLSTM(n_vocab, embedding_size, hidden_dim, seq_length, num_layers=2) # Initialize global LSTM model
    server = Server(test_data, val_loader, global_model, device)
    criterion = nn.CrossEntropyLoss()  # Loss function
    optimizer = optim.SGD(global_model.parameters(), learning_rate, momentum, weight_decay)  # Optimizer
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)  # Learning rate scheduler
    global_model, train_accuracies, train_losses, validation_accuracies, validation_losses, client_sel_count = server.train_federated(
        train_loader, val_loader, test_loader, criterion, rounds, num_classes, num_clients, fraction, device, learning_rate, momentum, 
        batch_size, weight_decay, seq_length, C, local_steps, "uniform")

    # Test
    test_loss, test_accuracy = evaluate_model(global_model, test_loader, criterion, device) 
    print(f"Local steps={local_steps} -> Test Accuracy: {test_accuracy}")

    filename = f"Uniform_Client_partecipation_Num_classes_{num_classes}_local_steps_{local_steps}"
    save_results_federated(global_model, train_accuracies, train_losses, test_accuracy, test_loss, client_sel_count, filename)
    plot_results_federated(train_losses, train_accuracies, filename)


    print("Skewed partecipation")

    # Values of gamma to test
    gamma_values = [0.1, 0.5, 1.0, 5.0]  # Skewness parameter for Dirichlet sampling

    for gamma in gamma_values:
        global_model = CharLSTM(n_vocab, embedding_size, hidden_dim, seq_length, num_layers=2) # Initialize global LSTM model
        server = Server(test_data, val_loader, global_model, device)
        criterion = nn.CrossEntropyLoss()  # Loss function
        optimizer = optim.SGD(global_model.parameters(), learning_rate, momentum, weight_decay)  # Optimizer
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)  # Learning rate scheduler
        global_model, train_accuracies, train_losses, validation_accuracies, validation_losses, client_sel_count = server.train_federated(
            train_loader, val_loader, test_loader, criterion, rounds, num_classes, num_clients, fraction, device, learning_rate, momentum, 
            batch_size, weight_decay, seq_length, C, local_steps, "skewed", gamma)

        # Test
        test_loss, test_accuracy = evaluate_model(global_model, test_loader, criterion, device) 
        print(f"Local steps={local_steps} -> Test Accuracy: {test_accuracy}")

        filename = f"Skewed_Client_partecipation_Gamma_{gamma}_Num_classes_{num_classes}_local_steps_{local_steps}"
        save_results_federated(global_model, train_accuracies, train_losses, test_accuracy, test_loss, client_sel_count, filename)
        plot_results_federated(train_losses, train_accuracies, filename)


    # Simulate heterogeneous distributions 
    print("SIMULATE HETEROGENEOUS DISTRIBUTIONS")


    print("Non-iid shardings")
    num_clients = 100
    num_classes = [1, 5, 10, 50] # non-iid
    C = 0.1

    local_steps = 4

    rounds = num_rounds[local_steps]

    for nc in num_classes:
        global_model = CharLSTM(n_vocab, embedding_size, hidden_dim, seq_length, num_layers=2) # Initialize global LSTM model
        server = Server(test_data, val_loader, global_model, device)
        criterion = nn.CrossEntropyLoss()  # Loss function
        optimizer = optim.SGD(global_model.parameters(), learning_rate, momentum, weight_decay)  # Optimizer
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)  # Learning rate scheduler
        global_model, train_accuracies, train_losses, validation_accuracies, validation_losses, client_sel_count = server.train_federated(
            train_loader, val_loader, test_loader, criterion, rounds, nc, num_clients, fraction, device, learning_rate, momentum, 
            batch_size, weight_decay, seq_length, C, local_steps, "uniform")

        # Test
        test_loss, test_accuracy = evaluate_model(global_model, test_loader, criterion, device) 
        print(f"Local steps={local_steps} -> Test Accuracy: {test_accuracy}")

        filename = f"Non_iid_Num_classes_{nc}_local_steps_{local_steps}"
        save_results_federated(global_model, train_accuracies, train_losses, test_accuracy, test_loss, client_sel_count, filename)
        plot_results_federated(train_losses, train_accuracies, filename)


    print("iid shardings")
    num_clients = 100
    num_classes = 100 # iid
    C = 0.1

    local_steps_list = [4, 8, 16]  # Varying local steps

    for local_steps in local_steps_list:

        rounds = num_rounds[local_steps]
    
        global_model = CharLSTM(n_vocab, embedding_size, hidden_dim, seq_length, num_layers=2) # Initialize global LSTM model
        server = Server(test_data, val_loader, global_model, device)
        criterion = nn.CrossEntropyLoss()  # Loss function
        optimizer = optim.SGD(global_model.parameters(), learning_rate, momentum, weight_decay)  # Optimizer
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)  # Learning rate scheduler
        global_model, train_accuracies, train_losses, validation_accuracies, validation_losses, client_sel_count = server.train_federated(
            train_loader, val_loader, test_loader, criterion, rounds, nc, num_clients, fraction, device, learning_rate, momentum, 
            batch_size, weight_decay, seq_length, C, local_steps, "uniform")

        # Test
        test_loss, test_accuracy = evaluate_model(global_model, test_loader, criterion, device) 
        print(f"Local steps={local_steps} -> Test Accuracy: {test_accuracy}")

        filename = f"iid_Num_classes_{nc}_local_steps_{local_steps}"
        save_results_federated(global_model, train_accuracies, train_losses, test_accuracy, test_loss, client_sel_count, filename)
        plot_results_federated(train_losses, train_accuracies, filename)



    # # Plot federated training performance
    # subfolder_path = os.path.join(OUTPUT_DIR, "/Federated")
    # os.makedirs(subfolder_path, exist_ok=True)

    # file_path = os.path.join(subfolder_path, filename)

    # # Create a list of epochs for the x-axis
    # epochs = list(range(1, len(train_losses) + 1))

    # # Plot the training loss
    # plt.figure(figsize=(10, 6))
    # plt.plot(epochs, train_losses, label='Train Loss', color='blue')
    # plt.xlabel('Rounds', fontsize=14)
    # plt.ylabel('Loss', fontsize=14)
    # plt.title('Federated Training Loss', fontsize=16)
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig(file_path.replace('.png', '_loss.png'), format='png', dpi=300)
    # plt.close()

    # # Plot the training accuracy
    # plt.figure(figsize=(10, 6))
    # plt.plot(epochs, train_accuracies, label='Train Accuracy', color='blue')
    # plt.xlabel('Rounds', fontsize=14)
    # plt.ylabel('Accuracy', fontsize=14)
    # plt.title('Federated Training Accuracy', fontsize=16)
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig(file_path.replace('.png', '_accuracy.png'), format='png', dpi=300)
    # plt.close()

    # Plot sampling distributions 
    # subfolder_path = os.path.join(OUTPUT_DIR, "/Federated")
    # os.makedirs(subfolder_path, exist_ok=True)

    # file_path = os.path.join(subfolder_path, filename)
    # num_clients = len(client_sel_count)
    # plt.figure(figsize=(10, 6))
    # plt.bar(range(num_clients), client_sel_count, alpha=0.7, edgecolor='black')
    # plt.xlabel("Client ID", fontsize=14)
    # plt.ylabel("Selection Count", fontsize=14)
    # plt.title("Client Selection Frequency", fontsize=16)
    # plt.xticks(range(num_clients), fontsize=10, rotation=90 if num_clients > 20 else 0)
    # plt.tight_layout()
    # plt.savefig(file_path, format="png", dpi=300)
    # plt.close()

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
