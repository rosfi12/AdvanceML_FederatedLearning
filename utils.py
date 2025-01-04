import os
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms


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
        input_tensor = torch.tensor([self.char_to_idx[char] for char in input_seq], dtype=torch.long)
        target_tensor = torch.tensor([self.char_to_idx[char] for char in target_seq], dtype=torch.long)
        return input_tensor, target_tensor


def get_cifar100_dataloader(batch_size=32, iid=True):
    """Load CIFAR-100 dataset with data augmentation."""
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

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

        return DataLoader(sequences, batch_size=batch_size, shuffle=True)
    except FileNotFoundError:
        print("Shakespeare dataset file not found. Using dummy data instead.")
        # Return dummy data if file not found
        dummy_data = [
            (torch.randint(0, 100, (100,)), torch.randint(0, 100, (100,)))
            for _ in range(1000)
        ]
        return DataLoader(dummy_data, batch_size=batch_size, shuffle=True)

def preprocess_shakespeare(file_path, sequence_length=100):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    chars = sorted(set(text))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for ch, i in char_to_idx.items()}

    sequences = []
    for i in range(len(text) - sequence_length):
        seq = text[i:i + sequence_length]
        target = text[i + 1:i + sequence_length + 1]
        sequences.append((seq, target))

    return sequences, char_to_idx, idx_to_char


# def get_shakespeare_client_datasets(file_path="./data/shakespeare.txt", sequence_length=100, batch_size=32, num_clients=5):
#     sequences, char_to_idx, idx_to_char = preprocess_shakespeare(file_path, sequence_length)

#     split_idx = int(0.8 * len(sequences))
#     train_data = sequences[:split_idx]
#     test_data = sequences[split_idx:]

#     partition_size = len(train_data) // num_clients
#     client_train_datasets = [
#         DataLoader(ShakespeareDataset(train_data[i * partition_size:(i + 1) * partition_size], char_to_idx),
#                    batch_size=batch_size, shuffle=True)
#         for i in range(num_clients)
#     ]
#     test_loader = DataLoader(ShakespeareDataset(test_data, char_to_idx), batch_size=batch_size, shuffle=False)

#     return client_train_datasets, test_loader, idx_to_char



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



def get_shakespeare_client_datasets(file_path="./data/shakespeare.txt", batch_size=32, num_clients=5, iid=True):
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
            seq = text[i:i + sequence_length]
            target = text[i + 1:i + sequence_length + 1]
            sequences.append((seq, target))

        # Suddividi in training e test (80% training, 20% test)
        split_idx = int(0.8 * len(sequences))
        train_data = sequences[:split_idx]
        test_data = sequences[split_idx:]

        # Suddividi i dati tra i client
        client_train_datasets = []
        partition_size = len(train_data) // num_clients
        for i in range(num_clients):
            client_train = train_data[i * partition_size:(i + 1) * partition_size]
            client_train_datasets.append(
                DataLoader(ShakespeareDataset(client_train, char_to_idx), batch_size=batch_size, shuffle=True)
            )

        # Crea un DataLoader per il test set
        test_loader = DataLoader(
            ShakespeareDataset(test_data, char_to_idx), batch_size=batch_size, shuffle=False
        )

        return client_train_datasets, test_loader, idx_to_char
    except FileNotFoundError:
        print(f"File {file_path} not found!")
        return [], None, {}
