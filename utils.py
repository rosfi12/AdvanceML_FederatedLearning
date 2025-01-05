import os
import logging
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


def preprocess_shakespeare(file_path, sequence_length=100):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Normalizza il testo
    text = text.lower()
    text = ''.join(c for c in text if c.isalnum() or c.isspace())  # Rimuovi \n e caratteri speciali

    chars = sorted(set(text))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for ch, i in char_to_idx.items()}

    # Calcola la frequenza dei token
    token_counts = {char: text.count(char) for char in chars}
    logging.info(f"Frequenza dei token: {token_counts}")

    # Log del vocabolario
    logging.info(f"Vocabolario ({len(chars)} caratteri): {chars}")

    sequences = []
    for i in range(len(text) - sequence_length):
        seq = text[i:i + sequence_length]
        target = text[i + 1:i + sequence_length + 1]
        sequences.append((seq, target))

    return sequences, char_to_idx, idx_to_char


def get_shakespeare_client_datasets(file_path="./data/shakespeare.txt", batch_size=32, num_clients=5):
    """Load the Shakespeare dataset and split it for clients."""
    try:
        # Preprocessa il testo
        sequences, char_to_idx, idx_to_char = preprocess_shakespeare(file_path, sequence_length=100)

        # Suddividi in training e test (80% training, 20% test)
        split_idx = int(0.8 * len(sequences))
        train_data = sequences[:split_idx]
        test_data = sequences[split_idx:]

        logging.info(f"Numero totale di sequenze: {len(sequences)}")
        logging.info(f"Numero di sequenze di training: {len(train_data)}")
        logging.info(f"Numero di sequenze di test: {len(test_data)}")

        # Suddividi i dati tra i client
        client_train_datasets = []
        partition_size = len(train_data) // num_clients
        for i in range(num_clients):
            client_train = train_data[i * partition_size:(i + 1) * partition_size]
            client_train_datasets.append(
                DataLoader(ShakespeareDataset(client_train, char_to_idx), batch_size=batch_size, shuffle=True)
            )
            logging.info(f"Client {i + 1} ha {len(client_train)} sequenze.")

        # Crea un DataLoader per il test set
        test_loader = DataLoader(
            ShakespeareDataset(test_data, char_to_idx), batch_size=batch_size, shuffle=False
        )

        return client_train_datasets, test_loader, idx_to_char
    except FileNotFoundError:
        logging.error(f"File {file_path} not found!")
        return [], None, {}
    

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



if __name__ == "__main__":
    logging.basicConfig(
    level=logging.INFO,  # Livello del logging
    format="%(asctime)s - %(levelname)s - %(message)s",  # Formato del log
    handlers=[
        logging.StreamHandler(),  # Per vedere i log nel terminale
        logging.FileHandler("training_logs.log", mode="w")  # Per salvare i log in un file
    ]
)
    preprocess_shakespeare("./data/shakespeare.txt")
    client_train_datasets, test_loader, idx_to_char = get_shakespeare_client_datasets()
    logging.info(f"Numero di client: {len(client_train_datasets)}")
    for client_idx, client_loader in enumerate(client_train_datasets):
        for batch_idx, (inputs, labels) in enumerate(client_loader):
            logging.info(f"Client {client_idx + 1}, Batch {batch_idx + 1}: Input shape {inputs.shape}, Label shape {labels.shape}")
            break  # Log solo del primo batch
