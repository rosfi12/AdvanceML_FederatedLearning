import os
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def get_cifar100_dataloader(batch_size=32, iid=True):
    """Load CIFAR-100 dataset with data augmentation."""
    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
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
