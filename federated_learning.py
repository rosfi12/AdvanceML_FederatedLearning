import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from centralized_training import build_model
from utils import save_in_models_folder, split_dataset, get_cifar100_dataloader, get_shakespeare_client_datasets, get_shakespeare_dataloader


def train_local_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int = 1,
) -> float:
    """Train a local model on client data."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for epoch in range(epochs):
        for inputs, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            if outputs.dim() > 1:  # For sequence models like LSTM
                labels = labels[:, -1]  # Use only the final target
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches if num_batches > 0 else float("inf")


def evaluate_global_model(
    model: nn.Module, dataloader: DataLoader, device: torch.device
) -> float:
    """Evaluate the global model on a validation set."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            # LSTM case: Only evaluate the final output in the sequence
            if len(inputs.shape) == 2:  # Shakespeare dataset
                outputs = model(inputs)  # [batch_size, vocab_size]
                labels = labels[:, -1]  # Take the last character as target
            else:
                outputs = model(inputs)  # CIFAR-100 or other datasets
            
            _, predicted = outputs.max(1)  # Get the class predictions
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return 100.0 * correct / total



def federated_training(
    dataset: str = "cifar100",
    architecture: str = "cnn",
    rounds: int = 10,
    clients: int = 5,
    local_epochs: int = 1,
    batch_size: int = 32,
    lr: float = 0.01,
):
    """Run federated learning across multiple clients."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    if dataset.lower() == "cifar100":
        trainloader = get_cifar100_dataloader(batch_size=batch_size, iid=True)
        testloader = DataLoader(
            torchvision.datasets.CIFAR100(
                root="./data",
                train=False,
                download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])
            ),
            batch_size=batch_size,
            shuffle=False,
        )
        client_datasets = split_dataset(trainloader.dataset, num_clients=clients)
        client_loaders = [
            DataLoader(ds, batch_size=batch_size, shuffle=True) for ds in client_datasets
        ]
        num_classes = 100
    elif dataset.lower() == "shakespeare":
        client_loaders, testloader, idx_to_char = get_shakespeare_client_datasets(
            batch_size=batch_size, num_clients=clients
        )
        num_classes = len(idx_to_char)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    # Initialize the global model
    global_model = build_model(architecture, num_classes=num_classes).to(device)
    global_model.train()

    # Federated training loop
    for round in range(rounds):
        print(f"\nRound {round + 1}/{rounds}")

        # Local training on clients
        client_models = [copy.deepcopy(global_model) for _ in range(clients)]
        client_losses = []

        for client_idx, client_model in enumerate(client_models):
            print(f"Training on Client {client_idx + 1}/{clients}")
            optimizer = optim.SGD(
                client_model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4
            )
            criterion = nn.CrossEntropyLoss()
            client_loss = train_local_model(
                client_model,
                client_loaders[client_idx],
                criterion,
                optimizer,
                device,
                epochs=local_epochs,
            )
            client_losses.append(client_loss)

        # Aggregate client models
        with torch.no_grad():
            global_state_dict = global_model.state_dict()
            for key in global_state_dict.keys():
                global_state_dict[key] = torch.stack(
                    [client_models[i].state_dict()[key].float() for i in range(clients)]
                ).mean(dim=0)
            global_model.load_state_dict(global_state_dict)

        # Evaluate the global model
        if testloader is not None:
            test_accuracy = evaluate_global_model(global_model, testloader, device)
            print(f"Test Accuracy: {test_accuracy:.2f}%")
        else:
            print("No test set available for evaluation.")

        print(
            f"Round {round + 1} completed. Average client loss: {sum(client_losses) / len(client_losses):.4f}"
        )

    # Save the global model
    model_name = f"federated_{architecture}_{dataset}.pth"
    save_in_models_folder(model=global_model, model_name=model_name, feedback=True)


if __name__ == "__main__":
    federated_training(
        dataset="shakespeare",  # Change to "cifar100" for CIFAR-100 dataset
        architecture="lstm",    # Change to "cnn" for CNN architecture
        rounds=10,
        clients=5,
        local_epochs=1,
        batch_size=32,
        lr=0.01,
    )
