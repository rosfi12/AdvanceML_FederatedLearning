import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from centralized_training import ImprovedLSTM, build_model
from utils import (
    get_cifar100_dataloader,
    get_shakespeare_client_datasets,
    get_shakespeare_dataloader,
    save_in_models_folder,
    split_dataset,
)


def train_local_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    local_epochs: int,
) -> tuple[list[float], list[float]]:
    """Train a local model on client data."""
    model.train()
    total_iterations = len(dataloader) * local_epochs
    epoch_losses = []
    epoch_accuracies = []

    # Single progress bar for all epochs
    progress_bar = tqdm(total=total_iterations, desc="Training")

    for epoch in range(local_epochs):
        total_loss = 0.0
        correct, total = 0, 0

        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            # Handle LSTM sequence predictions
            if len(inputs.shape) == 2:  # LSTM case
                labels = labels[:, -1]

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Update metrics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Update progress bar
            progress_bar.set_postfix(
                {
                    "epoch": f"{epoch + 1}/{local_epochs}",
                    "loss": f"{loss.item():.4f}",
                    "acc": f"{100.0 * correct / total:.2f}%",
                }
            )
            progress_bar.update()

        # Record epoch metrics
        avg_loss = total_loss / len(dataloader)
        accuracy = 100.0 * correct / total
        epoch_losses.append(avg_loss)
        epoch_accuracies.append(accuracy)

        # Update description for next epoch
        progress_bar.set_description(f"Completed epoch {epoch + 1}/{local_epochs}")

    progress_bar.close()
    return epoch_losses, epoch_accuracies


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
    local_epochs: int = 5,
    batch_size: int = 32,
    lr: float = 0.001,
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
                transform=transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ]
                ),
            ),
            batch_size=batch_size,
            shuffle=False,
        )
        client_datasets = split_dataset(trainloader.dataset, num_clients=clients)
        client_loaders = [
            DataLoader(ds, batch_size=batch_size, shuffle=True)
            for ds in client_datasets
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
    if architecture.lower() == "lstm":
        global_model = ImprovedLSTM(
            vocab_size=num_classes, embedding_dim=256, hidden_dim=512, num_layers=2
        ).to(device)
    else:
        global_model = build_model(architecture, num_classes=num_classes).to(device)
    global_model.train()

    # Federated training loop
    for round in range(rounds):
        print(f"\nRound {round + 1}/{rounds}")

        client_models = [copy.deepcopy(global_model) for _ in range(clients)]
        client_losses = []
        client_accuracies = []

        for client_idx, client_model in enumerate(client_models):
            print(f"\nTraining Client {client_idx + 1}/{clients}")

            if architecture.lower() == "lstm":
                optimizer = optim.Adam(client_model.parameters(), lr=lr)
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode="min", factor=0.5, patience=2
                )
            else:
                optimizer = optim.SGD(
                    client_model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4
                )

            criterion = nn.CrossEntropyLoss()

            # Train for multiple epochs locally
            epoch_losses, epoch_accuracies = train_local_model(
                client_model,
                client_loaders[client_idx],
                criterion,
                optimizer,
                device,
                local_epochs,
            )

            if architecture.lower() == "lstm":
                scheduler.step(epoch_losses[-1])  # Use last epoch loss

            # Calculate client averages
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            avg_accuracy = sum(epoch_accuracies) / len(epoch_accuracies)
            client_losses.append(avg_loss)
            client_accuracies.append(avg_accuracy)

            print(f"Client {client_idx + 1} Summary:")
            print(f"Average Loss: {avg_loss:.4f}")
            print(f"Average Accuracy: {avg_accuracy:.2f}%")

        # Aggregate client models
        with torch.no_grad():
            global_state_dict = global_model.state_dict()
            for key in global_state_dict.keys():
                global_state_dict[key] = torch.stack(
                    [client_models[i].state_dict()[key].float() for i in range(clients)]
                ).mean(dim=0)
            global_model.load_state_dict(global_state_dict)

        # Evaluate the global model
        if round % 5 == 0 and testloader is not None:
            test_accuracy = evaluate_global_model(global_model, testloader, device)
            print(f"Round {round + 1} Global Test Accuracy: {test_accuracy:.2f}%")

        print(f"Round {round + 1} completed.")
        print(f"Average client loss: {sum(client_losses) / len(client_losses):.4f}")
        print(
            f"Average client accuracy: {sum(client_accuracies) / len(client_accuracies):.2f}%"
        )

    # Save the global model
    model_name = f"federated_{architecture}_{dataset}.pth"
    save_in_models_folder(model=global_model, model_name=model_name, feedback=True)


if __name__ == "__main__":
    federated_training(
        dataset="shakespeare",  # Change to "cifar100" for CIFAR-100 dataset
        architecture="lstm",  # Change to "cnn" for CNN architecture
        rounds=10,
        clients=5,
        local_epochs=5,
        batch_size=64,
        lr=0.01,
    )
