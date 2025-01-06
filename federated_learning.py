import copy
import logging

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from centralized_training import ImprovedLSTM, build_model, evaluate_model, train_model
from utils import (
    get_cifar100_dataloader,
    get_shakespeare_client_datasets,
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
    dataset: str,
) -> tuple[list[float], list[float]]:
    """Train a local model on client data."""
    epoch_losses = []
    epoch_metrics = []

    for epoch in range(local_epochs):
        logging.info(f"Local Epoch {epoch + 1}/{local_epochs}")
        loss, metric = train_model(
            model, dataloader, criterion, optimizer, device, dataset
        )
        epoch_losses.append(loss)
        if dataset.lower() == "cifar100":
            epoch_metrics.append(metric)

    return epoch_losses, epoch_metrics


def evaluate_global_model(
    model: nn.Module, dataloader: DataLoader, device: torch.device, dataset: str
) -> tuple[float, float]:
    """Evaluate the global model on a validation set."""
    return evaluate_model(model, dataloader, device, dataset)


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
            vocab_size=num_classes, embedding_dim=256, hidden_dim=512, num_layers=3
        ).to(device)
    else:
        global_model = build_model(architecture, num_classes=num_classes).to(device)

    global_model.train()

    # Federated training loop
    progress_bar = tqdm(total=rounds, desc="Federated Training")
    for round in range(rounds):
        print(f"\nRound {round + 1}/{rounds}")

        client_models = [copy.deepcopy(global_model) for _ in range(clients)]
        client_losses = []
        client_accuracies = []
        client_metrics = []


        for client_idx, client_model in enumerate(client_models):
            print(f"\nTraining Client {client_idx + 1}/{clients}")

            if architecture.lower() == "lstm":
                optimizer = optim.Adam(client_model.parameters(), lr=lr)
            else:
                optimizer = optim.SGD(
                    client_model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4
                )

            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=2
            )
            criterion = nn.CrossEntropyLoss()

            # Train model locally
            epoch_losses, epoch_accuracies = train_local_model(
                client_model,
                client_loaders[client_idx],
                criterion,
                optimizer,
                device,
                local_epochs,
                dataset,
            )

            avg_loss = sum(epoch_losses) / len(epoch_losses)
            client_losses.append(avg_loss)

            if dataset.lower() == "cifar100":
                avg_metric = sum(epoch_accuracies) / len(epoch_accuracies)
                client_metrics.append(avg_metric)

            logging.info(f"Client {client_idx + 1} Loss: {avg_loss:.4f}")
            if dataset.lower() == "cifar100":
                logging.info(f"Client {client_idx + 1} Accuracy: {avg_metric:.2f}%")

        # Aggregate client models
        with torch.no_grad():
            global_state_dict = global_model.state_dict()
            for key in global_state_dict.keys():
                global_state_dict[key] = torch.stack(
                    [client_models[i].state_dict()[key].float() for i in range(clients)]
                ).mean(dim=0)
            global_model.load_state_dict(global_state_dict)

        # Evaluate global model
        eval_loss, eval_metric = evaluate_global_model(
            global_model, testloader, device, dataset
        )
        if dataset.lower() == "shakespeare":
            logging.info(
                f"Round {round + 1} Test Loss: {eval_loss:.4f}, Test Perplexity: {eval_metric:.4f}"
            )
        else:
            logging.info(
                f"Round {round + 1} Test Loss: {eval_loss:.4f}, Test Accuracy: {eval_metric:.2f}%"
            )

        logging.info(f"Round {round + 1} completed.")
        logging.info(
            f"Average Client Loss: {sum(client_losses) / len(client_losses):.4f}"
        )
        if dataset.lower() == "cifar100":
            logging.info(
                f"Average Client Accuracy: {sum(client_metrics) / len(client_metrics):.2f}%"
            )
        scheduler.step(eval_loss)
        progress_bar.update(1)
        progress_bar.set_postfix(
            {
                "Loss": f"{eval_loss:.4f}",
                "Metric": f"{eval_metric:.4f}"
                if dataset.lower() == "shakespeare"
                else f"{eval_metric:.2f}%",
            }
        )

    progress_bar.close()

    # Save the global model
    model_name = f"federated_{architecture}_{dataset}.pth"
    save_in_models_folder(model=global_model, model_name=model_name, feedback=True)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,  # Livello del logging
        format="%(asctime)s - %(levelname)s - %(message)s",  # Formato del log
        handlers=[
            logging.StreamHandler(),  # Per vedere i log nel terminale
            logging.FileHandler(
                "training_logs.log", mode="w"
            ),  # Per salvare i log in un file
        ],
    )

    federated_training(
        dataset="shakespeare",  # Change to "cifar100" for CIFAR-100 dataset
        architecture="lstm",  # Change to "cnn" for CNN architecture
        rounds=10,
        clients=5,
        local_epochs=5,
        batch_size=64,
        lr=0.0005,
    )
