# Project Overview
This project explores **Federated Learning (FL)**, a decentralized learning paradigm where multiple clients collaboratively train a shared global model without sharing their local data. The primary focus is to evaluate the performance of **FedAvg** (Federated Averaging) under different FL scenarios and compare it with centralized learning.

The experiments are conducted on two datasets:

1. **Shakespeare dataset** – Character-level language modeling, representing a realistic non-IID FL setting where clients correspond to different Shakespearean characters.
2. **CIFAR-100 dataset** – Image classification task, exploring the impact of data heterogeneity, client selection strategies, and local update frequencies.
The project follows **Track B: Federated Learning** and is inspired by the methodologies described in **McMahan et al. (2017) and Reddi et al. (2021)**.

# Motivation and Background
## Why Federated Learning?
Traditional deep learning models require centralized data aggregation, which poses challenges related to **privacy, data ownership, and communication efficiency**. Federated Learning mitigates these concerns by enabling clients (e.g., mobile devices, edge nodes) to **train models locally and share only model updates** with a central server.

## Challenges in Federated Learning
- **Data Heterogeneity (Non-IID)** – Clients may have different data distributions, making global model convergence difficult.
- **Client Participation Variability** – Not all clients participate equally, leading to imbalanced updates.
- **Communication Constraints** – Frequent model synchronization can be expensive.

This project investigates these aspects by implementing and analyzing **different FL settings and configurations**.

## Dataset and Preprocessing
### CIFAR-100 Dataset
- **Task:** Image classification across 100 classes.
- **Clients:** Each client is assigned a subset of the dataset.
- **Challenges:**
    - **Non-IID settings** where clients receive only a fraction of the 100 classes.
    - **Class imbalance** across different clients.

###  Shakespeare Dataset (LEAF Benchmark)
- **Task:** Character-level next-character prediction.
- **Clients:** Each client represents a different Shakespearean character.
- **Challenges:**
    - **Non-IID data distribution** due to unique writing styles.
    - **Imbalanced data** – some characters have more text than others.

## Implementation Details
The project is based on **FedAvg (Federated Averaging)**, a widely used optimization algorithm for FL. The key steps include:

1. Centralized Learning Baseline
- A **single global model** is trained using all available data in a traditional supervised learning manner.
- This serves as the **upper bound** for performance comparison.

2. Federated Learning with FedAvg
- Each client trains a local model on its own data.
- A **central server aggregates updates** and improves the global model.
- The FL process consists of:
    1. **Client Selection:** A subset of clients (C%) is chosen per round.
    2. **Local Training:** Each selected client updates the model locally for J steps using Stochastic Gradient Descent (SGD).
    3. **Global Aggregation:** The central server averages model updates.

## Experiments
This project systematically evaluates different federated learning configurations.

1. Effect of Client Participation
    - **Uniform Participation:** Every client has an equal probability of being selected at each round.
    - **Skewed Participation:** Clients participate based on a probability distribution (e.g., Dirichlet distribution with hyperparameter γ), simulating real-world variability

2. Data Distribution (IID vs Non-IID)
    - **IID Setting:** Each client receives randomly sampled data from all classes.
    - **Non-IID Setting:** Clients receive data from a limited number of classes (Nc), where Nc = {1, 5, 10, 50}.

3. Impact of Local Training Steps (J)
    - J = {4, 8, 16} represents how many local updates clients perform before synchronization.
    - **Larger J reduces communication costs but may cause local models to drift away from the global optimum.**
#### Scaling Rules:
When J doubles, the **number of FL rounds is halved** to balance computation and communication.

## Contribution

## Federated Learning with Two-Phase Training 
 
### Overview 
The implementation extends standard Federated Averaging (FedAvg) by introducing an innovative two-phase training approach. This modification aims to enhance model performance through intermediate model shuffling between clients. 
 
### Two-Phase Innovation 
The key innovation lies in splitting each federation round into two distinct training phases: 
 
1. **Phase 1 (Initial Training)** 
   - Selected clients receive global model 
   - Each client trains for J/2 epochs (set automatically in the config for experimenting with both standard and two-phase approach) 
   - Training occurs on local datasets 
   - Models capture initial client-specific features 
 
2. **Intermediate Shuffling** 
   - Client models are randomly redistributed 
   - Each client receives a different model 
   - Ensures knowledge sharing across clients 
 
3. **Phase 2 (Extended Training)** 
   - Clients train received models for J/2 epochs 
   - Training continues on local datasets 
   - Models benefit from different data distributions 
 
4. **Final Aggregation** 
   - Twice-trained models are collected 
   - FedAvg applied with dataset size weighting 
   - Results update global model 
 
### Implementation Benefits 
 
1. **Enhanced Knowledge Sharing** 
   - Models exposed to multiple data distributions 
   - Better feature generalization 
   - Reduced impact of client data heterogeneity 
 
2. **Improved Non-IID Handling** 
   - Significant gains in non-IID scenarios (+3.08% with 50 classes) 
   - Consistent performance in moderate non-IID settings 
   - Better adaptation to data heterogeneity 
 
3. **Flexible Configuration** 
   - Adjustable local epochs (J parameter) 
   - Compatible with various client selection strategies 
   - Maintains FedAvg's core benefits


## Installation

>[!WARNING]
> Please follow the guide below before installing the local environment

This project uses poetry to manage the environment. 

Before installing make sure that the pytorch is using a compatible cuda version for your machine.
There is already a friendly script in the project to update the `pyproject.toml` file to best fit your machine hardware capabilities.

Simply run the script from the root folder of the project with your python global environment
```bash
python detect_cuda.py
```

For more information the script can accept multiple arguments, try running
```bash
python detect_cuda.py -h
```

Anyway the basics idea is that the script will ask if you want to:
1. Update the package to match your cuda version
2. Use the cpu version of `pytorch`

After the script terminate you can safely install the environment with 
```bash
poetry install
```

### If the script doesn't work

If for some reason the script doesn't work, the best and fasted method is to go to the official [pytorch installation page](https://pytorch.org/get-started/locally/), and get there the source knowing you CUDA capabilities, or, if don't have cuda or want to use the cpu just go into the `pyproject.toml` file and remove the source from both `torch` and `torchvision`

Example with custom cuda version:
```toml
[...]
    [tool.poetry.dependencies]
        python = "^3.12"
        torch = { version = "^2.5.1", source = "pytorch" }
        torchvision = { version = "^0.20.1", source = "pytorch" }
[...]
```

Example without the cuda (cpu version):
```toml
[...]
    [tool.poetry.dependencies]
        python = "^3.12"
        torch = "^2.5.1"
        torchvision = "^0.20.1"
[...]
```

>[!NOTE]
> Removing the `source="pytorch"` and leaving the version inside the parenthesis, like `torch = { version = "^2.5.1"}`, would have produced the same result, it's a matter of personal preference at this point.