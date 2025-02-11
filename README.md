# 🚀 Project Overview

This project explores **Federated Learning (FL)**, a decentralized learning paradigm where multiple clients collaboratively train a shared global model **without sharing their local data**. The primary focus is to evaluate the performance of **FedAvg** (Federated Averaging) under different FL scenarios and compare it with centralized learning.

## 📂 Datasets Used
The experiments are conducted on two datasets:

1. 📜 **Shakespeare dataset** – Character-level language modeling, representing a realistic **non-IID** FL setting where clients correspond to different Shakespearean characters.
2. 🖼️ **CIFAR-100 dataset** – Image classification task, exploring the impact of **data heterogeneity, client selection strategies, and local update frequencies**.

The project follows **Track B: Federated Learning** and is inspired by the methodologies described in **McMahan et al. (2017) and Reddi et al. (2021)**.

---

# 🎯 Motivation and Background

## 🤖 Why Federated Learning?
Traditional deep learning models require **centralized data aggregation**, which poses challenges related to **privacy, data ownership, and communication efficiency**. Federated Learning mitigates these concerns by enabling clients (e.g., mobile devices, edge nodes) to **train models locally and share only model updates** with a central server.

## ⚠️ Challenges in Federated Learning
- 📊 **Data Heterogeneity (Non-IID)** – Clients may have different data distributions, making global model convergence difficult.
- 🏃 **Client Participation Variability** – Not all clients participate equally, leading to imbalanced updates.
- 🌍 **Communication Constraints** – Frequent model synchronization can be expensive.

This project investigates these aspects by implementing and analyzing **different FL settings and configurations**.

---

## 📊 Dataset and Preprocessing

### 🖼️ CIFAR-100 Dataset
- **Task:** Image classification across 100 classes.
- **Clients:** Each client is assigned a subset of the dataset.
- **Challenges:**
    - **Non-IID settings** where clients receive only a fraction of the 100 classes.
    - **Class imbalance** across different clients.

### 📜 Shakespeare Dataset (LEAF Benchmark)
- **Task:** Character-level next-character prediction.
- **Clients:** Each client represents a different Shakespearean character.
- **Challenges:**
    - **Non-IID data distribution** due to unique writing styles.
    - **Imbalanced data** – some characters have more text than others.

---

## ⚙️ Implementation Details
The project is based on **FedAvg (Federated Averaging)**, a widely used optimization algorithm for FL. The key steps include:

### 🔍 1. Centralized Learning Baseline
- A **single global model** is trained using all available data in a traditional supervised learning manner.
- This serves as the **upper bound** for performance comparison.

### 🤝 2. Federated Learning with FedAvg
- Each client trains a **local model** on its own data.
- A **central server aggregates updates** and improves the global model.
- The FL process consists of:
    1. **Client Selection** 🎯: A subset of clients (C%) is chosen per round.
    2. **Local Training** 🏋️: Each selected client updates the model locally for J steps using Stochastic Gradient Descent (SGD).
    3. **Global Aggregation** 🌎: The central server averages model updates.

---

## 🧪 Experiments
This project systematically evaluates different federated learning configurations:

### 👥 1. Effect of Client Participation
- **Uniform Participation**: Every client has an equal probability of being selected at each round.
- **Skewed Participation**: Clients participate based on a probability distribution (e.g., Dirichlet distribution with hyperparameter γ), simulating real-world variability.

### 🔀 2. Data Distribution (IID vs Non-IID)
- **IID Setting**: Each client receives randomly sampled data from all classes.
- **Non-IID Setting**: Clients receive data from a limited number of classes (Nc), where Nc = {1, 5, 10, 50}.

### 🔧 3. Impact of Local Training Steps (J)
- J = {4, 8, 16} represents how many local updates clients perform before synchronization.
- **Larger J reduces communication costs but may cause local models to drift away from the global optimum.**

📌 **Scaling Rules**: When J doubles, the **number of FL rounds is halved** to balance computation and communication.

---

## 🏆 Contribution: Federated Learning with Two-Phase Training

### 🌟 Overview
The implementation extends standard **Federated Averaging (FedAvg)** by introducing an **innovative two-phase training approach**. This modification enhances model performance through intermediate model shuffling between clients.

### 🔄 Two-Phase Innovation
Each federation round is split into two distinct training phases:

1. **Phase 1 (Initial Training) 🏋️**
   - Selected clients receive the global model.
   - Each client trains for J/2 epochs.
   - Models capture initial client-specific features.

2. **Intermediate Shuffling 🔄**
   - Client models are randomly redistributed.
   - Each client receives a different model.
   - Ensures knowledge sharing across clients.

3. **Phase 2 (Extended Training) 🔥**
   - Clients train received models for J/2 epochs.
   - Models benefit from different data distributions.

4. **Final Aggregation 📊**
   - Twice-trained models are collected.
   - **FedAvg is applied with dataset size weighting**.
   - Results update the global model.

### 📈 Benefits
1. **Enhanced Knowledge Sharing** 🌎 – Models exposed to multiple data distributions for better feature generalization.
2. **Improved Non-IID Handling** 📊 – Significant gains in non-IID scenarios (+3.08% with 50 classes).
3. **Flexible Configuration** 🔧 – Adjustable **local epochs (J parameter)** and compatible with various **client selection strategies**.

---

## 🛠️ Installation

> ⚠️ **Before installing, ensure PyTorch is using a compatible CUDA version for your machine.**

This project uses **Poetry** to manage the environment.

### 🔍 Check CUDA Compatibility
A friendly script is available to update the `pyproject.toml` file to best fit your machine hardware capabilities.

```bash
python detect_cuda.py
```

For more details, run:
```bash
python detect_cuda.py -h
```

After running the script, install the environment with:
```bash
poetry install
```

### 🛠️ Manual Installation
If the script fails, manually set up PyTorch based on your CUDA capabilities by referring to the official [PyTorch installation page](https://pytorch.org/get-started/locally/). Alternatively, modify the `pyproject.toml` file:

✅ **With CUDA support:**
```toml
[tool.poetry.dependencies]
python = "^3.12"
torch = { version = "^2.5.1", source = "pytorch" }
torchvision = { version = "^0.20.1", source = "pytorch" }
```

❌ **Without CUDA (CPU version):**
```toml
[tool.poetry.dependencies]
python = "^3.12"
torch = "^2.5.1"
torchvision = "^0.20.1"
```

> 📝 **Tip:** Removing `source="pytorch"` achieves the same result!

---

🎉 **Now you're all set to run your Federated Learning experiments!** 🚀

