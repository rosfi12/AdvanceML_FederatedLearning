from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_federated_baseline():
    """Create side-by-side plots of federated baseline metrics."""
    # Setup paths and find directory
    runs_dir = Path("runs")
    baseline_paths = list(runs_dir.glob("federated_test/lenet_test_j4_st*"))
    if not baseline_paths:
        raise FileNotFoundError("No baseline directory found")

    metrics_dir = baseline_paths[0] / "metrics"

    # Load metrics
    val_metrics = pd.read_csv(metrics_dir / "val_metrics.csv")

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Plot loss
    ax1.plot(val_metrics["step"], val_metrics["loss"], color="#1f77b4", linewidth=2)
    ax1.set_xlabel("Rounds")
    ax1.set_ylabel("Loss")
    ax1.grid(True, alpha=0.3)

    # Plot accuracy
    ax2.plot(val_metrics["step"], val_metrics["accuracy"], color="#1f77b4", linewidth=2)
    ax2.set_xlabel("Rounds")
    ax2.set_ylabel("Accuracy (%)")
    ax2.grid(True, alpha=0.3)

    # Adjust layout
    plt.tight_layout()

    # Save plot
    output_dir = Path("plots")
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / "federated_test_metrics.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_participation_gamma():
    """Create accuracy plot for participation mode with gamma 0.1."""
    # Setup paths
    runs_dir = Path("runs")
    participation_paths = list(
        runs_dir.glob("federated_participation/*participation_skewed_gamma0*")
    )
    if not participation_paths:
        raise FileNotFoundError("No participation directory found")

    metrics_dir = participation_paths[0] / "metrics"
    if not metrics_dir.exists():
        raise FileNotFoundError("No metrics found for gamma 0.1 participation mode")

    # Load metrics
    val_metrics = pd.read_csv(metrics_dir / "val_metrics.csv")

    # Create figure
    plt.figure(figsize=(8, 5))

    # Plot accuracy
    plt.plot(val_metrics["step"], val_metrics["accuracy"], color="#1f77b4", linewidth=2)

    # Configure plot
    plt.xlabel("Rounds")
    plt.ylabel("Accuracy (%)")
    plt.grid(True, alpha=0.3)
    plt.title("Participation Mode (γ=0.1) Validation Accuracy on CIFAR-100")

    # Save plot
    output_dir = Path("plots")
    output_dir.mkdir(exist_ok=True)
    plt.savefig(
        output_dir / "participation_gamma01_accuracy.png", dpi=300, bbox_inches="tight"
    )
    plt.close()


def print_heterogeneity_tables():
    """Print tables comparing heterogeneity study results."""
    runs_dir = Path("runs")
    results = []

    # Find all heterogeneity experiment directories
    pattern = "federated_heterogeneity/lenet_heterogeneity_*"
    for exp_dir in runs_dir.glob(pattern):
        name = exp_dir.name

        mode = "two_phase" if "two_phase" in name else "standard"

        # Parse experiment configuration
        if "noniid" in name.lower():  # Make case insensitive
            distribution = "non-IID"
            try:
                # Extract number of classes (10cls or 50cls)
                classes = name.split("noniid_")[1].split("cls")[0]
            except IndexError:
                print(f"Failed to parse classes from: {name}")
                continue
        elif "iid" in name.lower():
            distribution = "IID"
            classes = "100"
        else:
            print(f"Skipping unknown distribution: {name}")  # Debug line
            continue

        # Extract J value
        try:
            j_value = name.split("_J")[1].split("_")[0]
        except IndexError:
            print(f"Failed to parse J value from: {name}")  # Debug line
            continue

        # Load metrics
        metrics_file = exp_dir / "metrics" / "test_metrics.csv"
        if not metrics_file.exists():
            print(f"No metrics file found at: {metrics_file}")  # Debug line
            continue

        metrics = pd.read_csv(metrics_file)
        best_acc = metrics["accuracy"].max()

        results.append(
            {
                "Training Mode": mode,
                "Distribution": distribution,
                "Classes": int(classes),
                "J": int(j_value),
                "Acc (%)": f"{best_acc:.2f}",
            }
        )

    if not results:
        print("No results found!")
        return

    # Create DataFrame and sort
    df = pd.DataFrame(results)
    df = df.sort_values(["Training Mode", "Distribution", "Classes", "J"])

    # Print tables for each mode
    print("\nStandard Training Results:")
    print(
        df[df["Training Mode"] == "standard"]
        .drop("Training Mode", axis=1)
        .to_string(index=False)
    )

    print("\nComparison of Standard vs Two-Phase Training:")

    # Create pivot table for comparison
    comparison = df.pivot_table(
        index=["Distribution", "Classes", "J"],
        columns="Training Mode",
        values="Acc (%)",
        aggfunc="first",  # Take first value if duplicates
    ).reset_index()

    # Reorder columns and rename for clarity
    comparison = comparison[["Distribution", "Classes", "J", "standard", "two_phase"]]
    comparison.columns = [
        "Distribution",
        "Classes",
        "J",
        "Standard Acc (%)",
        "Two-Phase Acc (%)",
    ]

    # Sort and print
    comparison = comparison.sort_values(["Distribution", "Classes", "J"])
    print(comparison.to_string(index=False))


def plot_heterogeneity_comparison():
    """Create comparison plot of standard vs two-phase for J=8, 50cls."""
    # Setup paths
    runs_dir = Path("runs")
    standard_dir = next(
        runs_dir.glob("federated_heterogeneity/*noniid_50cls_J8_standard*/metrics")
    )


    # Load metrics
    standard_metrics = pd.read_csv(standard_dir / "val_metrics.csv")

    # Create figure
    plt.figure(figsize=(8, 5))

    # Plot both accuracies
    plt.plot(
        standard_metrics["step"],
        standard_metrics["accuracy"],
        color="#1f77b4",
        linewidth=2,
    )

    # Configure plot
    plt.xlabel("Rounds")
    plt.ylabel("Accuracy (%)")
    plt.grid(True, alpha=0.3)
    plt.title("Non-IID Heterogeneity on CIFAR-100 (50 classes, J=8)")

    # Save plot
    output_dir = Path("plots")
    output_dir.mkdir(exist_ok=True)
    plt.savefig(
        output_dir / "heterogeneity_50cls_J8_standard.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


if __name__ == "__main__":
    ... # Here you can call the functions to generate the plots or to print the tables
