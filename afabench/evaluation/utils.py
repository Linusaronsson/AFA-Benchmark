from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure


def plot_metrics(metrics: dict[str, Any]) -> Figure:
    """Return a figure containing metrics."""
    assert "accuracy_all" in metrics, "Metrics must contain 'accuracy_all'."
    assert "f1_all" in metrics, "Metrics must contain 'f1_all'."
    assert "bce_all" in metrics, "Metrics must contain 'bce_all'."

    budget = len(metrics["accuracy_all"])
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    fig.subplots_adjust(wspace=0.6)
    budgets = np.arange(1, budget + 1, 1)

    # Convert to numpy arrays and handle NaN values
    accuracy_all = np.array(metrics["accuracy_all"])
    f1_all = np.array(metrics["f1_all"])
    bce_all = np.array(metrics["bce_all"])

    # Find valid (non-NaN) indices
    valid_acc = ~np.isnan(accuracy_all)
    valid_f1 = ~np.isnan(f1_all)
    valid_bce = ~np.isnan(bce_all)

    # Convert to regular Python lists for better Plotly compatibility
    acc_x = budgets[valid_acc].tolist()
    acc_y = accuracy_all[valid_acc].tolist()
    f1_x = budgets[valid_f1].tolist()
    f1_y = f1_all[valid_f1].tolist()
    bce_x = budgets[valid_bce].tolist()
    bce_y = bce_all[valid_bce].tolist()

    axs[0].plot(acc_x, acc_y, label="Accuracy")
    axs[0].plot(f1_x, f1_y, label="F1 Score")
    axs[0].set_xlabel("Number of Selected Features (Budget)")
    axs[0].set_ylabel("Score")
    axs[0].set_title("Classification Metrics")
    axs[0].legend(loc="upper left")
    axs[0].grid(True, alpha=0.3)

    axs[1].plot(bce_x, bce_y, color="red")
    axs[1].set_xlabel("Number of Selected Features (Budget)")
    axs[1].set_ylabel("Binary Cross-Entropy")
    axs[1].set_title("Loss Metric")
    axs[1].grid(True, alpha=0.3)

    plt.tight_layout()

    return fig
