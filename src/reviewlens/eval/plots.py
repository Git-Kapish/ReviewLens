"""Figures saved under reports/figures."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def plot_curves(history: list[dict[str, Any]], path: str | Path) -> Path:
    """Train/val loss and val macro-F1 per epoch."""
    epochs = [h["epoch"] for h in history]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.4))
    ax1.plot(epochs, [h["train_loss"] for h in history], marker="o", label="train")
    ax1.plot(epochs, [h["val_loss"] for h in history], marker="o", label="val")
    ax1.set(xlabel="epoch", ylabel="class-weighted CE loss", title="Loss")
    ax1.legend()
    ax2.plot(epochs, [h["val_macro_f1"] for h in history], marker="o", color="C2")
    ax2.set(xlabel="epoch", ylabel="macro-F1", title="Validation macro-F1")
    for ax in (ax1, ax2):
        ax.set_xticks(epochs)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
