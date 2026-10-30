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


def plot_ci_comparison(boot: dict[str, dict], path: str | Path) -> Path:
    """Macro-F1 with 95% CIs for each model on each split (`boot[split][model]`)."""
    fig, ax = plt.subplots(figsize=(6, 3.4))
    models = ["baseline", "distilbert"]
    for j, split in enumerate(boot):
        for i, m in enumerate(models):
            r = boot[split][m]
            x = j + (i - 0.5) * 0.3
            yerr = [[r["point"] - r["ci_low"]], [r["ci_high"] - r["point"]]]
            ax.errorbar(x, r["point"], yerr=yerr,
                        fmt="o", color=f"C{i}", capsize=4, label=m if j == 0 else None)
    ax.set_xticks(range(len(boot)), list(boot))
    ax.set(ylabel="macro-F1 (95% bootstrap CI)", title="Baseline vs DistilBERT")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_robustness(deltas: dict[str, dict[str, dict]], path: str | Path) -> Path:
    """Change in macro-F1 per perturbation, one bar per model, with CI whiskers.

    `deltas[model][perturbation]` has `point`, `ci_low`, `ci_high`.
    """
    models = list(deltas)
    perts = list(next(iter(deltas.values())))
    fig, ax = plt.subplots(figsize=(8, 3.6))
    w = 0.38
    for i, m in enumerate(models):
        pts = [deltas[m][p]["point"] for p in perts]
        err = [[deltas[m][p]["point"] - deltas[m][p]["ci_low"] for p in perts],
               [deltas[m][p]["ci_high"] - deltas[m][p]["point"] for p in perts]]
        ax.bar([k + (i - 0.5) * w for k in range(len(perts))], pts, w, yerr=err, capsize=3,
               label=m, color=f"C{i}")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(len(perts)), perts)
    ax.set(ylabel="change in macro-F1", title="Robustness to input perturbations")
    ax.grid(alpha=0.3, axis="y")
    ax.legend()
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
