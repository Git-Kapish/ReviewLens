"""Fine-tune DistilBERT for 3-class sentiment (device-agnostic, class-weighted loss)."""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from reviewlens.config import set_seed
from reviewlens.data.clean import LABELS
from reviewlens.eval.metrics import macro_f1
from reviewlens.models.predict import get_device, predict_logits

logger = logging.getLogger(__name__)


def class_weights(labels: np.ndarray, n_classes: int = len(LABELS)) -> torch.Tensor:
    """Inverse-frequency ('balanced') weights: n / (n_classes * count). Mean over samples is 1."""
    counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
    return torch.tensor(len(labels) / (n_classes * np.maximum(counts, 1)), dtype=torch.float32)


def train(
    cfg: dict[str, Any],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    model_dir: str | Path,
    model: nn.Module | None = None,
) -> dict[str, Any]:
    """Fine-tune and save the best epoch (by validation macro-F1) to `model_dir`.

    Returns the per-epoch history (train/val loss, val macro-F1), the best epoch and timings.
    `model` can be passed in (tests use a tiny random one); otherwise it is loaded from
    `cfg["model_name"]`.
    """
    set_seed(cfg["seed"])
    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    if model is None:
        model = AutoModelForSequenceClassification.from_pretrained(
            cfg["model_name"],
            num_labels=len(LABELS),
            id2label=dict(enumerate(LABELS)),
            label2id={name: i for i, name in enumerate(LABELS)},
        )
    model.to(device)

    weight = None
    if cfg["class_weighted_loss"]:
        weight = class_weights(train_df["label_id"].to_numpy()).to(device)
        logger.info("class weights %s", dict(zip(LABELS, weight.tolist(), strict=True)))
    loss_fn = nn.CrossEntropyLoss(weight=weight)

    def collate(rows: list[tuple[str, int]]):
        enc = tokenizer(
            [t for t, _ in rows],
            truncation=True,
            max_length=cfg["max_len"],
            padding=True,
            return_tensors="pt",
        )
        enc["labels"] = torch.tensor([y for _, y in rows])
        return enc

    rows = list(zip(train_df["text"], train_df["label_id"].astype(int), strict=True))
    gen = torch.Generator().manual_seed(cfg["seed"])
    loader = DataLoader(
        rows, batch_size=cfg["batch_size"], shuffle=True, collate_fn=collate, generator=gen
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    total_steps = cfg["epochs"] * math.ceil(len(rows) / cfg["batch_size"])
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(cfg["warmup_frac"] * total_steps), total_steps
    )
    use_amp = device.type == "cuda" and cfg["fp16"]
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    logger.info("device=%s amp=%s steps=%d", device, use_amp, total_steps)

    val_texts, val_y = val_df["text"].tolist(), val_df["label_id"].to_numpy()
    history: list[dict[str, float]] = []
    best_f1, best_epoch, t0 = -1.0, 0, time.perf_counter()
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        losses = []
        for batch in loader:
            batch = batch.to(device)
            labels = batch.pop("labels")
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(**batch).logits
            loss = loss_fn(logits.float(), labels)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            losses.append(loss.item())
            if len(losses) % 200 == 0:
                recent = np.mean(losses[-200:])
                logger.info("epoch %d step %d loss %.4f", epoch, len(losses), recent)

        val_logits = predict_logits(
            model, tokenizer, val_texts, cfg["max_len"], cfg["eval_batch_size"], device
        )
        val_loss = F.cross_entropy(
            torch.from_numpy(val_logits),
            torch.from_numpy(val_y.copy()),
            weight=None if weight is None else weight.cpu(),
        ).item()
        f1 = macro_f1(val_y, val_logits.argmax(axis=1))
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)),
                        "val_loss": val_loss, "val_macro_f1": f1})
        logger.info("epoch %d: %s", epoch, history[-1])
        if f1 > best_f1:
            best_f1, best_epoch = f1, epoch
            Path(model_dir).mkdir(parents=True, exist_ok=True)
            model.save_pretrained(model_dir)
            tokenizer.save_pretrained(model_dir)

    return {
        "history": history,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_f1,
        "train_seconds": round(time.perf_counter() - t0, 1),
        "device": str(device),
        "mixed_precision": use_amp,
        "n_parameters": sum(p.numel() for p in model.parameters()),
    }
