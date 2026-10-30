"""Token-level attributions for the fine-tuned DistilBERT (Integrated Gradients via Captum)."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def merge_wordpieces(tokens: list[str]) -> list[list[int]]:
    """Group token positions into words: a `##` piece joins the word before it.

    Special tokens (`[CLS]`, `[SEP]`, `[PAD]`) form their own groups.
    """
    groups: list[list[int]] = []
    for i, t in enumerate(tokens):
        if t.startswith("##") and groups and not tokens[groups[-1][0]].startswith("["):
            groups[-1].append(i)
        else:
            groups.append([i])
    return groups


def explain_text(
    model, tokenizer, text: str, max_len: int, target: int | None = None, n_steps: int = 30,
) -> dict[str, Any]:
    """Integrated-gradients attribution of each word to the class `target` (default: predicted).

    The baseline replaces every non-special token with `[PAD]`; attributions are taken on the
    word-embedding layer and summed over the hidden dimension, then over the pieces of a word.
    Positive means the word pushes towards the target class. `delta` is Captum's convergence
    error (how far the attributions are from summing to the output difference).
    """
    from captum.attr import LayerIntegratedGradients

    device = next(model.parameters()).device
    enc = tokenizer(text, truncation=True, max_length=max_len, return_tensors="pt").to(device)
    ids, mask = enc["input_ids"], enc["attention_mask"]
    special = torch.tensor([i in tokenizer.all_special_ids for i in ids[0].tolist()], device=device)
    baseline = torch.where(special, ids[0], torch.full_like(ids[0], tokenizer.pad_token_id))[None]

    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(input_ids=ids, attention_mask=mask).logits, dim=-1)[0]
    pred = int(probs.argmax())
    target = pred if target is None else target

    def forward(input_ids: torch.Tensor) -> torch.Tensor:
        return model(input_ids=input_ids, attention_mask=mask).logits[:, target]

    lig = LayerIntegratedGradients(forward, model.distilbert.embeddings.word_embeddings)
    attr, delta = lig.attribute(
        ids, baselines=baseline, n_steps=n_steps, return_convergence_delta=True)
    piece = attr.sum(dim=-1)[0].detach().cpu().numpy()
    tokens = tokenizer.convert_ids_to_tokens(ids[0].tolist())
    groups = merge_wordpieces(tokens)
    words = ["".join(tokens[i].removeprefix("##") for i in g) for g in groups]
    scores = np.array([piece[g].sum() for g in groups])
    return {
        "ids": ids[0].tolist(), "words": words, "groups": groups, "scores": scores.tolist(),
        "target": target, "pred": pred, "probs": probs.tolist(),
        "delta": float(delta.abs().max()),
        "is_special": [tokens[g[0]].startswith("[") for g in groups],
    }


def top_words(result: dict[str, Any], k: int) -> list[tuple[str, float]]:
    """The `k` real words (not `[CLS]`/`[SEP]`) with the largest attribution to the target."""
    order = np.argsort(result["scores"])[::-1]
    keep = [i for i in order if not result["is_special"][i]][:k]
    return [(result["words"][i], float(result["scores"][i])) for i in keep]


@torch.no_grad()
def prob_without(model, result: dict[str, Any], drop_words: list[int]) -> float:
    """Probability of the target class after deleting the given word groups from the input."""
    device = next(model.parameters()).device
    drop = {p for w in drop_words for p in result["groups"][w]}
    ids = [t for i, t in enumerate(result["ids"]) if i not in drop]
    x = torch.tensor([ids], device=device)
    logits = model(input_ids=x, attention_mask=torch.ones_like(x)).logits
    return float(torch.softmax(logits, dim=-1)[0, result["target"]])


def deletion_check(
    model, results: list[dict[str, Any]], k: int, n_random: int, seed: int,
) -> dict[str, float]:
    """Do the top-attributed words matter more than random ones?

    For each explained review, delete the `k` words with the highest attribution to the
    predicted class, and separately `k` random real words (average over `n_random` draws), and
    record the drop in the predicted class's probability.
    """
    rng = np.random.default_rng(seed)
    top_drop, rand_drop, wins = [], [], 0
    for r in results:
        real = [i for i, s in enumerate(r["is_special"]) if not s]
        if len(real) <= k:
            continue
        base = r["probs"][r["target"]]
        top = [i for i in np.argsort(r["scores"])[::-1] if not r["is_special"][i]][:k]
        d_top = base - prob_without(model, r, top)
        d_rand = float(np.mean([
            base - prob_without(model, r, rng.choice(real, size=k, replace=False).tolist())
            for _ in range(n_random)
        ]))
        top_drop.append(d_top)
        rand_drop.append(d_rand)
        wins += d_top > d_rand
    return {
        "n_reviews": len(top_drop), "k_words_removed": k,
        "mean_prob_drop_top_attributed": float(np.mean(top_drop)),
        "mean_prob_drop_random": float(np.mean(rand_drop)),
        "share_top_beats_random": wins / len(top_drop),
    }
