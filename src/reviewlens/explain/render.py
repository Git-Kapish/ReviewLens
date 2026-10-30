"""Static HTML rendering of token attributions (no JavaScript, no external assets)."""

from __future__ import annotations

import html
from typing import Any

from reviewlens.data.clean import LABELS

_STYLE = """body{font-family:system-ui,sans-serif;max-width:860px;margin:2rem auto;padding:0 1rem;
line-height:1.9}h1{font-size:1.3rem}.ex{border:1px solid #ccc;border-radius:6px;padding:.8rem 1rem;
margin:1rem 0}.meta{font-size:.85rem;color:#555;line-height:1.4;margin-bottom:.4rem}
span.w{padding:1px 2px;border-radius:3px}.key{font-size:.85rem}"""


def _color(score: float, scale: float) -> str:
    """Green for words pushing towards the predicted class, red for words pushing against it."""
    a = min(abs(score) / scale, 1.0) * 0.75 if scale > 0 else 0.0
    return f"rgba(34,160,60,{a:.2f})" if score >= 0 else f"rgba(215,40,40,{a:.2f})"


def render_example(ex: dict[str, Any]) -> str:
    """One review: header with labels and probabilities, then words shaded by attribution."""
    scores = [s for s, sp in zip(ex["scores"], ex["is_special"], strict=True) if not sp]
    scale = max((abs(s) for s in scores), default=0.0)
    spans = "".join(
        f'<span class="w" style="background:{_color(s, scale)}" title="{s:+.3f}">'
        f"{html.escape(w)}</span> "
        for w, s, sp in zip(ex["words"], ex["scores"], ex["is_special"], strict=True) if not sp
    )
    probs = ", ".join(f"{n} {p:.2f}" for n, p in zip(LABELS, ex["probs"], strict=True))
    note = f" | tag: {html.escape(ex['tag'])}" if ex.get("tag") else ""
    return (
        f'<div class="ex"><div class="meta">{html.escape(ex["id"])} | true: '
        f'<b>{ex["true"]}</b> | predicted: <b>{LABELS[ex["pred"]]}</b> ({probs}){note}</div>'
        f"{spans}</div>"
    )


def render_gallery(examples: list[dict[str, Any]], title: str) -> str:
    """A full HTML page for a list of explained examples."""
    key = ('<p class="key">Shading shows each word\'s Integrated Gradients attribution to the '
           'predicted class: <span class="w" style="background:rgba(34,160,60,.6)">green</span> '
           'supports it, <span class="w" style="background:rgba(215,40,40,.6)">red</span> '
           "argues against it. Intensity is relative within each review.</p>")
    body = "".join(render_example(e) for e in examples)
    return (f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>{html.escape(title)}</title><style>{_STYLE}</style></head>"
            f"<body><h1>{html.escape(title)}</h1>{key}{body}</body></html>")
