"""Pydantic request and response models for the API."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

MAX_CHARS = 5000
MAX_BATCH = 32


def _clean(text: str) -> str:
    text = text.strip()
    if not text:
        raise ValueError("text must not be empty or whitespace")
    return text


class AnalyzeRequest(BaseModel):
    """One review to analyze."""

    text: str = Field(..., min_length=1, max_length=MAX_CHARS,
                      description=f"Review text, at most {MAX_CHARS} characters.")
    explain: bool = Field(True, description="Include token-level attributions (slower).")

    @field_validator("text")
    @classmethod
    def _strip(cls, text: str) -> str:
        return _clean(text)


class BatchRequest(BaseModel):
    """Several reviews to analyze."""

    texts: list[str] = Field(..., min_length=1, max_length=MAX_BATCH,
                             description=f"1 to {MAX_BATCH} reviews, each up to {MAX_CHARS} chars.")
    explain: bool = Field(False, description="Include token-level attributions (much slower).")

    @field_validator("texts")
    @classmethod
    def _check(cls, texts: list[str]) -> list[str]:
        for i, t in enumerate(texts):
            if len(t) > MAX_CHARS:
                raise ValueError(f"texts[{i}] has {len(t)} characters, the limit is {MAX_CHARS}")
        return [_clean(t) for t in texts]


class ModelPrediction(BaseModel):
    """A model's label and class probabilities."""

    label: str
    probabilities: dict[str, float]


class TopicResult(BaseModel):
    """Nearest complaint topic of a negative review."""

    id: int
    label: str
    similarity: float = Field(..., description="Cosine similarity to the topic centroid.")


class AspectResult(BaseModel):
    """Sentiment towards one aspect, from the sentences that mention it."""

    sentiment: str
    probabilities: dict[str, float]
    n_sentences: int


class TokenScore(BaseModel):
    """A word and its attribution to the predicted class (positive supports it)."""

    token: str
    score: float


class Explanation(BaseModel):
    """Integrated Gradients attributions for the predicted class."""

    tokens: list[TokenScore]
    top_tokens: list[TokenScore]


class AnalyzeResponse(BaseModel):
    """Full analysis of one review."""

    sentiment: str
    probabilities: dict[str, float]
    baseline: ModelPrediction
    topic: TopicResult | None = Field(None, description="Only for reviews predicted negative.")
    aspects: dict[str, AspectResult]
    explanation: Explanation | None = None
    truncated: bool = Field(..., description="The model reads only the first 128 word pieces.")
    latency_ms: float


class BatchResponse(BaseModel):
    """Analyses in the order of the request."""

    n: int
    results: list[AnalyzeResponse]


class HealthResponse(BaseModel):
    """Service status."""

    status: str
    models: dict[str, str]
