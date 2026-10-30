"""FastAPI application: POST /analyze, POST /analyze/batch, GET /health.

Run with `uvicorn reviewlens.api.main:app`. Models are loaded once in the lifespan handler.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from reviewlens.api.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    BatchRequest,
    BatchResponse,
    HealthResponse,
)

logger = logging.getLogger(__name__)


def create_app(service: Any | None = None) -> FastAPI:
    """Build the app. Pass a ready `service` (tests do); otherwise it is loaded at startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if service is None:
            from reviewlens.api.service import ReviewLensService

            app.state.service = ReviewLensService()
        else:
            app.state.service = service
        yield

    app = FastAPI(title="ReviewLens", version="0.1.0", lifespan=lifespan,
                  description="Sentiment, complaint topic, aspects and explanations for reviews.")

    @app.exception_handler(Exception)
    async def unexpected(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled error on %s", request.url.path)
        return JSONResponse(status_code=500, content={"detail": "internal error"})

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request) -> dict[str, Any]:
        return {"status": "ok", "models": request.app.state.service.health()}

    @app.post("/analyze", response_model=AnalyzeResponse)
    def analyze(body: AnalyzeRequest, request: Request) -> dict[str, Any]:
        return request.app.state.service.analyze(body.text, explain=body.explain)

    @app.post("/analyze/batch", response_model=BatchResponse)
    def analyze_batch(body: BatchRequest, request: Request) -> dict[str, Any]:
        results = request.app.state.service.analyze_batch(body.texts, explain=body.explain)
        return {"n": len(results), "results": results}

    return app


app = create_app()
