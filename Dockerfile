# Two images from one file:  docker build --target api .   and   docker build --target app .
# Model weights are not baked in: mount ./models at /app/models (see docker-compose.yml).
FROM python:3.11-slim AS base
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
COPY configs ./configs
COPY reports ./reports

FROM base AS api
# CPU-only torch keeps the image small; `-e` keeps the package next to configs/ and models/
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install -e ".[api]"
# the sentence encoder is public; fetch it at build time so the container starts offline
RUN python -c "from sentence_transformers import SentenceTransformer; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
EXPOSE 8000
CMD ["uvicorn", "reviewlens.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

FROM base AS app
COPY app ./app
RUN pip install -e ".[app]"
EXPOSE 8501
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501", \
     "--browser.gatherUsageStats=false"]
