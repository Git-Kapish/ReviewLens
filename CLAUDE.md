# CLAUDE.md: ReviewLens

You are helping build **ReviewLens**, an NLP portfolio project. Read `PRD.md` first; it is the source of truth for scope, requirements, and phase order. If this file and the PRD conflict, ask me before choosing.

## Project summary
Review-intelligence pipeline: 3-class sentiment (TF-IDF+LR baseline vs fine-tuned DistilBERT), BERTopic themes on negative reviews, lightweight aspect-level sentiment, trend monitoring, explainability, robustness checks, and a FastAPI + Streamlit + Docker deployment.

## How to work
1. Work **phase by phase** in the order given in PRD Section 10. Do not start a phase until the previous one is finished and tested.
2. At the start of each phase, state the plan in a few lines, then implement.
3. After each phase, run tests and the relevant script, then report: what was built, the metrics or outputs, and any deviations from the PRD.
4. Prefer small, reviewable commits with clear messages (`feat:`, `fix:`, `docs:`, `test:`).
5. Ask me before: changing the dataset, adding a heavy dependency, or dropping a requirement. Do **not** silently reduce scope.
6. Never fabricate results. Every number in the README or docs must come from a saved metrics file produced by the code.

## Repo structure (create this)
```
reviewlens/
├── PRD.md
├── CLAUDE.md
├── README.md
├── Makefile
├── pyproject.toml          # or requirements.txt + requirements-dev.txt
├── Dockerfile
├── docker-compose.yml
├── configs/
│   ├── data.yaml
│   ├── baseline.yaml
│   ├── transformer.yaml
│   ├── topics.yaml
│   └── aspects.yaml        # aspect lexicon
├── data/                   # gitignored; raw/, processed/, labeled/
├── models/                 # gitignored; baseline/, distilbert/, bertopic/
├── reports/                # metrics JSON, figures (committed small ones)
├── docs/
│   ├── DECISIONS.md
│   ├── RESULTS.md
│   └── ERROR_ANALYSIS.md
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_train_distilbert_colab.ipynb   # thin: imports from src/
│   └── 03_topics_and_trends.ipynb
├── src/reviewlens/
│   ├── __init__.py
│   ├── config.py           # load YAML configs, set seeds
│   ├── data/               # load.py, clean.py, split.py, profile.py
│   ├── models/             # baseline.py, transformer.py, predict.py
│   ├── eval/               # metrics.py, bootstrap.py, errors.py, robustness.py
│   ├── topics/             # bertopic_runner.py, labels.py, trends.py
│   ├── aspects/            # lexicon.py, extract.py, score.py
│   ├── explain/            # lr_coefs.py, token_attr.py
│   └── api/                # main.py, schemas.py, service.py
├── app/
│   └── streamlit_app.py    # pages: Analyze, Themes & Trends, Model Comparison
└── tests/
    ├── test_clean.py
    ├── test_labels_split.py
    ├── test_aspects.py
    └── test_api.py
```

## Conventions
- Python 3.11, type hints on all function signatures, short docstrings on public functions.
- Logic lives in `src/reviewlens/`; notebooks only import and visualize.
- Config in YAML, loaded via `config.py`. No hard-coded paths or magic numbers.
- Single `set_seed(seed)` helper used everywhere (python, numpy, torch). Default seed 42.
- Use `logging`, not `print`, in library code.
- Formatting and lint: `ruff` (`make lint`). Tests: `pytest` (`make test`).
- Each training/eval script writes a metrics JSON to `reports/` with the config used and a timestamp.

## Data rules
- Splits are stratified 80/10/10 with a fixed seed, **plus** a time-based hold-out of the latest window.
- Tune hyperparameters on **validation only**. Touch the test set once per final model for reporting.
- Never leak: fit TF-IDF vectorizers and any scalers on train only.
- Label map: 1-2 stars = negative, 3 = neutral, 4-5 = positive.
- Verify dataset fields/license in Phase 0 and log the decision in `docs/DECISIONS.md`. Fallback: `yelp_review_full` (no timestamps; swap the trend view for topic share by rating).

## Modeling rules
- **Baseline:** TF-IDF (word 1-2 grams, optional char n-grams) + LogisticRegression with `class_weight="balanced"`.
- **Transformer:** `distilbert-base-uncased`, max_len 128, 2-3 epochs, class-weighted loss. Training must run on a free Colab/Kaggle GPU; keep `transformer.py` device-agnostic (`cuda` if available, else `cpu`). Use a subsample if CPU-only.
- Primary metric is **macro-F1**. Always report per-class P/R/F1 and the confusion matrix too.
- Report bootstrap 95% CIs (1000 resamples) on macro-F1 for both models and for the difference.
- Error analysis: sample at least 50 misclassified test reviews, tag each with a failure type (sarcasm, mixed sentiment, neutral boundary, label noise, other), and summarize counts in a table in `docs/ERROR_ANALYSIS.md`.
- Robustness: perturbations (typos, upper/lower case, emoji, truncation to <10 tokens, negation insertion); report the macro-F1 change for both models.

## Topics, aspects, trends
- BERTopic on **negative** reviews only. Use `all-MiniLM-L6-v2` embeddings, **cache embeddings to disk**. Cap documents if needed. Save the model to `models/bertopic/`.
- Topic labels are written by hand (and stored in `configs/topics.yaml` or `reports/topic_labels.json`), not auto-generated and then trusted blindly.
- Aspects: lexicon in `configs/aspects.yaml`; sentence split, keyword + embedding-similarity mapping, then sentence-level sentiment via the sentiment model. Evaluate on ~100 hand-labeled sentences in `data/labeled/` and report accuracy honestly, including failure modes.
- Trends: monthly sentiment share and topic share; flag rising themes with a z-score or simple change-point rule; document the rule used.

## Serving rules
- FastAPI endpoints: `POST /analyze`, `POST /analyze/batch`, `GET /health`. Pydantic schemas in `schemas.py`.
- Load models once at startup (lifespan handler), not per request.
- `/analyze` returns: sentiment label, class probabilities, topic id and label (if negative), aspect sentiments, and top contributing tokens.
- Validate input length (max 5,000 chars), return clear 422/400 errors.
- Streamlit app calls the API (or the service layer directly if `API_URL` is unset). Three pages: Analyze, Themes & Trends, Model Comparison.
- `docker-compose up` should start both API and app.

## Makefile targets (create these)
```
make setup      # create venv, install deps
make data       # download, clean, split
make baseline   # train + evaluate TF-IDF+LR
make transformer# train DistilBERT (GPU recommended)
make eval       # full evaluation, bootstrap, robustness, error sample
make topics     # BERTopic + trends
make aspects    # aspect scoring + eval
make api        # run FastAPI
make app        # run Streamlit
make test       # pytest
make lint       # ruff
```

## Definition of done (per phase)
- Code runs from a clean clone via the Makefile target.
- Tests for the new logic pass (`make test`) and lint is clean.
- Metrics/figures saved under `reports/`; `docs/RESULTS.md` updated with real numbers.
- Short summary posted to me: what changed, key numbers, open questions.

## README requirements (final phase)
Problem statement, dataset and label mapping, results table (baseline vs DistilBERT, with CIs), error analysis highlights, topic list with labels, aspect results, robustness table, architecture diagram, screenshots of the app, how to run (Makefile + Docker), limitations, and future work. Every number must trace to a file in `reports/`.

## Do NOT
- Use paid APIs or require API keys.
- Commit data, model weights, or large files (use `.gitignore`; provide a download/train script).
- Tune on the test set, or report only the best of several test runs.
- Skip tests or write placeholder results.
- Expand scope beyond the PRD without asking.

## First task
Phase 0: scaffold the repo structure above, set up the environment and Makefile, verify the dataset loads (print columns, row count, date range, and class balance for the chosen category), and write `docs/DECISIONS.md` with the dataset choice and rationale. Then stop and report before starting Phase 1.
