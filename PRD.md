# ReviewLens: Product Requirements Document

**Tagline:** Turn raw customer reviews into decisions: sentiment, complaint themes, aspect-level insight, and trends in one tool.
**Owner:** Kapish | **Type:** NLP portfolio project (Data Science / ML roles) | **Status:** Ready to build

---

## 1. Problem Statement

Product teams receive thousands of reviews and support messages. Star ratings alone hide *why* customers are unhappy. A team needs to know:

1. Is overall sentiment improving or declining?
2. What are the top complaint themes right now?
3. Which product aspects (delivery, quality, price, support) drive negative sentiment?
4. Can we classify a new review instantly and explain the prediction?

**ReviewLens** answers these with a reproducible NLP pipeline, an honest baseline-vs-transformer comparison, and a small deployed app.

## 2. Goals and Non-Goals

| Goals | Non-Goals |
|---|---|
| Sentiment classification (3-class) with a rigorous evaluation | Building a production-scale streaming system |
| Honest comparison: TF-IDF + LR baseline vs fine-tuned DistilBERT | Training a transformer from scratch |
| Topic discovery on negative reviews (BERTopic) | Multilingual support |
| Aspect-level sentiment (lightweight, interpretable) | State-of-the-art leaderboard chasing |
| Trend monitoring of themes over time | User accounts / auth |
| Explainability on sample predictions | Paid APIs (everything must run on free tools) |
| FastAPI service + Streamlit app + Docker | |
| Robustness tests (typos, short text, imbalance) | |

## 3. Success Metrics

| Metric | Target |
|---|---|
| Baseline (TF-IDF + LR) macro-F1 | Recorded as the reference number |
| DistilBERT macro-F1 | Beats baseline; the margin is reported honestly (even if small) |
| Topic coherence (c_v) and manual label quality | Top 8-12 interpretable themes, each with a human-written label |
| API latency (CPU, single review) | Under 500 ms for sentiment, under 2 s for the full analysis |
| Reproducibility | `make setup && make train && make app` works from a clean clone |
| Documentation | README with problem, results table, error analysis, limitations |

## 4. Users and Use Cases

| Persona | Need | Feature |
|---|---|---|
| Product manager | "What are customers complaining about this month?" | Dashboard: theme trends, top complaint topics |
| Support lead | "Triage this incoming review" | Single-review analyzer: sentiment, topic, aspects, explanation |
| Recruiter / interviewer | "Does this person understand NLP evaluation?" | README results table, error analysis, notebooks |

## 5. Data

**Primary dataset:** Amazon Reviews 2023 (McAuley Lab, Hugging Face), **one mid-sized category** (e.g., `Video_Games` or `All_Beauty`), subsampled to ~60K reviews. Chosen because it has a rating, review text, and a **timestamp**, which makes the trend view possible.

**Fallback dataset:** `yelp_review_full` (Hugging Face). It has no timestamps, so replace the trend view with "topic prevalence by star rating".

> Phase 0 task: verify the dataset loads, check the license and fields, and pick the category. Record the decision in `docs/DECISIONS.md`.

**Label mapping (3-class sentiment):**

| Stars | Label |
|---|---|
| 1-2 | negative |
| 3 | neutral |
| 4-5 | positive |

**Splits:** stratified 80/10/10 train/val/test with a fixed seed. For the trend analysis, additionally hold out the **latest time window** as a time-based test to check for drift.

**Known data issues to handle and document:** heavy class imbalance (positive dominates), very short reviews, duplicates, non-English text, rating-text mismatch (sarcasm, wrong star).

## 6. Functional Requirements

### FR1: Data pipeline
- Load, clean (dedupe, drop empty/non-English, normalize whitespace), label-map, split, and save as parquet.
- Keep raw text for transformers; apply heavier normalization only for the TF-IDF path.
- Print and save a data profile: class balance, length distribution, date range.

### FR2: Baseline model
- TF-IDF (word 1-2 grams + optional char n-grams) + Logistic Regression with class weights.
- Light hyperparameter search on validation data only.
- Save the model, vectorizer, and metrics JSON.

### FR3: Transformer model
- Fine-tune `distilbert-base-uncased` for 3-class sentiment (max length 128, 2-3 epochs, weighted loss or class-balanced sampling).
- Train on a free GPU (Colab/Kaggle) via a notebook that imports from `src/`; save the model to `models/`.
- Log train/val curves.

### FR4: Evaluation and error analysis
- Report accuracy, **macro-F1**, per-class precision/recall/F1, and the confusion matrix for both models on the **test set**.
- Evaluate the time-based hold-out split for drift.
- Error analysis: sample at least 50 misclassified reviews, categorize failure types (sarcasm, mixed sentiment, neutral boundary, label noise), and write up the findings with a table.
- Statistical check: bootstrap CI on macro-F1 to show whether the model gap is real.

### FR5: Topic modeling
- Run BERTopic (sentence-transformer embeddings, UMAP, HDBSCAN) on **negative** reviews.
- Reduce to a readable number of topics; manually label each with a short human name.
- Report topic coherence and show representative reviews per topic.

### FR6: Aspect-based sentiment (lightweight)
- Define an aspect lexicon (e.g., delivery, quality, price, support, usability, packaging), adapted to the chosen category.
- Split reviews into sentences, map sentences to aspects via keyword and embedding similarity, and score each sentence with the sentiment model.
- Output per-review aspect → sentiment, and aggregate aspect sentiment across the dataset.
- Evaluate on a small hand-labeled set (about 100 sentences) and report the accuracy honestly.

### FR7: Trend monitoring
- Monthly sentiment share and monthly topic share over time.
- Flag themes with a significant increase (simple z-score or change-point heuristic).

### FR8: Explainability
- LR: top positive/negative n-gram coefficients.
- DistilBERT: token-level attributions (SHAP or Captum) for sample predictions; a static gallery of ~10 examples is enough.

### FR9: Robustness checks
- Perturbation tests: typos, lowercasing/uppercasing, emoji, very short reviews, and negation. Report the macro-F1 change per perturbation for both models.

### FR10: Serving
- **FastAPI**: `POST /analyze` (sentiment + probabilities, topic, aspects, top-token explanation), `POST /analyze/batch`, `GET /health`.
- **Streamlit app** with three pages: *Analyze a review*, *Themes and trends dashboard*, *Model comparison*.
- Dockerfile (+ docker-compose for API + app). Models are loaded once at startup.

## 7. Non-Functional Requirements
- Python 3.11, typed function signatures, docstrings on public functions.
- Config in YAML (`configs/`), no hard-coded paths, and a fixed random seed everywhere.
- Unit tests for the data cleaning, label mapping, aspect mapping, and API schema (pytest).
- Free-tier only: no paid API keys.
- CPU inference must work; GPU is needed only for training.

## 8. Architecture

```
Raw reviews ──► Data pipeline ──► train/val/test parquet
                                      │
              ┌───────────────────────┼────────────────────────┐
              ▼                       ▼                        ▼
      TF-IDF + LR            DistilBERT fine-tune      BERTopic (negatives)
              │                       │                        │
              └──────────┬────────────┘                        │
                         ▼                                     ▼
                Evaluation + error analysis          Topic labels + trends
                         │                                     │
                         └──────────────┬──────────────────────┘
                                        ▼
                         FastAPI service  ◄──►  Streamlit app
                                        │
                                     Docker
```

## 9. Tech Stack

| Layer | Choice |
|---|---|
| Language | Python 3.11 |
| Data | pandas, pyarrow, datasets (Hugging Face) |
| Classical ML | scikit-learn |
| Transformers | transformers, torch, accelerate |
| Topics | bertopic, sentence-transformers, umap-learn, hdbscan |
| Explainability | shap and/or captum |
| Serving | FastAPI, uvicorn, pydantic |
| App | Streamlit, plotly |
| Quality | pytest, ruff |
| Packaging | Docker, docker-compose, Makefile |

## 10. Milestones and Time Estimates

| Phase | Work | Est. time |
|---|---|---|
| 0 | Repo setup, dataset verification, decisions doc | 0.5 h |
| 1 | Data pipeline + profile (FR1) | 1 h |
| 2 | Baseline + evaluation harness (FR2, FR4 part) | 1 h |
| 3 | DistilBERT training on a free GPU (FR3) | 1.5 h |
| 4 | Full evaluation, bootstrap CI, error analysis, robustness (FR4, FR9) | 1.5 h |
| 5 | BERTopic + labeling (FR5) | 1 h |
| 6 | Aspect sentiment + small labeled eval (FR6) | 1.5 h |
| 7 | Trends + explainability (FR7, FR8) | 1 h |
| 8 | FastAPI + Streamlit + Docker (FR10) | 2 h |
| 9 | Tests, README, results table, screenshots | 1 h |
| | **Total** | **~12 h** |

Phases are ordered so that the project is **resume-ready after Phase 4** (baseline + transformer + honest evaluation). Later phases add depth without blocking that milestone.

## 11. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| No local GPU | Train on Colab/Kaggle, and use a subsample (30-40K) with max length 128 |
| BERTopic is slow or unstable on CPU | Cache embeddings to disk, use a smaller embedding model (`all-MiniLM-L6-v2`), cap documents |
| Dataset loading problems on Hugging Face | Fallback dataset (Yelp) defined in Section 5 |
| Transformer barely beats baseline | That's a valid finding: report it, and support it with the error analysis and CI |
| Aspect extraction is noisy | Keep it lightweight and interpretable, and report accuracy on the labeled set honestly |
| Scope creep | Follow the phase order; do not start a new phase until the previous one has tests and written results |

## 12. Deliverables Checklist

- [ ] Public GitHub repo with a clean structure and CI (ruff + pytest)
- [ ] README: problem, data, results table (baseline vs DistilBERT), error analysis, limitations, screenshots, how to run
- [ ] `docs/RESULTS.md` with all metrics and figures
- [ ] `docs/ERROR_ANALYSIS.md`
- [ ] Trained model artifacts (or a script to fetch them) and Docker image
- [ ] Streamlit app running locally (and optionally on Hugging Face Spaces or Streamlit Cloud)
- [ ] 3-4 resume bullets with real numbers

## 13. Resume Bullet Templates (fill in real numbers)

- Built an end-to-end NLP pipeline on N Amazon reviews; fine-tuned DistilBERT to X macro-F1 vs Y for a TF-IDF + LR baseline, with bootstrap CIs and a categorized error analysis of 50+ misclassifications.
- Used BERTopic to extract K complaint themes from negative reviews and built a monthly trend view that flags rising issues.
- Implemented aspect-level sentiment (delivery, quality, price, ...) with Z% accuracy on a hand-labeled set and robustness tests across typo/negation perturbations.
- Deployed the system as a FastAPI service and Streamlit dashboard in Docker, with sub-second CPU inference and token-level explanations.
