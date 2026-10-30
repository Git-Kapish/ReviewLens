ifeq ($(OS),Windows_NT)
PY := .venv/Scripts/python.exe
else
PY := .venv/bin/python
endif

.PHONY: setup data baseline transformer eval topics aspects explain api app test lint strip-nb

setup:  ## create venv, install deps
	uv venv --python 3.11 .venv
	uv pip install --python $(PY) -e ".[dev,notebooks,transformer,topics,explain]"

data:  ## sample, clean, label, split, profile
	$(PY) scripts/build_data.py

baseline:  ## select + train + evaluate TF-IDF+LR
	$(PY) scripts/train_baseline.py

transformer:  ## fine-tune DistilBERT (GPU recommended; see notebooks/02)
	$(PY) scripts/train_transformer.py

eval:  ## bootstrap CIs, drift, error sample, robustness (needs baseline + DistilBERT artifacts)
	$(PY) scripts/run_eval.py
	$(PY) scripts/run_robustness.py --model baseline
	$(PY) scripts/run_robustness.py --model distilbert
	$(PY) scripts/run_eval.py

topics:  ## BERTopic on negative reviews, then monthly trends
	$(PY) scripts/run_topics.py
	$(PY) scripts/run_trends.py

aspects:  ## aspect scoring + eval on the hand-labeled sentences (tune first: scripts/tune_aspects.py)
	$(PY) scripts/run_aspects.py

explain:  ## LR n-grams, DistilBERT attribution gallery, deletion check
	$(PY) scripts/run_explain.py

api:  ## run FastAPI
	@echo "not implemented until Phase 8"

app:  ## run Streamlit
	@echo "not implemented until Phase 8"

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check .

strip-nb:  ## clear notebook outputs before committing
	$(PY) -m jupyter nbconvert --clear-output --inplace notebooks/*.ipynb
