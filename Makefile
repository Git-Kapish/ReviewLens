ifeq ($(OS),Windows_NT)
PY := .venv/Scripts/python.exe
else
PY := .venv/bin/python
endif

.PHONY: setup data baseline transformer eval topics aspects api app test lint

setup:  ## create venv, install deps
	uv venv --python 3.11 .venv
	uv pip install --python $(PY) -e ".[dev]"

data:  ## download, clean, split (Phase 0: dataset check only)
	$(PY) scripts/check_dataset.py

baseline:  ## train + evaluate TF-IDF+LR
	@echo "not implemented until Phase 2"

transformer:  ## train DistilBERT (GPU recommended)
	@echo "not implemented until Phase 3"

eval:  ## full evaluation, bootstrap, robustness, error sample
	@echo "not implemented until Phase 4"

topics:  ## BERTopic + trends
	@echo "not implemented until Phase 5"

aspects:  ## aspect scoring + eval
	@echo "not implemented until Phase 6"

api:  ## run FastAPI
	@echo "not implemented until Phase 8"

app:  ## run Streamlit
	@echo "not implemented until Phase 8"

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check .
