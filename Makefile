PYTHON=python
VENV=.venv

.PHONY: install dev test lint format run

install:
	$(PYTHON) -m venv $(VENV)
	. $(VENV)/bin/activate && pip install --upgrade pip setuptools wheel && pip install -r requirements.txt

dev:
	. $(VENV)/bin/activate && pip install -r requirements-dev.txt

test:
	conda run -n week1 pytest -q

lint:
	. $(VENV)/bin/activate && flake8 src tests

format:
	. $(VENV)/bin/activate && black src tests

run:
	$(PYTHON) src/main.py
