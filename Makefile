.PHONY: help install fmt lint typecheck test test-leakage \
        ingest features train backtest paper-trade clean-derived \
        all check

PY := python -m
CONFIG ?= config/default.yaml

help:
	@grep -E '^[a-zA-Z_-]+:.*?##' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?##"}{printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

install: ## Editable install with dev extras
	pip install -e ".[dev]"

fmt: ## Ruff format
	ruff format src tests

lint: ## Ruff lint
	ruff check src tests

typecheck: ## Mypy
	mypy src

test: ## Run full test suite
	pytest

test-leakage: ## Leakage tests only — must pass before any model run
	pytest -m leakage -v

check: lint typecheck test ## All static + tests

ingest: ## A1: scrape GBGB + ingest Betfair BSP -> races_with_market.parquet
	$(PY) greyhound.ingest.gbgb_scraper   --config $(CONFIG)
	$(PY) greyhound.ingest.gbgb_parser    --config $(CONFIG)
	$(PY) greyhound.ingest.betfair_bsp    --config $(CONFIG)
	$(PY) greyhound.data.joins            --config $(CONFIG)

features: test-leakage ## A2: build features.parquet (blocks on leakage tests)
	$(PY) greyhound.features.pipeline     --config $(CONFIG)

train: ## A3: train + calibrate; writes models/artifacts/<timestamp>/
	$(PY) greyhound.models.lgbm_ranker    --config $(CONFIG)

backtest: ## A4: walk-forward backtest -> backtest_results.parquet + report
	$(PY) greyhound.betting.backtest      --config $(CONFIG)

paper-trade: ## A5: daily paper-trading run
	$(PY) greyhound.live.paper_trade      --config $(CONFIG)

all: ingest features train backtest ## End-to-end Phase A

clean-derived: ## Wipe interim+processed (keeps raw cache)
	rm -rf data/interim data/processed
