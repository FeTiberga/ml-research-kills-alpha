#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = ml_research_kills_alpha
PYTHON_VERSION = 3.13
PYTHON_INTERPRETER = python
AZURE_STORAGE_ACCOUNT = your_storage_account
AZURE_STORAGE_KEY = your_storage_key
AZURE_CONTAINER_NAME = da_cambiare

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
	

## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check
	ruff check


## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format


## Run tests
.PHONY: test
test:
	python -m pytest tests


## Download Data from storage system
.PHONY: sync_data_down
sync_data_down:
	az storage blob download-batch -s $(AZURE_CONTAINER_NAME)/data/ \
		-d data/
	

## Upload Data to storage system
.PHONY: sync_data_up
sync_data_up:
	az storage blob upload-batch -d $(AZURE_CONTAINER_NAME)/data/ \
		-s data/
	

## Export all results as LaTeX tables/figures
.PHONY: latex
latex:
	@echo "📝  Generating LaTeX exports…"
	python -m ml_research_kills_alpha.reports.generate_latex


## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	@bash -c "if [ ! -z `which virtualenvwrapper.sh` ]; then source `which virtualenvwrapper.sh`; mkvirtualenv $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER); else mkvirtualenv.bat $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER); fi"
	@echo ">>> New virtualenv created. Activate with:\nworkon $(PROJECT_NAME)"
	

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

# --------- User-facing knobs (safe defaults) ---------
# Data step (daily/monthly cutoff for building the master panel)
END_DATE    ?=
FORCE_RAW   ?=
FORCE_CLEAN ?=

# Modeling/backtest step
END_YEAR    ?=
TARGET_COL  ?=
TEST        ?=
FORCE_ML    ?=

# Papers collection step
PAPERS_INCLUDE_SUBTIER ?=
PAPERS_MAX_RECORDS     ?=
SCRAPE_SUBMISSION      ?= 0

# Where the parallel runner lives (fallback if no scripts/ folder)
RUN_PAR_SCRIPT := scripts/run_model_parallel.py
ifeq ("$(wildcard $(RUN_PAR_SCRIPT))","")
	RUN_PAR_SCRIPT := run_model_parallel.py
endif

# Sensible thread caps for BLAS/NumExpr/PyTorch inside each worker
export OMP_NUM_THREADS      ?= 8
export MKL_NUM_THREADS      ?= 8
export OPENBLAS_NUM_THREADS ?= 8
export NUMEXPR_NUM_THREADS  ?= 8
export PYTORCH_NUM_THREADS  ?= 8

DATA_FLAGS :=
ifneq ($(strip $(END_DATE)),)
	DATA_FLAGS += --end-date "$(END_DATE)"
endif
ifneq ($(strip $(FORCE_RAW)),)
	DATA_FLAGS += --force-raw
endif
ifneq ($(strip $(FORCE_CLEAN)),)
	DATA_FLAGS += --force-clean
endif

PREDICTION_FLAGS :=
ifneq ($(strip $(END_YEAR)),)
	PREDICTION_FLAGS += --end_year "$(END_YEAR)"
endif
ifneq ($(strip $(TARGET_COL)),)
	PREDICTION_FLAGS += --target_col "$(TARGET_COL)"
endif
ifneq ($(strip $(TEST)),)
	PREDICTION_FLAGS += --test "$(TEST)"
endif
ifneq ($(strip $(FORCE_ML)),)
	PREDICTION_FLAGS += --force_ml
endif

# Compose papers CLI flags
PAPERS_FLAGS :=
ifneq ($(strip $(PAPERS_INCLUDE_SUBTIER)),)
	PAPERS_FLAGS += --include-subtier
endif
ifneq ($(strip $(PAPERS_MAX_RECORDS)),)
	PAPERS_FLAGS += --max-records "$(PAPERS_MAX_RECORDS)"
endif

# 1) Build/refresh processed data (panel)
.PHONY: data
data:
	@echo "Running data pipeline (END_DATE=$(END_DATE), FORCE_RAW=$(FORCE_RAW), FORCE_CLEAN=$(FORCE_CLEAN))"
	$(PYTHON_INTERPRETER) -m ml_research_kills_alpha.datasets.data_pipeline $(DATA_FLAGS)

.PHONY: papers
papers:
	@echo "📚 Collecting ML papers (include_subtier=$(PAPERS_INCLUDE_SUBTIER), max=$(PAPERS_MAX_RECORDS), scrape_submission=$(SCRAPE_SUBMISSION))"
	SCRAPE_SUBMISSION=$(SCRAPE_SUBMISSION) $(PYTHON_INTERPRETER) -m ml_research_kills_alpha.datasets.processed.paper_collector $(PAPERS_FLAGS)

.phony: all_data
all_data: data papers
	@echo "All data steps completed."

# 2) Train ALL models in parallel (each worker writes predictions_<MODEL>.csv)
.PHONY: predictions-parallel
predictions-parallel:
	@echo "Launching parallel training for all models (END_YEAR=$(END_YEAR), TARGET_COL=$(TARGET_COL))"
	YEAR=$(END_YEAR) TARGET=$(TARGET_COL) $(PYTHON_INTERPRETER) $(RUN_PAR_SCRIPT)

# 3) Merge shards into one combined predictions.csv (idempotent)
.PHONY: predictions-merge
predictions-merge:
	@echo "Merging per-model shards into predictions.csv"
	$(PYTHON_INTERPRETER) -m ml_research_kills_alpha.modeling.prediction_pipeline --merge_shards True

# 4) Run portfolio backtests + summary (training is skipped if predictions.csv exists)
.PHONY: predictions-backtest
predictions-backtest:
	@echo "Running backtests (training will be skipped because predictions.csv exists)"
	$(PYTHON_INTERPRETER) -m ml_research_kills_alpha.modeling.prediction_pipeline $(PREDICTION_FLAGS)

# 5) One-liner: do everything
.PHONY: predictions
predictions: predictions-parallel predictions-merge predictions-backtest
	@echo "All models ran, shards merged, and backtests completed."


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
