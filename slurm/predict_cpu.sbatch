#!/bin/bash
#SBATCH -J ml-cpu
#SBATCH -p htc
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 01:30:00
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
#SBATCH --array=0-18              # 19 items -> years 2005..2023 (edit list below)

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate

YEARS=(2005 2006 2007 2008 2009 2010 2011 2012 2013 2014 2015 2016 2017 2018 2019 2020 2021 2022 2023)
YEAR=${YEARS[$SLURM_ARRAY_TASK_ID]}

# Runs your end-to-end prediction pipeline:
# - trains models with RollingTrainer and writes preds to processed/preds/
# - runs portfolio backtests and writes results to processed/backtests/
python -m ml_research_kills_alpha.prediction_pipeline \
  --end_year "${YEAR}" \
  --target_col ret \
  --force_ml
