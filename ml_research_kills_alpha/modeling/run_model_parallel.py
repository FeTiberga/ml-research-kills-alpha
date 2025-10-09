#!/usr/bin/env python3
import os
import sys
import subprocess
import concurrent.futures

from ml_research_kills_alpha.support import MODELS

def run_one(model_name: str, end_year: int, target_col: str) -> None:
    """
    Launch a single-model training process that writes a shard:
    processed/preds/predictions_<MODEL>.csv
    """
    env = os.environ.copy()

    total_cores = os.cpu_count() or 8
    max_workers = min(len(MODELS), max(1, total_cores // 4))
    threads_per_worker = max(1, total_cores // max_workers)
    for var in ["OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS","PYTORCH_NUM_THREADS"]:
        env[var] = str(threads_per_worker)

    python_exe = sys.executable

    cmd = [
        python_exe, "-m", "ml_research_kills_alpha.modeling.prediction_pipeline",
        "--end_year", str(end_year),
        "--target_col", target_col,
        "--models", model_name,
        "--skip_backtest",   # worker only trains & writes shard
    ]
    subprocess.run(cmd, env=env, check=True)

def main() -> None:
    end_year = int(os.environ.get("YEAR", "2023"))
    target_col = os.environ.get("TARGET", "ret")
    total_cores = os.cpu_count() or 8
    max_workers = min(len(MODELS), max(1, total_cores // 4))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(run_one, m, end_year, target_col) for m in MODELS]
        for f in concurrent.futures.as_completed(futs):
            f.result()

if __name__ == "__main__":
    main()
