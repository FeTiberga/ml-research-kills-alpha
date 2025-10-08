# ml_research_kills_alpha/prediction_pipeline.py
from __future__ import annotations

from pathlib import Path
import argparse
import sys
import re

import pandas as pd

from ml_research_kills_alpha.config import PROCESSED_DATA_DIR
from ml_research_kills_alpha.support import Logger
from ml_research_kills_alpha.modeling.algorithms.base_model import Modeler
from ml_research_kills_alpha.modeling.algorithms.elastic_net import ElasticNetModel
from ml_research_kills_alpha.modeling.algorithms.huber_ols import HuberRegressorModel
from ml_research_kills_alpha.modeling.algorithms.neural_networks import FFNNModel
# from ml_research_kills_alpha.modeling.algorithms.lstm import LSTMModel

from ml_research_kills_alpha.modeling.rolling_trainer import RollingTrainer
from ml_research_kills_alpha.modeling.portfolio import Portfolio


logger = Logger()


def _sanitize_model_name(name: str) -> str:
    return re.sub(r"[^\w\-]+", "_", name)


def _read_panel_fast(panel_path: Path) -> pd.DataFrame:
    """
    Load the master panel with the fastest available format.
    """
    parquet_path = panel_path.parent / "master_panel.parquet"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    else:
        logger.warning(f"Parquet file not found. Falling back to CSV.")
        if not panel_path.exists():
            logger.error(f"Panel file not found: {panel_path} - Run data pipeline first.")
            raise FileNotFoundError(f"Panel file not found: {panel_path} - Run data pipeline first.")
    return pd.read_csv(panel_path, low_memory=False)


def build_models(selected_names: list[str] | None = None) -> list[Modeler]:
    """
    Instantiate and return the list of models to train.
    Replace the placeholders with your actual constructors / hyperparams.
    """
    MODEL_REGISTRY = {
        "ENET": lambda: ElasticNetModel(),
        "OLS-H": lambda: HuberRegressorModel(),
        "FFNN2": lambda: FFNNModel(num_layers=2),
        "FFNN3": lambda: FFNNModel(num_layers=3),
        "FFNN4": lambda: FFNNModel(num_layers=4),
        "FFNN5": lambda: FFNNModel(num_layers=5),
        # "LSTM1": lambda: LSTMModel(num_layers=1),
        # "LSTM2": lambda: LSTMModel(num_layers=2),
    }
    names = selected_names or list(MODEL_REGISTRY.keys())
    logger.info(f"Building models: {', '.join(names)}")
    return [MODEL_REGISTRY[n]() for n in names if n in MODEL_REGISTRY]


def save_per_model_shards(preds_all: pd.DataFrame, out_dir: Path) -> list[Path]:
    """
    Save one CSV per model to avoid file conflicts during parallel runs.

    Args:
        preds_all (pd.DataFrame): Tidy predictions with a 'model' column.
        out_dir (Path): Directory where per-model files will be written.

    Returns:
        list[Path]: Paths of written shard files.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    written_paths: list[Path] = []
    model_names = sorted(preds_all["model"].unique().tolist())
    for model_name in model_names:
        safe_name = _sanitize_model_name(model_name)
        shard_path = out_dir / f"predictions_{safe_name}.csv"
        preds_all.loc[preds_all["model"] == model_name].to_csv(shard_path, index=False)
        written_paths.append(shard_path)
    return written_paths


def merge_prediction_shards(in_dir: Path, out_file: Path) -> pd.DataFrame:
    """
    Merge all 'predictions_*.csv' shards in a directory into a single 'predictions.csv'.

    Args:
        in_dir (Path): Directory containing per-model shards.
        out_file (Path): Destination CSV for the merged predictions.

    Returns:
        pd.DataFrame: The merged predictions DataFrame.
    """
    import glob
    shard_paths = sorted(glob.glob(str(in_dir / "predictions_*.csv")))
    if not shard_paths:
        raise FileNotFoundError(f"No prediction shards found in {in_dir}")
    frames = [pd.read_csv(p, low_memory=False) for p in shard_paths]
    merged = pd.concat(frames, axis=0, ignore_index=True)
    # Optional: enforce column order (helps when some shards miss rare columns)
    cols = sorted(merged.columns.tolist())
    merged = merged[cols]
    out_file.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_file, index=False)
    return merged


def step_train_and_predict(panel: pd.DataFrame, models: list[Modeler], end_year: int,
                           target_col: str, out_dir: Path, force_ml: bool) -> pd.DataFrame:
    """
    Train models with RollingTrainer and produce predictions.

    Args:
        panel: DataFrame containing the processed master panel.
        models: List of instantiated Modeler objects to train.
        end_year: Last test year (inclusive).
        target_col: Realized return column used as target
        out_dir: Directory to write predictions and losses.
        force_ml: If True, retrain models even if predictions exist.
    """

    preds_file = out_dir / "predictions.csv"
    if preds_file.exists() and not force_ml:
        logger.info(f"Predictions already exist. Reading -> {preds_file}")
        return pd.read_csv(preds_file, low_memory=False)

    if not models:
        logger.warn("No models were added in build_models().")
    trainer = RollingTrainer(models=models, data=panel, end_year=end_year, target_col=target_col)

    preds_all, losses = trainer.run()

    out_dir.mkdir(parents=True, exist_ok=True)

    # Always write per-model shards (safe for parallel)
    _ = save_per_model_shards(preds_all, out_dir)

    # Only write the combined file when multiple models ran in THIS process
    n_models_in_this_run = preds_all["model"].nunique()
    if n_models_in_this_run > 1:
        preds_all.to_csv(preds_file, index=False)
        logger.info(f"Wrote combined predictions -> {preds_file}")
    else:
        # Single-model run (likely parallel): skip combined file to avoid races
        only_model = preds_all["model"].iloc[0]
        logger.info(f"Single-model run ({only_model}). Wrote shard only; skipping combined file.")

    # Per-year losses: if available per model, consider writing per-model files
    losses_df = pd.DataFrame.from_dict(losses, orient='index')
    losses_df.index.name = 'year'
    if n_models_in_this_run > 1:
        losses_df.to_csv(out_dir / "per_year_losses.csv")
    else:
        only_model = _sanitize_model_name(preds_all["model"].iloc[0])
        losses_df.to_csv(out_dir / f"per_year_losses_{only_model}.csv")

    return preds_all


def step_backtest_portfolios(panel: pd.DataFrame, predictions: pd.DataFrame,
                             target_col: str, out_dir: Path) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """
    For each model, run the decile long-short backtest and write monthly time series and a comparison summary.

    Args:
        panel: DataFrame containing the processed master panel.
        predictions: Tidy predictions DataFrame from RollingTrainer.run().
        target_col: Realized return column used in the portfolio (same as trainer target).
        out_dir: Directory to write per-model monthly results and the summary table.

    Returns:
        (summary_table, results_by_model)
    """
    portfolio = Portfolio(panel=panel, target_column=target_col)
    model_names = sorted(predictions["model"].unique().tolist())

    out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    results_by_model: dict[str, pd.DataFrame] = {}

    for name in model_names:
        logger.info(f"=== Backtest for model: {name} ===")
        merged = portfolio.merge_preds(preds=predictions, model_name=name)
        monthly = portfolio.backtest_decile_long_short(merged, model_name=name)
        results_by_model[name] = monthly

        # write monthly results
        model_file = out_dir / f"backtest_{name}.csv"
        monthly.to_csv(model_file)
        logger.info(f"Wrote monthly backtest -> {model_file}")

        # summary row
        summary = portfolio.summarize_backtest(monthly)
        summary["model"] = name
        summary_rows.append(summary)

    summary_df = pd.DataFrame(summary_rows).set_index("model").sort_values("avg_net_%", ascending=False)
    summary_file = out_dir / "summary.csv"
    summary_df.to_csv(summary_file)
    logger.info(f"Wrote summary -> {summary_file}")

    return


def main():
    """
    End-to-end prediction pipeline:
      1) Train models and generate predictions (RollingTrainer).
      2) Build decile long-short portfolios and compute gross/net (Portfolio).
    """
    parser = argparse.ArgumentParser(description="Train models and backtest portfolios.")
    
    parser.add_argument("--end_year", type=int, default=2023, help="Last test year (inclusive).")
    
    parser.add_argument("--target_col", type=str, default="ret",
                        help="Realized return column (e.g., 'ret' or 'retx').")
    
    parser.add_argument("--test", type=bool, default=False,
                        help="If True, runs on smaller dataset")
    
    parser.add_argument("--force_ml", action="store_true", default=False,
                        help="Force retraining ML models even if predictions exist. Default: False")
    
    parser.add_argument("--merge_shards", action="store_true", default=False,
                        help="Merge predictions_*.csv shards in the preds dir into predictions.csv and exit.")

    
    # models and output data for parallel runs
    parser.add_argument("--models", type=str, default="ALL",
        help="Comma-separated model names (ENET,OLS-H,FFNN2,FFNN3,FFNN4,FFNN5,...) or 'ALL'")

    args = parser.parse_args()
    
    selected = None if args.models.upper() == "ALL" else [s.strip() for s in args.models.split(",")]
    out_root = PROCESSED_DATA_DIR

    # then customize output dirs with suffix to keep runs separate
    panel_path = out_root / ("master_panel_reduced.csv" if args.test else "master_panel.csv")
    preds_dir = out_root / f"preds" if not args.test else out_root / f"preds_reduced"
    bt_dir    = out_root / f"backtests" if not args.test else out_root / f"backtests_reduced"
    
    # If only merging shards, do that and exit early
    if args.merge_shards:
        logger.info(f"Merging prediction shards in {preds_dir} and exiting.")
        merged = merge_prediction_shards(in_dir=preds_dir, out_file=preds_dir / "predictions.csv")
        logger.info(f"Merged {len(merged)} rows into -> {preds_dir / 'predictions.csv'}")
        return
        
    # read panel and models
    panel = _read_panel_fast(panel_path)
    models = build_models(selected)

    logger.info("=== STEP 1: TRAIN & PREDICT ===")
    predictions = step_train_and_predict(
        panel=panel,
        models=models,
        end_year=args.end_year,
        target_col=args.target_col,
        out_dir=preds_dir,
        force_ml=args.force_ml
    )

    logger.info("=== STEP 2: PORTFOLIO BACKTESTS ===")
    step_backtest_portfolios(
        panel=panel,
        predictions=predictions,
        target_col=args.target_col,
        out_dir=bt_dir
    )
    
    logger.info("=== PIPELINE COMPLETE ===")

if __name__ == "__main__":
    sys.exit(main())
