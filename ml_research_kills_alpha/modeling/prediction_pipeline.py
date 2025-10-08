# ml_research_kills_alpha/prediction_pipeline.py
from __future__ import annotations

from pathlib import Path
import argparse
import sys

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


def build_models() -> list[Modeler]:
    """
    Instantiate and return the list of models to train.
    Replace the placeholders with your actual constructors / hyperparams.
    """
    models: list[Modeler] = []
    models.append(ElasticNetModel())
    models.append(HuberRegressorModel())
    models.append(FFNNModel(num_layers=2))
    models.append(FFNNModel(num_layers=3))
    models.append(FFNNModel(num_layers=4))
    models.append(FFNNModel(num_layers=5))
    # models.append(LSTMModel())
    return models


def step_train_and_predict(panel_path: Path, end_year: int,
                           target_col: str, out_dir: Path, force_ml: bool) -> pd.DataFrame:
    """
    Train models with RollingTrainer and produce a tidy predictions table.

    Args:
        panel_path: Path to the processed master panel CSV.
        end_year: Last test year (inclusive).
        target_col: Realized return column to predict (e.g., 'ret' or 'ret_total').
        out_dir: Directory for writing the predictions CSV.

    Returns:
        predictions
    """
    
    # if the files already exist and not forcing, skip
    preds_file = out_dir / "predictions.csv"
    if preds_file.exists() and not force_ml:
        logger.info(f"Predictions already exis. Reading existing predictions -> {preds_file}")
        preds_all = pd.read_csv(preds_file, low_memory=False)
        return preds_all

    logger.info(f"Reading panel -> {panel_path}")
    panel = pd.read_csv(panel_path, low_memory=False)

    models = build_models()
    if not models:
        logger.warn("No models were added in build_models(). Add your models before running.")
    trainer = RollingTrainer(models=models, data=panel, end_year=end_year, target_col=target_col)

    preds_all, losses = trainer.run()

    out_dir.mkdir(parents=True, exist_ok=True)
    preds_file = out_dir / "predictions.csv"
    preds_all.to_csv(preds_file, index=False)
    logger.info(f"Wrote predictions -> {preds_file}")
    
    losses_df = pd.DataFrame.from_dict(losses, orient='index')
    losses_df.index.name = 'year'
    losses_df.to_csv(out_dir / "per_year_losses.csv")
    logger.info(f"Wrote per-year losses -> {out_dir / 'per_year_losses.csv'}")

    return preds_all


def step_backtest_portfolios(panel_path: Path, predictions: pd.DataFrame,
                             target_col: str, out_dir: Path) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """
    For each model, run the decile long-short backtest and write monthly time series and a comparison summary.

    Args:
        panel_path: Path to the processed master panel CSV.
        predictions: Tidy predictions DataFrame from RollingTrainer.run().
        target_col: Realized return column used in the portfolio (same as trainer target).
        out_dir: Directory to write per-model monthly results and the summary table.

    Returns:
        (summary_table, results_by_model)
    """
    logger = Logger()
    panel = pd.read_csv(panel_path, low_memory=False)

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
    args = parser.parse_args()


    out_root = PROCESSED_DATA_DIR
    if args.test:
        panel_path = PROCESSED_DATA_DIR / "master_panel_reduced.csv"
        preds_dir = out_root / "preds_reduced"
        bt_dir = out_root / "backtests_reduced"
    else:
        panel_path = PROCESSED_DATA_DIR / "master_panel.csv"
        preds_dir = out_root / "preds"
        bt_dir = out_root / "backtests"

    logger.info("=== STEP 1: TRAIN & PREDICT ===")
    predictions = step_train_and_predict(
        panel_path=panel_path,
        end_year=args.end_year,
        target_col=args.target_col,
        out_dir=preds_dir,
        force_ml=args.force_ml
    )

    logger.info("=== STEP 2: PORTFOLIO BACKTESTS ===")
    step_backtest_portfolios(
        panel_path=panel_path,
        predictions=predictions,
        target_col=args.target_col,
        out_dir=bt_dir
    )
    
    logger.info("=== PIPELINE COMPLETE ===")

if __name__ == "__main__":
    sys.exit(main())
