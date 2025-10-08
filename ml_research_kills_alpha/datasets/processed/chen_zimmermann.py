# ml_research_kills_alpha/datasets/processed/chen_zimmermann.py
# Cleaner for the Chen & Zimmermann dataset

from pathlib import Path

import pandas as pd
import numpy as np

from ml_research_kills_alpha.datasets.processed.cleaner import Cleaner
from ml_research_kills_alpha.support import START_DATE
import argparse
import re


class ChenZimmermannCleaner(Cleaner):
    """
    Cleans and normalizes Chen & Zimmermann signals.
    Steps:
      - consider data from START_DATE and end_date (parameter)
      - cross-sectional percent-rank features by month to [-1, 1]
      - replace missing values with 0
      - write full-sample and post-2005 variants
    """
    def __init__(self, end_date: str, dataset: pd.DataFrame | None = None):
        super().__init__(dataset_name="chen_zimmermann_signals", dataset=dataset)
        self.start_date = pd.to_datetime(START_DATE, format="%m/%d/%Y")
        self.end_date = pd.to_datetime(end_date, format="%m/%d/%Y")

    def clean(self) -> Path:

        # load the raw signals
        self.logger.info(f"Loading raw anomaly signals from {self.dataset_name}")
        file_name = self.raw_dir / f"{self.dataset_name}.csv"
        if not file_name.exists():
            self.logger.error(f"Raw signals file not found: {file_name}, run download file first")
            raise FileNotFoundError(f"Raw signals file not found: {file_name}, run download file first")

        if self.dataset is not None:
            df = self.dataset
        else:
            df = pd.read_csv(file_name)

        # time filtering
        df = self.filter_on_date(self.start_date, self.end_date, df)

        # copy "BidAskSpread" to "bid_ask_spread_abs" because we need it for portfolio building
        df["bid_ask_spread_abs"] = df["BidAskSpread"]
        
        # change all column names to i_want_this_format: lowercase, underscores for spaces and camel case
        def to_snake_case(name):
            # Replace spaces and hyphens with underscores
            name = re.sub(r"[ -]+", "_", name)
            # Convert CamelCase to snake_case
            name = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', name)
            # Lowercase everything
            return name.lower()
        self.logger.info("Renaming columns to snake_case")
        df.columns = [to_snake_case(c) for c in df.columns]

        # identify feature columns
        meta_cols = {"permno", "date", "yyyymm", "ret", "retx", "siccd", "exchcd", "shrcd", "cusip", "permco", "ticker", "bid_ask_spread_abs"}
        numeric_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
        feature_cols = [c for c in numeric_cols if c not in meta_cols]

        # features percent-ranked by month mapped to [-1, 1]
        if not feature_cols:
            self.logger.error("No numeric features identified. Nothing to rank")
            raise ValueError("No numeric features identified. Nothing to rank")
        self.logger.info(f"Percent-ranking {len(feature_cols)} features to [-1,1].")
        df[feature_cols] = (df.groupby("date")[feature_cols].transform(lambda g: g.rank(pct=True) * 2 - 1))

        # missing values replaced with 0
        missing_count = {col: df[col].isna().sum() for col in feature_cols if col in df.columns and df[col].isna().sum() > 0}
        for col, count in missing_count.items():
            pct = count / len(df) * 100
            self.logger.info(f"Column {col} has {count} missing values ({pct:.2f}%), replacing with 0")
            df[col] = df[col].fillna(0.0)

        # save full dataset
        out_full = self._save_dataframe(df.sort_values(["date", "permno"]).reset_index(drop=True), f"{self.dataset_name}.csv")

        self.logger.info("Cleaned Chen & Zimmermann signals saved")
        self.logger.info(f"Full sample: {out_full}")
        return out_full


def main(end_date: str, dataset: pd.DataFrame | None = None):
    cleaner = ChenZimmermannCleaner(end_date=end_date, dataset=dataset)
    cleaner.clean()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean Chen & Zimmermann signals dataset.")
    parser.add_argument("--end_date", type=str, help=f"End date in MM/DD/YYYY format")
    args = parser.parse_args()
    main(end_date=args.end_date)
