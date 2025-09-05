# ml_research_kills_alpha/datasets/processed/chen_zimmermann.py
# Cleaner for the Chen & Zimmermann dataset

from pathlib import Path

import pandas as pd
import numpy as np

from ml_research_kills_alpha.datasets.processed.cleaner import Cleaner
from ml_research_kills_alpha.support import START_DATE, END_DATE_2023
import argparse


DEFAULT_END_DATE = END_DATE_2023


class ChenZimmermannCleaner(Cleaner):
    """
    Cleans and normalizes Chen & Zimmermann signals (OpenAssetPricing).
    Steps:
      - consider data from START_DATE and end_date (parameter)
      - cross-sectional percent-rank features by month to [-1, 1]
      - replace missing values with 0
      - write full-sample and post-2005 variants
    """
    def __init__(self, end_date: str = DEFAULT_END_DATE):
        super().__init__(dataset_name="chen_zimmermann")
        self.start_date = pd.to_datetime(START_DATE, format="%m/%d/%Y")
        self.end_date = pd.to_datetime(end_date, format="%m/%d/%Y")

    def clean(self) -> Path:

        # load the raw signals
        self.logger.info(f"Loading raw anomaly signals from {self.dataset_name}")
        file_name = self.raw_dir / "chen_zimmermann_signals.csv"
        if not file_name.exists():
            self.logger.error(f"Raw signals file not found: {file_name}, run download file first")
            raise FileNotFoundError(f"Raw signals file not found: {file_name}, run download file first")

        df = pd.read_csv(file_name, low_memory=False)
        
        # time filtering
        df = self.filter_on_date(self.start_date, self.end_date, df)

        # identify feature columns
        meta_cols = {"permno", "date", "yyyymm", "ret", "retx", "siccd", "exchcd", "shrcd", "cusip", "permco", "ticker"}
        numeric_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
        feature_cols = [c for c in numeric_cols if c not in meta_cols]

        # features percent-ranked by month mapped to [-1, 1]
        if not feature_cols:
            self.logger.error("No numeric features identified. Nothing to rank")
            raise ValueError("No numeric features identified. Nothing to rank")
        self.logger.info(f"Percent-ranking {len(feature_cols)} features to [-1,1].")
        df[feature_cols] = (df.groupby("date")[feature_cols].transform(lambda g: g.rank(pct=True) * 2 - 1))

        # missing values replaced with 0
        df[feature_cols] = df[feature_cols].fillna(0.0)

        # save full and post-2005 datasets
        out_full, out_post = self._save_dataframe(df.sort_values(["date", "permno"]).reset_index(drop=True),
                                                  "chen_zimmermann_signals.csv")

        self.logger.info("Cleaned Chen & Zimmermann signals saved")
        self.logger.info(f"Full sample: {out_full}")
        self.logger.info(f"Post-2005: {out_post}")
        return out_full


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Clean Chen & Zimmermann signals dataset.")
    parser.add_argument("--end_date", type=str, default=DEFAULT_END_DATE,
                        help=f"End date in MM/DD/YYYY format (default: {DEFAULT_END_DATE})")
    args = parser.parse_args()

    ChenZimmermannCleaner(end_date=args.end_date).clean()
