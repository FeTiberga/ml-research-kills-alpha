# ml_research_kills_alpha/datasets/processed/chen_zimmermann.py
# Cleaner for the Chen & Zimmermann dataset

from pathlib import Path

import pandas as pd
import numpy as np

from ml_research_kills_alpha.datasets.processed.cleaner import Cleaner
from ml_research_kills_alpha.support import START_DATE, END_DATE_2023
import argparse


DEFAULT_END_DATE = END_DATE_2023


class CRSPCleaner(Cleaner):
    """
    Cleans and normalizes CRSP Stock data.
    Steps:
      - consider data from START_DATE and end_date (parameter)
      - write full-sample and post-2005 variants
    """
    def __init__(self, end_date: str = DEFAULT_END_DATE):
        super().__init__(dataset_name="crsp_stock")
        self.start_date = pd.to_datetime(START_DATE, format="%m/%d/%Y")
        self.end_date = pd.to_datetime(end_date, format="%m/%d/%Y")

    def clean(self) -> Path:

        # load the raw signals
        self.logger.info(f"Loading raw CRSP stock data from {self.dataset_name}")
        file_name = self.raw_dir / f"{self.dataset_name}.csv"
        if not file_name.exists():
            self.logger.error(f"Raw stock data file not found: {file_name}, run download file first")
            raise FileNotFoundError(f"Raw stock data file not found: {file_name}, run download file first")

        df = pd.read_csv(file_name, low_memory=False)
        
        # time filtering
        df = self.filter_on_date(self.start_date, self.end_date, df)

        # save full and post-2005 datasets
        out_full, out_post = self._save_dataframe(df.sort_values(["date", "permno"]).reset_index(drop=True), f"{self.dataset_name}.csv")

        self.logger.info("Cleaned Chen & Zimmermann signals saved")
        self.logger.info(f"Full sample: {out_full}")
        self.logger.info(f"Post-2005: {out_post}")
        return out_full


def main(end_date: str = DEFAULT_END_DATE):
    cleaner = CRSPCleaner(end_date=end_date)
    cleaner.clean()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean CRSP stock dataset.")
    parser.add_argument("--end_date", type=str, default=DEFAULT_END_DATE,
                        help=f"End date in MM/DD/YYYY format (default: {DEFAULT_END_DATE})")
    args = parser.parse_args()
    main(end_date=args.end_date)
