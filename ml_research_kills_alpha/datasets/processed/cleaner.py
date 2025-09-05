# ml_research_kills_alpha/datasets/cleaner.py
# Base class for datasets cleaners

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

from ml_research_kills_alpha.support import Logger
from ml_research_kills_alpha.config import RAW_DATA_DIR, PROCESSED_DATA_DIR
from ml_research_kills_alpha.support import CUTOFF_2005
RAW_DIR = Path(RAW_DATA_DIR)
PROCESSED_DIR = Path(PROCESSED_DATA_DIR)


class Cleaner(ABC):
    """
    Base class for all dataset cleaners.
    Defines common interface and utilities (logging, paths).
    """
    def __init__(self, dataset_name: str):
        self.logger = Logger()
        self.dataset_name = dataset_name
        self.processed_dir = PROCESSED_DIR
        self.raw_dir = RAW_DIR
        
        # cleaning comes only after downloading
        if not self.raw_dir.exists():
            self.logger.error(f"Raw directory {self.raw_dir} does not exist.")
            raise FileNotFoundError(f"Raw directory {self.raw_dir} does not exist.")
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Initialized cleaner for {self.dataset_name} at {self.processed_dir}")

    @abstractmethod
    def clean(self) -> Path:
        """Implement processing logic; return the path written."""
        pass
    
    @staticmethod
    def filter_on_date(start_date: pd.Timestamp, end_date: pd.Timestamp, df: pd.DataFrame) -> pd.DataFrame:
        """ Utility to filter a DataFrame on a date range [start_date, end_date]. """

        # make sure date exists and is in right format
        if "date" not in df.columns:
            df["date"] = pd.to_datetime(df["yyyymm"].astype(str), format="%Y%m")
        else:
            df["date"] = pd.to_datetime(df["date"])
            
        filtered = df[(df["date"] >= start_date) & (df["date"] <= end_date)]
        return filtered

    def _save_dataframe(self, df: pd.DataFrame, filename: str) -> tuple[Path, Path]:
        # Save full cleaned dataset
        dest_full = self.processed_dir / filename
        df.to_csv(dest_full, index=False)
        self.logger.info(f"Saved clean file {self.dataset_name} -> {dest_full}")

        # Save post-2005 version
        if "date" in df.columns:
            df_post_2005 = df[df["date"] >= pd.to_datetime(CUTOFF_2005, format="%m/%d/%Y")]
            post_2005_filename = filename.replace(".csv", "_post2005.csv")
            dest_post_2005 = self.processed_dir / post_2005_filename
            df_post_2005.to_csv(dest_post_2005, index=False)
            self.logger.info(f"Saved post-2005 file {self.dataset_name} -> {dest_post_2005}")

        return dest_full, dest_post_2005
