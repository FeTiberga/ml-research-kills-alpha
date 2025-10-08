# ml_research_kills_alpha/datasets/download.py
# Base class for datasets downloaders

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

from ml_research_kills_alpha.support import Logger
from ml_research_kills_alpha.config import RAW_DATA_DIR
RAW_DATA_DIR = Path(RAW_DATA_DIR)


class Downloader(ABC):
    """
    Base class for all dataset downloaders.
    Defines common interface and utilities (logging, paths, retries).
    """
    def __init__(self, dataset_name: str):
        self.logger = Logger()
        self.dataset_name = dataset_name
        self.raw_dir = RAW_DATA_DIR
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Initialized downloader for {self.dataset_name} at {self.raw_dir}")

    @abstractmethod
    def download(self) -> pd.DataFrame:
        """Implement the logic to fetch and store raw data files."""
        pass

    def _save_dataframe(self, df: pd.DataFrame, filename: str) -> Path:
        """
        Helper to download a file from `url` and save to `dest` under `dir`.
        """
        dest = self.raw_dir / filename
        df.to_csv(dest, index=False)
        self.logger.info(f"Saved raw file {self.dataset_name} -> {dest}")
        return dest
