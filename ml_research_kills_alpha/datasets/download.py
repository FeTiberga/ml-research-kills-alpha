# ml_research_kills_alpha/datasets/download.py
# Base class for dataset downloaders in the ml_research_kills_alpha project.

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

from support import Logger
from config import RAW_DATA_DIR


class Downloader(ABC):
    """
    Base class for all dataset downloaders.
    Defines common interface and utilities (logging, paths, retries).
    """
    def __init__(self,
                 dataset_name: str,
                 version: int | None = None,
                 raw_dir: str = RAW_DATA_DIR):
        self.logger = Logger()
        self.dataset_name = dataset_name
        self.version = version
        self.raw_dir = Path(raw_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Initialized downloader for {self.dataset_name} (version={self.version}) at {self.raw_dir}")

    @abstractmethod
    def download(self) -> Path:
        """Implement the logic to fetch and store raw data files."""
        pass

    def _save_dataframe(self, df: pd.DataFrame, filename: str) -> Path:
        """
        Helper to download a file from `url` and save to `dest` under `dir`.
        """
        dest = self.raw_dir / filename
        df.to_csv(dest, index=False)
        self.logger.info(f"Saved {self.dataset_name} (version={self.version}) -> {dest}")
        return dest
