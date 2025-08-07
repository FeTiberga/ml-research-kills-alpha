# ml_research_kills_alpha/datasets/chen_zimmermann.py
# Downloader for the Chen & Zimmermann dataset using openassetpricing

from pathlib import Path
import argparse

import openassetpricing as oap
import pandas as pd

from datasets.download import Downloader
from config import RAW_DATA_DIR


class ChenZimmermannDownloader(Downloader):
    """
    Downloads anomaly signals from the Chen & Zimmermann dataset via the
    openassetpricing package.
    """
    def __init__(self,
                 version: int | None = None):
        super().__init__(dataset_name="chen_zimmermann", version=version, raw_dir=RAW_DATA_DIR)

    def download(self) -> Path:
        
        # initialise the API client (None for latest release)
        client = oap.OpenAP(self.version) if self.version else oap.OpenAP()

        # fetch into a pandas DataFrame
        df: pd.DataFrame = client.dl_all_signals("pandas")
        df["date"] = pd.to_datetime(df["date"])
        first_date = df["date"].min()
        last_date = df["date"].max()
        self.logger.info(
            f"Downloaded {len(df)} rows of Chen & Zimmermann signals "
            f"from {first_date} to {last_date}."
        )

        # Build a filename using the release tag
        filename = f"chen_zimmermann_signals_{self.version}.csv"
        # Use the base-class helper to persist it
        out_path = self._save_dataframe(df, filename)
        return out_path


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--version",
        type=str,
        default=None,
        help="OSAP release (e.g. 202408). Omit for latest."
    )
    args = p.parse_args()

    downloader = ChenZimmermannDownloader(
        version=args.version,
    )
    downloader.download()
