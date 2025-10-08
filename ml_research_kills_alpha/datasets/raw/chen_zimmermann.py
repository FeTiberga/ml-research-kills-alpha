# ml_research_kills_alpha/datasets/raw/chen_zimmermann.py
# Downloader for the Chen & Zimmermann dataset using openassetpricing

from pathlib import Path

import openassetpricing as oap
import pandas as pd

from ml_research_kills_alpha.datasets.raw.download import Downloader
from ml_research_kills_alpha.support.wrds_connection import prepare_wrds_noninteractive


class ChenZimmermannDownloader(Downloader):
    """
    Downloads anomaly signals from the Chen & Zimmermann dataset via the
    openassetpricing package.
    """
    def __init__(self):
        super().__init__(dataset_name="chen_zimmermann")

    def download(self) -> Path:
        
        # initialise the API client (None for latest release)
        prepare_wrds_noninteractive()
        client = oap.OpenAP()

        # fetch into a pandas DataFrame
        df: pd.DataFrame = client.dl_all_signals("pandas")
        df["date"] = pd.to_datetime(df["yyyymm"].astype(str), format="%Y%m")
        first_date = df["date"].min()
        last_date = df["date"].max()
        self.logger.info(
            f"Downloaded {len(df)} rows of Chen & Zimmermann signals "
            f"from {first_date} to {last_date}."
        )

        # Build a filename using the release tag
        filename = f"chen_zimmermann_signals.csv"
        # Use the base-class helper to persist it
        out_path = self._save_dataframe(df, filename)
        return df


def main():
    downloader = ChenZimmermannDownloader()
    downloader.download()

if __name__ == "__main__":
    main()
