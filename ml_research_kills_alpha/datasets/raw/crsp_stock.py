# ml_research_kills_alpha/datasets/chen_zimmermann.py
# Downloader for CRSP Stock data using the WRDS API.

from pathlib import Path

import pandas as pd

from ml_research_kills_alpha.datasets.raw.download import Downloader
from ml_research_kills_alpha.support import START_DATE_CRSP
from ml_research_kills_alpha.support.wrds_connection import connect_wrds


class CRSPDownloader(Downloader):
    """
    Downloads anomaly signals from the Chen & Zimmermann dataset via the
    openassetpricing package.
    """
    def __init__(self):
        super().__init__(dataset_name="crsp_stock")
        self.start_date = START_DATE_CRSP

    def download(self) -> Path:

        conn = connect_wrds()
        self.logger.info("Downloading CRSP Stock data...")
        try:
            crsp = conn.raw_sql(f"""
            SELECT
                m.permno, m.date,
                m.ret, m.retx, m.prc, m.shrout, m.altprc, m.vol, m.bid, m.ask,
                n.shrcd, n.exchcd
            FROM crsp.msf AS m
            JOIN crsp.msenames AS n
            ON m.permno = n.permno
            AND m.date >= n.namedt
            AND (n.nameendt IS NULL OR m.date <= n.nameendt)
            WHERE m.date >= '{self.start_date}'
            AND n.shrcd IN (10, 11)
            """)
            self.logger.info("CRSP Stock data downloaded successfully.")
        except Exception as e:
            self.logger.error(f"Failed to download CRSP Stock data: {e}")
            raise

        df: pd.DataFrame = pd.DataFrame(crsp)
        df["date"] = pd.to_datetime(df["date"])
        first_date = df["date"].min()
        last_date = df["date"].max()
        self.logger.info(
            f"Downloaded {len(df)} rows of CRSP Stock data "
            f"from {first_date} to {last_date}."
        )

        # handle possible None end_date
        first_date_short = first_date.strftime("%Y%m%d")
        last_date_short = last_date.strftime("%Y%m%d")
        filename = f"crsp_stock_{first_date_short}_{last_date_short}.csv"
        out_path = self._save_dataframe(df, filename)
        return out_path


if __name__ == "__main__":
    downloader = CRSPDownloader()
    downloader.download()
