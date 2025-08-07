# ml_research_kills_alpha/datasets/chen_zimmermann.py
# Downloader for CRSP Stock data using the WRDS API.

from pathlib import Path
import argparse
from dotenv import load_dotenv
import os

import pandas as pd
import wrds

from datasets.download import Downloader
from config import RAW_DATA_DIR
from support import START_DATE_CRSP


class CRSPDownloader(Downloader):
    """
    Downloads anomaly signals from the Chen & Zimmermann dataset via the
    openassetpricing package.
    """
    def __init__(self,
                 end_date: str | None = None):
        super().__init__(dataset_name="crsp_stock", version=None, raw_dir=RAW_DATA_DIR)
        self.start_date = START_DATE_CRSP
        self.end_date = self._convert_end_date(end_date) if end_date else None
        
    @staticmethod
    def _convert_end_date(end_date: str) -> str:
        """
        Convert str 'YYYYMM' to str 'MM/DD/YYYY'
        """
        return pd.to_datetime(end_date, format='%Y%m').strftime('%m/%d/%Y')

    def download(self) -> Path:
        
        self.logger.info("Connecting to WRDS...")
        load_dotenv()
        try:
            user = os.getenv("WRDS_USERNAME")
            password = os.getenv("WRDS_PASSWORD")
            
            conn = wrds.Connection(
                username=user,
                password=password
            )
            self.logger.info("Connected to WRDS successfully.")
        except Exception as e:
            self.logger.error(f"Failed to connect to WRDS: {e}")
            raise
        
        self.logger.info("Downloading CRSP Stock data...")
        try:
            # handle possible None end_date
            if self.end_date and not str(self.end_date).startswith('-'):
                date_condition = f"date between '{self.start_date}' and '{self.end_date}'"
            else:
                date_condition = f"date >= '{self.start_date}'"

            crsp = conn.raw_sql(f"""
                select permno, date, ret, retx, prc, shrout, altprc, vol, bid, ask
                from crsp.msf 
                where {date_condition}
                and shrcd in (10, 11)
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
        end_date_str = str(self.end_date) if self.end_date and not str(self.end_date).startswith('-') else "latest"
        filename = f"crsp_stock_{self.start_date}_{end_date_str}.csv"
        out_path = self._save_dataframe(df, filename)
        return out_path


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--end-date",
        dest="end_date",
        type=str,
        default=None,
        help="End date for CRSP data (YYYYMM). Omit for latest."
    )
    args = p.parse_args()

    downloader = CRSPDownloader(
        end_date=args.end_date
    )
    downloader.download()
