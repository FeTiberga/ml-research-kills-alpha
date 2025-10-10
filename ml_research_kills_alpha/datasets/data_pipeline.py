# ml_research_kills_alpha/datasets/data_pipeline.py
from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys

import pandas as pd

from ml_research_kills_alpha.config import RAW_DATA_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR
from ml_research_kills_alpha.support import Logger, START_DATE, END_DATE_2023
from ml_research_kills_alpha.datasets.useful_files import FF_INDUSTRY_CLASSIFICATION, GW_PREDICTOR_DATA_2024
from ml_research_kills_alpha.datasets.raw import CRSPDownloader, ChenZimmermannDownloader
from ml_research_kills_alpha.datasets.processed import CRSPCleaner, ChenZimmermannCleaner


RAW_DIR = Path(RAW_DATA_DIR)
INTERIM_DIR = Path(INTERIM_DATA_DIR)
PROCESSED_DIR = Path(PROCESSED_DATA_DIR)

log = Logger()


def file_exists(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def _ensure_date_col(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure the dataframe has a 'date' column of datetime type.
    If 'date' is missing but 'yyyymm' exists, convert 'yyyymm' to 'date'.
    If both are missing, raise an error.
    Date is always the first day of the month
    """
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    elif "yyyymm" in df.columns:
        df["date"] = pd.to_datetime(df["yyyymm"].astype(str), format="%Y%m")
    else:
        raise ValueError("DataFrame must have either 'date' or 'yyyymm' column.")
    df["date"] = pd.to_datetime(df["date"]).dt.to_period("M").dt.to_timestamp()
    return df


def _ensure_permno_int(s: pd.Series) -> pd.Series:
    # force permno to be Int64 (nullable int)
    int_s = pd.to_numeric(s, errors="coerce").astype("Int64")
    return int_s


def _read_monthly_wg() -> pd.DataFrame:
    """
    Load Welch & Goyal predictors.
    Uses only the 8 main predictors:
      - dividend-price ratio (D12)
      - earnings-price ratio (E12)
      - book-to-market ratio (b/m)
      - net equity expansion (ntis)
      - treasury-bill rate (tbl)
      - term-spread (tms)
      - default spread (dfy)
      - stock variance (svar)
    """
    df = pd.read_csv(GW_PREDICTOR_DATA_2024)
    df["date"] = pd.to_datetime(df["yyyymm"].astype(str), format="%Y%m")

    # safety: create columns if missing
    for c in ["D12","E12","b/m","tbl","lty","BAA","AAA","ntis","svar"]:
        if c not in df.columns:
            df[c] = pd.NA

    df["tms"] = pd.to_numeric(df["lty"], errors="coerce") - pd.to_numeric(df["tbl"], errors="coerce")
    df["dfy"] = pd.to_numeric(df["BAA"], errors="coerce") - pd.to_numeric(df["AAA"], errors="coerce")

    result = pd.DataFrame({
        "date": df["date"],
        "dp":   pd.to_numeric(df["D12"], errors="coerce"),
        "ep":   pd.to_numeric(df["E12"], errors="coerce"),
        "bm":   pd.to_numeric(df["b/m"], errors="coerce"),
        "ntis": pd.to_numeric(df["ntis"], errors="coerce"),
        "tbl":  pd.to_numeric(df["tbl"], errors="coerce"),
        "tms":  df["tms"],
        "dfy":  df["dfy"],
        "svar": pd.to_numeric(df["svar"], errors="coerce"),
    }).drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)

    return _ensure_date_col(result)


def _load_ff49_mapper():
    """
    Returns a function that maps a SIC code (int) -> FF49 short industry key,
    based on your JSON ranges file.
    """
    with open(FF_INDUSTRY_CLASSIFICATION, "r") as f:
        mapping = json.load(f)
        
    # format is industry_key -> {"full name": ..., "ranges": [[a,b], ...]}
    # flatten into list of (short_key, low, high)
    rows = []
    for short, spec in mapping.items():
        for lo, hi in spec["ranges"]:
            rows.append((short, int(lo), int(hi)))
    table = pd.DataFrame(rows, columns=["short", "lo", "hi"])

    def map_sic(sic: float | int | None) -> str | None:
        # return the short industry key for this SIC code, or None if no match
        if sic is None or pd.isna(sic):
            return None
        try:
            s = int(sic)
        except Exception:
            return None
        hit = table[(table["lo"] <= s) & (s <= table["hi"])]
        if hit.empty:
            return None
        # if multiple hits, take the first
        return hit.iloc[0]["short"]

    return map_sic


def _one_hot_ff49(s: pd.Series) -> pd.DataFrame:
    d = pd.get_dummies(s.astype("category"), prefix="ff49", dummy_na=False)
    # dropping one to prevent multicollinearity (48 dummies)
    d = d.iloc[:, 1:]
    return d


def step_download_raw(force_raw: bool) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """
    Download CZ & CRSP raw data if missing (or always if force_raw=True).
    Args:
        force_raw: if True, redownload even if files exist
    """
    cz_raw = RAW_DIR / "chen_zimmermann_signals.csv"
    crsp_raw = RAW_DIR / "crsp_stock.csv"
    
    cz_df = None
    crsp_df = None

    # CZ
    if force_raw or not file_exists(cz_raw):
        cz_df = ChenZimmermannDownloader().download()
    else:
        log.info(f"Found CZ raw -> {cz_raw}, skipping (use --force-raw to refresh).")

    # CRSP
    if force_raw or not file_exists(crsp_raw):
        crsp_df = CRSPDownloader().download()
    else:
        log.info(f"Found CRSP raw -> {crsp_raw}, skipping (use --force-raw to refresh).")
        
    return cz_df, crsp_df
    

def step_clean_interim(force_clean: bool, end_date: str,
                       cz_df: pd.DataFrame | None = None,
                       crsp_df: pd.DataFrame | None = None) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """
    Clean CZ & CRSP raw data into interim files if missing (or always if force_clean=True).
    Args:
        force_clean: if True, re-clean even if files exist
        end_date: end date for cleaning (MM/DD/YYYY)
    """
    cz_interim = INTERIM_DIR / "chen_zimmermann_signals.csv"
    crsp_interim = INTERIM_DIR / "crsp_stock.csv"

    # CZ
    if force_clean or not file_exists(cz_interim):
        cz_df = ChenZimmermannCleaner(end_date=end_date, dataset=cz_df).clean()
    else:
        log.info(f"Found cleaned CZ -> {cz_interim}, skipping (use --force-clean to refresh).")

    # CRSP
    if force_clean or not file_exists(crsp_interim):
        crsp_df = CRSPCleaner(end_date=end_date, dataset=crsp_df).clean()
    else:
        log.info(f"Found cleaned CRSP -> {crsp_interim}, skipping (use --force-clean to refresh).")
        
    return cz_df, crsp_df


def step_merge_processed(cz_df: pd.DataFrame | None, crsp_df: pd.DataFrame | None) -> str:
    """
    Merge:
      - interim CZ features (by permno, date),
      - interim CRSP (by permno, date),
      - FF49 industry one-hots from SICCD,
      - WG predictors (by date).
    Write processed/master_panel.csv.        
    """

    # load interim files
    cz_path = INTERIM_DIR / "chen_zimmermann_signals.csv"
    crsp_path = INTERIM_DIR / "crsp_stock.csv"
    if not file_exists(cz_path):
        raise FileNotFoundError(f"Missing interim CZ file: {cz_path}. Run cleaning first.")
    if not file_exists(crsp_path):
        raise FileNotFoundError(f"Missing interim CRSP file: {crsp_path}. Run cleaning first.")

    cz = cz_df if cz_df is not None else pd.read_csv(cz_path)
    crsp = crsp_df if crsp_df is not None else pd.read_csv(crsp_path)

    # ensure date and permno columns are in the same format
    cz   = _ensure_date_col(cz)
    crsp = _ensure_date_col(crsp)
    cz["permno"]   = _ensure_permno_int(cz["permno"])
    crsp["permno"] = _ensure_permno_int(crsp["permno"])
    cz.dropna(subset=["permno", "date"], inplace=True)
    crsp.dropna(subset=["permno", "date"], inplace=True)
    log.info(f"CZ data: {cz.shape[0]} rows, {cz.shape[1]} columns")
    log.info(f"CRSP data: {crsp.shape[0]} rows, {crsp.shape[1]} columns")
    
    # merge CZ + CRSP on (permno, date)
    panel = pd.merge(
        crsp, cz,
        on=["permno", "date"],
        how="left",
        suffixes=("", "_cz")
    )
    log.info(f"Merged CZ + CRSP -> {panel.shape[0]} rows, {panel.shape[1]} columns")

    # add one-hot FF49 industry dummies from SICCD
    sic = panel["siccd"]
    map_fn = _load_ff49_mapper()
    panel["ff49_short"] = sic.map(map_fn)
    ff_dummies = _one_hot_ff49(panel["ff49_short"])
    panel = pd.concat([panel, ff_dummies], axis=1)

    # add WG predictors by date
    wg = _read_monthly_wg()
    panel = pd.merge(panel, wg, on="date", how="left")
    
    # make sure data is past START_DATE
    panel = panel[panel["date"] >= pd.to_datetime(START_DATE)]
    
    # convert data to float32 and int32 where possible to save space and processing time
    for c in panel.select_dtypes(include=["float64"]).columns:
        panel[c] = pd.to_numeric(panel[c], errors="coerce").astype("float32")
    for c in panel.select_dtypes(include=["int64"]).columns:
        panel[c] = pd.to_numeric(panel[c], errors="coerce").astype("int32")

    # remove rows with no permno, ret, or retx
    meta_missing_count = {col: panel[col].isna().sum() for col in ["permno", "abret", "ret", "retx"]}
    for col, count in meta_missing_count.items():
        log.info(f"Column {col} has {count} missing values, dropping those rows")
    panel = panel.dropna(subset=["permno", "abret", "ret", "retx"])

    # final ordering
    out_name = "master_panel.parquet"
    panel = panel.sort_values(["date", "permno"]).reset_index(drop=True)
    out_full = PROCESSED_DIR / out_name
    log.info(f"Wrote processed panel -> {out_full}")
    
    # save head of panel for quick inspection
    head_name = "master_panel_head.csv"
    panel.to_parquet(out_full, index=False)
    panel.head(100).to_csv(PROCESSED_DIR / head_name, index=False)
    log.info(f"Wrote processed panel head (100 rows) -> {PROCESSED_DIR / head_name}")

    return out_full


def main():
    parser = argparse.ArgumentParser(description="End-to-end data pipeline: raw -> interim -> processed")
    parser.add_argument("--force-raw", action="store_true", default=False,
                        help="Redownload raw data even if files exist. Default: False")
    parser.add_argument("--force-clean", action="store_true", default=False,
                        help="Re-clean interim data even if files exist. Default: False")
    parser.add_argument("--end-date", type=str, default=END_DATE_2023,
                        help=f"End date for cleaners (MM/DD/YYYY). Default: {END_DATE_2023}")

    args = parser.parse_args()

    log.info("=== STEP 1: RAW DOWNLOAD ===")
    cz_df, crsp_df = step_download_raw(force_raw=args.force_raw)

    log.info("=== STEP 2: CLEAN DATA ===")
    cz_df, crsp_df = step_clean_interim(force_clean=args.force_clean, end_date=args.end_date,
                                        cz_df=cz_df, crsp_df=crsp_df)

    log.info("=== STEP 3: MERGE DATA ===")
    step_merge_processed(cz_df=cz_df, crsp_df=crsp_df)


if __name__ == "__main__":
    sys.exit(main())
