# ml_research_kills_alpha/datasets/data_pipeline.py
from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys

import pandas as pd
import numpy as np

# --- Project config & helpers
from ml_research_kills_alpha.config import RAW_DATA_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR
from ml_research_kills_alpha.support import Logger, START_DATE, END_DATE_2023
from ml_research_kills_alpha.datasets.data.useful_files import (
    CZ_SIGNAL_DOC, FF_INDUSTRY_CLASSIFICATION, GW_PREDICTOR_DATA_2024
)

RAW_DIR = Path(RAW_DATA_DIR)
INTERIM_DIR = Path(INTERIM_DATA_DIR)
PROCESSED_DIR = Path(PROCESSED_DATA_DIR)

log = Logger()

# --- Downloaders (raw)
# CZ via openassetpricing
from ml_research_kills_alpha.datasets.raw.chen_zimmermann import ChenZimmermannDownloader  # :contentReference[oaicite:0]{index=0}
# CRSP via WRDS
# (Your CRSP downloader happens to live in datasets/chen_zimmermann.py in the upload; import path below.)
try:
    from ml_research_kills_alpha.datasets.raw.crsp import CRSPDownloader  # expected location
except Exception:
    # fallback to the uploaded location (same class)
    from ml_research_kills_alpha.datasets.chen_zimmermann import CRSPDownloader  # :contentReference[oaicite:1]{index=1}

# --- Cleaners (interim)
# CZ cleaner
from ml_research_kills_alpha.datasets.processed.chen_zimmermann import ChenZimmermannCleaner  # :contentReference[oaicite:2]{index=2}
# CRSP cleaner (your uploaded CRSP cleaner sits in a file named like chen_zimmermann.py; import it too)
try:
    from ml_research_kills_alpha.datasets.processed.crsp import CRSPCleaner  # expected location
except Exception:
    from ml_research_kills_alpha.datasets.processed.chen_zimmermann import CRSPCleaner  # :contentReference[oaicite:3]{index=3}


# ---------- Utilities

def _ensure_dirs():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def file_exists(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def _read_monthly_wg(baseline_path: Path) -> pd.DataFrame:
    """
    Load Welch & Goyal predictors (already curated monthly CSV in datasets/data).
    Expected columns: date or yyyymm + (dp, ep, bm, ntis, tbl, tms, dfy, svar).
    """
    df = pd.read_csv(baseline_path)
    # normalize date
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["date"] = (df["date"] + pd.offsets.MonthEnd(0)).astype("datetime64[ns]")
    elif "yyyymm" in df.columns:
        s = pd.to_numeric(df["yyyymm"], errors="coerce").astype("Int64")
        df["date"] = pd.to_datetime(s.astype(str), format="%Y%m", errors="coerce")
        df["date"] = (df["date"] + pd.offsets.MonthEnd(0)).astype("datetime64[ns]")
    else:
        raise ValueError("Welch-Goyal file must contain date or yyyymm.")

    keep = ["dp", "ep", "bm", "ntis", "tbl", "tms", "dfy", "svar"]
    for c in keep:
        if c not in df.columns:
            df[c] = pd.NA
    df = df[["date"] + keep].drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df


def _load_ff49_mapper(json_path: Path):
    """
    Returns a function that maps a SIC code (int) -> FF49 short industry key,
    based on your JSON ranges file.
    """
    with open(json_path, "r") as f:
        mapping = json.load(f)  # short_key -> {"name": ..., "ranges": [[a,b], ...]}
    # flatten into list of (short_key, low, high)
    rows = []
    for short, spec in mapping.items():
        for lo, hi in spec["ranges"]:
            rows.append((short, int(lo), int(hi)))
    table = pd.DataFrame(rows, columns=["short", "lo", "hi"])

    def map_sic(sic: float | int | None) -> str | None:
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


def _one_hot_ff49(s: pd.Series, drop_first=True) -> pd.DataFrame:
    d = pd.get_dummies(s.astype("category"), prefix="ff49", dummy_na=False)
    if drop_first and d.shape[1] > 0:
        # Drop one to prevent multicollinearity (48 dummies)
        d = d.iloc[:, 1:]
    return d


# ---------- Steps

def step_download_raw(force_raw: bool):
    """
    Download CZ & CRSP raw data if missing (or always if force_raw=True).
    Outputs:
      - RAW_DIR/chen_zimmermann_signals.csv
      - RAW_DIR/crsp_stock.csv
    """
    _ensure_dirs()
    cz_raw = RAW_DIR / "chen_zimmermann_signals.csv"
    crsp_raw = RAW_DIR / "crsp_stock.csv"

    # CZ
    if force_raw or not file_exists(cz_raw):
        log.info("Downloading Chen & Zimmermann raw signals...")
        ChenZimmermannDownloader().download()  # :contentReference[oaicite:4]{index=4}
    else:
        log.info(f"Found CZ raw -> {cz_raw}, skipping (use --force-raw to refresh).")

    # CRSP
    if force_raw or not file_exists(crsp_raw):
        log.info("Downloading CRSP raw data...")
        CRSPDownloader().download()  # :contentReference[oaicite:5]{index=5}
    else:
        log.info(f"Found CRSP raw -> {crsp_raw}, skipping (use --force-raw to refresh).")


def step_clean_interim(end_date: str = END_DATE_2023):
    """
    Run cleaners to produce interim datasets.
    Outputs (by your cleaner base): full & post-2005 CSVs under INTERIM_DIR.
    """
    _ensure_dirs()
    # CZ -> interim
    log.info("Cleaning CZ signals into interim...")
    ChenZimmermannCleaner(end_date=end_date).clean()  # :contentReference[oaicite:6]{index=6}

    # CRSP -> interim
    log.info("Cleaning CRSP stock into interim...")
    CRSPCleaner(end_date=end_date).clean()  # :contentReference[oaicite:7]{index=7}


def step_merge_processed(
    ff_json_path: Path,
    wg_path: Path,
    out_name: str = "master_panel.csv",
    drop_one_ff_dummy: bool = True,
):
    """
    Merge:
      - interim CZ features (by permno, date),
      - interim CRSP (by permno, date),
      - FF49 industry one-hots from SICCD,
      - WG predictors (by date).
    Write processed/master_panel.csv (+ _post2005).
    """
    _ensure_dirs()

    # ----- Load interim inputs
    cz_path = INTERIM_DIR / "chen_zimmermann_signals.csv"
    crsp_path = INTERIM_DIR / "crsp_stock.csv"

    if not file_exists(cz_path):
        raise FileNotFoundError(f"Missing interim CZ file: {cz_path}. Run cleaning first.")
    if not file_exists(crsp_path):
        raise FileNotFoundError(f"Missing interim CRSP file: {crsp_path}. Run cleaning first.")

    cz = pd.read_csv(cz_path, low_memory=False)
    cr = pd.read_csv(crsp_path, low_memory=False)

    for df in (cz, cr):
        if "date" not in df.columns:
            df["date"] = pd.to_datetime(df["yyyymm"].astype(str), format="%Y%m")
        else:
            df["date"] = pd.to_datetime(df["date"])

    # ----- Merge CZ + CRSP on (permno, date)
    # inner join keeps common universe by date/permno
    panel = pd.merge(
        cz, cr,
        on=["permno", "date"],
        how="inner",
        suffixes=("", "_crsp")
    )

    # ----- FF49 one-hot from SIC
    # Prefer CRSP's siccd if present; otherwise use CZ's
    sic = panel["siccd_crsp"] if "siccd_crsp" in panel.columns else panel.get("siccd", pd.Series(index=panel.index))
    map_fn = _load_ff49_mapper(Path(ff_json_path))  # :contentReference[oaicite:8]{index=8}
    panel["ff49_short"] = sic.map(map_fn)

    ff_dummies = _one_hot_ff49(panel["ff49_short"], drop_first=drop_one_ff_dummy)
    panel = pd.concat([panel, ff_dummies], axis=1)

    # ----- Add monthly WG predictors by date
    wg = _read_monthly_wg(Path(wg_path))  # uses curated CSV in datasets/data
    panel = pd.merge(panel, wg, on="date", how="left")

    # ----- Final ordering & write
    panel = panel.sort_values(["date", "permno"]).reset_index(drop=True)
    out_full = PROCESSED_DIR / out_name
    panel.to_csv(out_full, index=False)
    log.info(f"Wrote processed panel -> {out_full}")

    # Post-2005 convenience slice (mirrors your Cleaner convention)  :contentReference[oaicite:9]{index=9}
    post2005 = panel[panel["date"] >= pd.Timestamp("2005-01-01")]
    out_post = PROCESSED_DIR / out_name.replace(".csv", "_post2005.csv")
    post2005.to_csv(out_post, index=False)
    log.info(f"Wrote processed post-2005 -> {out_post}")

    return out_full, out_post


# ---------- CLI

def main():
    parser = argparse.ArgumentParser(description="End-to-end data pipeline: raw -> interim -> processed")
    parser.add_argument("--force-raw", action="store_true", help="Redownload raw data even if files exist.")
    parser.add_argument("--end-date", type=str, default=END_DATE_2023,
                        help=f"End date for cleaners (MM/DD/YYYY). Default: {END_DATE_2023}")
    parser.add_argument("--wg-path", type=str, default=GW_PREDICTOR_DATA_2024,
                        help="Path to curated Welch-Goyal monthly CSV (datasets/data).")
    parser.add_argument("--ff49-json", type=str, default=FF_INDUSTRY_CLASSIFICATION,
                        help="Path to Fama-French 49 mapping JSON (datasets/data).")
    parser.add_argument("--out-name", type=str, default="master_panel.csv",
                        help="Output processed filename.")

    args = parser.parse_args()

    log.info("=== STEP 1: RAW DOWNLOAD ===")
    step_download_raw(force_raw=args.force_raw)

    log.info("=== STEP 2: CLEAN → INTERIM ===")
    step_clean_interim(end_date=args.end_date)

    log.info("=== STEP 3: MERGE → PROCESSED ===")
    step_merge_processed(
        ff_json_path=Path(args.ff49_json),
        wg_path=Path(args.wg_path),
        out_name=args.out_name,
        drop_one_ff_dummy=True,
    )

if __name__ == "__main__":
    _ensure_dirs()
    sys.exit(main())
