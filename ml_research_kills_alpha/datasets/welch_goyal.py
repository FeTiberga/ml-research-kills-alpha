# ml_research_kills_alpha/datasets/macros_welch_splice.py
from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict
import os
import argparse
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fredapi import Fred

from .download import Downloader
from ..config import RAW_DATA_DIR

GW_URL = "https://www.ivo-welch.info/papers/Goyal_Welch_Data/goyal-welch-a.csv"

class WelchGoyalSpliceDownloader(Downloader):
    """
    Load Welch–Goyal monthly predictors (dp, ep, bm, ntis, tbl, tms, dfy, svar) up to 2020 from CSV,
    then extend to END with FRED:
      - Always: tbl (TB3MS), tms (T10Y3M or GS10 - TB3MS), dfy (BAA - AAA), svar (monthly var of daily SP500)
      - Optionally: dp, ep, bm, ntis via user-supplied FRED series (EXTRA mapping)
      - Otherwise: forward-fill dp/ep/bm/ntis from last WG value (if --ffill-missing)
    """
    KEEP = ["dp", "ep", "bm", "ntis", "tbl", "tms", "dfy", "svar"]

    def __init__(
        self,
        end: str,
        start: str = "1957-03-01",
        url: str = GW_URL,
        extra_map: Optional[Dict[str, str]] = None,
        ffill_missing: bool = True,
        raw_dir: Path | str = RAW_DATA_DIR,
    ):
        super().__init__("welch_goyal_splice", version=None, raw_dir=raw_dir)
        self.start = pd.to_datetime(start)
        self.end = pd.to_datetime(end)
        self.url = url
        self.extra_map = extra_map or {}
        self.ffill_missing = ffill_missing

        load_dotenv()
        api_key = os.getenv("FRED_API_KEY")
        if not api_key:
            self.logger.error("FRED_API_KEY missing in environment (.env).")
            raise RuntimeError("Missing FRED_API_KEY")
        self.fred = Fred(api_key=api_key)

    # ---------- FRED helpers ----------
    def _fred(self, code: str) -> pd.Series:
        s = self.fred.get_series(code, observation_start=self.start, observation_end=self.end)
        s.index = pd.to_datetime(s.index)
        return s

    def _fred_monthly(self, code: str, how: str = "last") -> pd.Series:
        s = self._fred(code)
        if how == "mean":
            s = s.resample("M").mean()
        else:
            s = s.resample("M").last()
        s.name = code
        return s

    # ---------- build extension after WG last date ----------
    def _build_extension(self, idx: pd.DatetimeIndex) -> pd.DataFrame:
        # T-bill (decimal)
        tbl = (self._fred_monthly("TB3MS", "last") / 100.0).reindex(idx)

        # Term spread
        try:
            tms = (self._fred_monthly("T10Y3M", "last") / 100.0).reindex(idx)
        except Exception:
            self.logger.warn("T10Y3M unavailable; using GS10 - TB3MS.")
            tms = (self._fred_monthly("GS10", "last") / 100.0 - tbl).reindex(idx)

        # Default spread
        baa = (self._fred_monthly("BAA", "last") / 100.0).reindex(idx)
        aaa = (self._fred_monthly("AAA", "last") / 100.0).reindex(idx)
        dfy = (baa - aaa)
        dfy.name = "dfy"

        # SVAR: monthly variance of daily SP500 returns
        spx = self._fred("SP500").asfreq("B").ffill()
        spx_ret = spx.pct_change()
        svar = spx_ret.groupby(pd.Grouper(freq="M")).var().reindex(idx)
        svar.name = "svar"

        ext = pd.concat(
            [tbl.rename("tbl"), tms.rename("tms"), dfy, svar],
            axis=1
        )

        # Optional constructions: dp/ep/bm/ntis via proxies
        # dp = log(rolling_12m(div) / price), ep = log(rolling_12m(earn) / price)
        if {"div", "price"} <= set(self.extra_map):
            div = self._fred_monthly(self.extra_map["div"], "mean").reindex(idx)
            price = self._fred_monthly(self.extra_map["price"], "last").reindex(idx)
            dp = np.log((div.rolling(12, min_periods=12).sum() / price).replace({0: np.nan}))
            ext["dp"] = dp
        if {"earn", "price"} <= set(self.extra_map):
            earn = self._fred_monthly(self.extra_map["earn"], "mean").reindex(idx)
            price = ext.get("price") or self._fred_monthly(self.extra_map["price"], "last").reindex(idx)
            ep = np.log((earn.rolling(12, min_periods=12).sum() / price).replace({0: np.nan}))
            ext["ep"] = ep
        if "bm" in self.extra_map:
            ext["bm"] = self._fred_monthly(self.extra_map["bm"], "last").reindex(idx)
        if "ntis" in self.extra_map:
            ext["ntis"] = self._fred_monthly(self.extra_map["ntis"], "last").reindex(idx)

        return ext

    def download(self) -> Path:
        # 1) Load WG CSV (monthly)
        df = pd.read_csv(self.url)
        if "date" in df.columns:
            dt = pd.to_datetime(df["date"])
        elif "yyyymm" in df.columns:
            dt = pd.to_datetime(df["yyyymm"].astype(str), format="%Y%m")
        else:
            raise ValueError("Welch–Goyal CSV needs a 'date' or 'yyyymm' column.")

        df.index = pd.PeriodIndex(dt, freq="M").to_timestamp("M")
        df.index.name = "date"
        df = df.sort_index()

        # keep the 8 predictors
        cols = [c for c in self.KEEP if c in df.columns]
        base = df[cols].copy()

        # 2) If END <= last WG date: just truncate and save
        if self.end <= base.index.max():
            out = base.loc[(base.index >= self.start) & (base.index <= self.end)].reset_index()
            return self._save_file(out, "goyal_welch_monthly.csv")

        # 3) Else splice: base through last WG, then FRED extension to END
        last_wg = base.index.max()
        base_trunc = base.loc[base.index <= last_wg].copy()

        ext_idx = pd.date_range(last_wg + pd.offsets.MonthEnd(1), self.end, freq="M")
        self.logger.info(f"Extending WG beyond {last_wg.date()} to {self.end.date()} for {len(ext_idx)} months…")

        ext = self._build_extension(ext_idx)

        # Fill dp/ep/bm/ntis if missing and ffill requested
        for k in ["dp", "ep", "bm", "ntis"]:
            if k not in ext.columns and self.ffill_missing:
                ext[k] = base_trunc[k].iloc[-1] if k in base_trunc.columns else np.nan
                self.logger.warn(f"{k} not provided by EXTRA; forward-filled from last WG value.")

        # Ensure column order and concat
        ext = ext.reindex(columns=self.KEEP)
        out = pd.concat([base_trunc, ext], axis=0).loc[self.start:self.end]
        out = out.reset_index()
        return self._save_file(out, "goyal_welch_monthly.csv")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Welch–Goyal macros spliced with FRED to END date")
    p.add_argument("--start", type=str, default="1957-03-01")
    p.add_argument("--end", type=str, required=True, help="YYYY-MM-DD (e.g. 2021-12-31)")
    p.add_argument("--url", type=str, default=GW_URL)
    p.add_argument("--ffill-missing", action="store_true")
    p.add_argument(
        "--extra",
        type=str,
        default="",
        help="Optional k=v pairs for dp/ep/bm/ntis (e.g. div=SP500DIV,earn=SP500EPS,price=SP500,ntis=NCBEILQ027S)"
    )
    args = p.parse_args()

    extra_map = {}
    if args.extra:
        for kv in args.extra.split(","):
            if "=" in kv:
                k, v = kv.split("=", 1)
                extra_map[k.strip()] = v.strip()

    dl = WelchGoyalSpliceDownloader(
        start=args.start, end=args.end, url=args.url,
        extra_map=extra_map, ffill_missing=args.ffill_missing
    )
    dl.download()
