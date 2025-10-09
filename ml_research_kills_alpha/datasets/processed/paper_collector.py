from __future__ import annotations
import os
import re
import json
import time
from datetime import datetime
from typing import Any

from tqdm import tqdm
import requests
import requests_cache
from bs4 import BeautifulSoup
from dateutil import parser as dateparser

from ml_research_kills_alpha.support import Logger, START_DATE
from ml_research_kills_alpha.config import PROCESSED_DATA_DIR


TOP_JOURNALS = [
    "Journal of Finance",
    "Journal of Financial Economics",
    "Review of Financial Studies",  # Crossref often stores this as "The Review of Financial Studies"
    "Journal of Financial and Quantitative Analysis",
    "Management Science",
]
SUB_TIER_JOURNALS = [
    "Journal of Banking and Finance",
    "Journal of Empirical Finance",
    "Journal of Financial Markets",
    "Financial Analysts Journal",
    "Journal of Portfolio Management",
]


class PaperCollector:
    """
    Collects machine-learning-related finance papers (START_DATE-present) from specified journals.
    Keeps the same output format: CSV with boolean `contains_*` columns and a JSONL mirror.
    """

    def __init__(self, include_subtier: bool = False, max_records: int | None = None):
        """
        Initialize the collector.

        Args:
            include_subtier (bool): Whether to include a broader set of finance journals beyond the top list.
            max_records (int | None): If set, limit the number of records to fetch (for dry-run testing).
        """
        # Copy the list to avoid in-place modification of module-level constants
        self.journals: list[str] = list(TOP_JOURNALS)
        if include_subtier:
            self.journals += SUB_TIER_JOURNALS

        self.max_records = max_records
        self.logger = Logger()

        # Transparent HTTP caching (Crossref & S2)
        requests_cache.install_cache("mlfinance_cache", expire_after=86400)  # 24h

        # Synonym-aware regex patterns (case-insensitive)
        self.model_patterns: dict[str, re.Pattern] = {
            "ols": re.compile(r"\b(?:O\.?L\.?S\.?|ordinary\s+least\s+squares)\b", re.IGNORECASE),
            "elastic_net": re.compile(r"\b(?:elastic-?net|elasticnet|elastic\s+net)\b", re.IGNORECASE),
            "lasso": re.compile(r"\b(?:lasso|l1-?\s*regulari[sz]ed\s+regression)\b", re.IGNORECASE),
            "ridge": re.compile(r"\b(?:ridge|l2-?\s*regulari[sz]ed\s+regression)\b", re.IGNORECASE),
            "ffnn": re.compile(
                r"\b(?:FFNN|feed\s*forward\s+neural\s+network(?:s)?|feedforward\s+neural\s+network(?:s)?|fully\s+connected\s+network(?:s)?)\b",
                re.IGNORECASE,
            ),
            "lstm": re.compile(r"\b(?:LSTM|long\s+short-?\s*term\s+memory)\b", re.IGNORECASE),
            "random_forests": re.compile(r"\brandom\s+forest(?:s)?\b|\brf\s+(?:classifier|regressor)\b", re.IGNORECASE),
            "xgboost": re.compile(r"\b(?:xgboost|xgb|extreme\s+gradient\s+boosting)\b", re.IGNORECASE),
        }

        # Output directory (keeping your current location)
        self.output_dir = PROCESSED_DATA_DIR / "ml_finance_papers"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def collect(self) -> None:
        """
        Execute the data collection: search for papers, retrieve details, and save outputs.
        """
        start_year = datetime.strptime(START_DATE, "%m/%d/%Y").year
        end_year = datetime.now(datetime.UTC).year
        self.logger.info(f"Searching for Machine Learning papers in journals: {self.journals} ({start_year}-present)")

        papers = self._search_crossref(start_year=start_year, end_year=end_year)
        if not papers:
            self.logger.warn("No papers found from Crossref search.")
            # Still write empty files to be consistent
            self._save_results([])
            self.logger.info(f"Saved 0 records to {self.output_dir}")
            return

        self.logger.info(f"Found {len(papers)} candidate papers from Crossref.")
        if self.max_records:
            papers = papers[: self.max_records]
            self.logger.info(f"Truncated paper list to --max-records={self.max_records}")

        results: list[dict[str, Any]] = []
        for idx, item in tqdm(
            list(enumerate(papers, start=1)),
            total=len(papers),
            desc="Retrieving Detailed Info for Papers",
        ):
            doi = item.get("DOI")
            # Crossref 'title' is usually a list
            raw_title = item.get("title")
            if isinstance(raw_title, list):
                title = raw_title[0] if raw_title else ""
            else:
                title = raw_title or ""

            raw_journal = item.get("container-title")
            if isinstance(raw_journal, list):
                journal = raw_journal[0] if raw_journal else ""
            else:
                journal = raw_journal or ""

            self.logger.info(f"[{idx}/{len(papers)}] Processing DOI: {doi} - \"{title}\"")
            try:
                pub_date = self._get_crossref_pub_date(item)

                # Get abstract (and possibly publicationDate) from Semantic Scholar
                paper_data = self._fetch_semantic_scholar_data(doi)
                abstract = paper_data.get("abstract") or ""
                if paper_data.get("publicationDate"):
                    pub_date = paper_data["publicationDate"]

                # Submission date via best-effort scrape
                submission_date = self._find_submission_date(item.get("URL") or f"https://doi.org/{doi}")

                # Model mentions from title + abstract
                text_for_search = f"{title} {abstract}"
                model_flags = {m: bool(p.search(text_for_search)) for m, p in self.model_patterns.items()}

                # Normalize dates
                pub_date_norm = self._normalize_date(pub_date)
                sub_date_norm = self._normalize_date(submission_date)
                if pub_date and pub_date_norm is None:
                    self.logger.warning(f"Could not parse publication date '{pub_date}' for DOI {doi}")
                if submission_date and sub_date_norm is None:
                    self.logger.warning(f"Could not parse submission date '{submission_date}' for DOI {doi}")

                result = {
                    "title": title or paper_data.get("title") or "",
                    "journal": journal or paper_data.get("venue") or paper_data.get("journal") or "",
                    "doi": doi,
                    "url": item.get("URL") or paper_data.get("url") or f"https://doi.org/{doi}",
                    "publication_date": pub_date_norm,
                    "submission_date": sub_date_norm,
                    "abstract": abstract,
                }
                # Add boolean model flags
                for model, flag in model_flags.items():
                    result[f"contains_{model}"] = flag
                results.append(result)

            except Exception as e:
                self.logger.error(f"Error processing DOI {doi}: {e}")
                continue

        self._save_results(results)
        self.logger.info(f"Saved {len(results)} records to {self.output_dir}")

    def _search_crossref(self, start_year: int, end_year: int) -> list[dict[str, Any]]:
        """
        Search Crossref for papers mentioning 'machine learning' in the specified journals,
        using proper multi-filter syntax and cursor-based pagination.

        Args:
            start_year (int): Lower bound of publication year.
            end_year (int): Upper bound of publication year.

        Returns:
            list[dict[str, Any]]: Crossref work items.
        """
        base_url = "https://api.crossref.org/works"

        # Build filter with separate container-title filters
        filter_parts = [
            f"from-pub-date:{start_year}-01-01",
            f"until-pub-date:{end_year}-12-31",
        ] + [f"container-title:{j}" for j in self.journals]

        params = {
            "query.bibliographic": "machine learning",
            "filter": ",".join(filter_parts),
            "select": "DOI,title,container-title,URL,published-print,published-online,created,issued",
            "rows": 200,           # polite page size
            "cursor": "*",         # enable deep paging
        }

        headers = {"User-Agent": "PaperCollector"}
        items: list[dict[str, Any]] = []
        seen_dois: set[str] = set()

        i = 1
        with tqdm(desc="Searching Crossref", unit="requests") as pbar:
            while True:
                try:
                    resp = requests.get(base_url, params=params, headers=headers, timeout=30)
                    if resp.status_code == 400:
                    # Most common cause: malformed filter; surface payload for debugging
                        raise RuntimeError(f"Crossref 400 Bad Request. Params: {params} | Body: {resp.text[:300]}")
                    resp.raise_for_status()
                except Exception as e:
                    self.logger.error(f"Crossref API request failed: {e}")
                    break

                msg = resp.json().get("message", {})
                page_items = msg.get("items", []) or []
                for it in page_items:
                    doi = (it.get("DOI") or "").lower()
                    if not doi or doi in seen_dois:
                        continue
                    seen_dois.add(doi)
                    items.append(it)
                    if i % 50 == 0:
                        self.logger.info(f"{len(items)} papers collected")
                    i += 1
                    if self.max_records and len(items) >= self.max_records:
                        break

                # Update progress bar with current count
                pbar.set_postfix(papers=len(items))
                pbar.update(1)
                
                # Stop if max_records hit or no more cursor
                if self.max_records and len(items) >= self.max_records:
                    break
                next_cursor = msg.get("next-cursor")
                if not next_cursor or not page_items:
                    break
                params["cursor"] = next_cursor
                time.sleep(0.3)

        return items

    def _fetch_semantic_scholar_data(self, doi: str) -> dict[str, Any]:
        """
        Fetch paper details from Semantic Scholar by DOI.

        Args:
            doi (str): Paper DOI.

        Returns:
            dict[str, Any]: Semantic Scholar fields.
        """
        base_url = f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}"
        params = {"fields": "title,venue,year,publicationDate,url,abstract"}
        headers: dict[str, str] = {}
        api_key = os.getenv("SEMANTICSCHOLAR_API_KEY")
        if api_key:
            headers["x-api-key"] = api_key

        self.logger.debug(f"Fetching Semantic Scholar data for DOI {doi}")
        try:
            resp = requests.get(base_url, params=params, headers=headers, timeout=30)
            if resp.status_code == 404:
                self.logger.debug(f"Semantic Scholar: DOI {doi} not found (404).")
                return {}
            resp.raise_for_status()
        except Exception as e:
            self.logger.warning(f"Semantic Scholar request failed for {doi}: {e}")
            return {}
        return resp.json()

    def _get_crossref_pub_date(self, item: dict[str, Any]) -> str | None:
        """
        Extract a publication date string from Crossref item (prefer online → print → issued → created).

        Args:
            item (dict[str, Any]): Crossref work item.

        Returns:
            str | None: Date string in YYYY-MM-DD (or raw for later parsing) or None.
        """
        if item.get("published-online", {}).get("date-parts"):
            return self._format_date_parts(item["published-online"]["date-parts"][0])
        if item.get("published-print", {}).get("date-parts"):
            return self._format_date_parts(item["published-print"]["date-parts"][0])
        if item.get("issued", {}).get("date-parts"):
            return self._format_date_parts(item["issued"]["date-parts"][0])
        if item.get("created", {}).get("date-parts"):
            return self._format_date_parts(item["created"]["date-parts"][0])
        return None

    def _format_date_parts(self, date_parts: list[int]) -> str:
        """
        Format a Crossref date-parts list (year, month, day) into YYYY-MM-DD.

        Args:
            date_parts (list[int]): e.g. [2020, 7, 15]

        Returns:
            str: Normalized YYYY-MM-DD (fills missing month/day with '01').
        """
        if not date_parts:
            return ""
        year = date_parts[0]
        month = date_parts[1] if len(date_parts) > 1 else 1
        day = date_parts[2] if len(date_parts) > 2 else 1
        try:
            return datetime(year, month, day).strftime("%Y-%m-%d")
        except Exception:
            return f"{year:04d}-{month:02d}-{day:02d}"

    def _find_submission_date(self, url: str) -> str | None:
        """
        Scrape the publisher page to find a 'Received' or 'Submitted' date string.

        Args:
            url (str): DOI resolver or publisher landing page.

        Returns:
            str | None: Raw date string if found, else None.
        """
        self.logger.debug(f"Scraping publisher page for submission date: {url}")
        try:
            resp = requests.get(url, headers={"Accept": "text/html"}, timeout=15)
            resp.raise_for_status()
        except Exception as e:
            self.logger.warning(f"Failed to fetch publisher page at {url}: {e}")
            return None

        soup = BeautifulSoup(resp.text, "html.parser")
        text = soup.get_text(separator=" ")
        patterns = [
            r"\bReceived\s*[:\-]?\s*(.+?)(?=(Accepted|Published|;|$))",
            r"\bSubmitted\s*[:\-]?\s*(.+?)(?=(Accepted|Published|;|$))",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                date_str = match.group(1).strip()
                date_str = re.split(r"[\.;,]", date_str)[0]
                self.logger.debug(f"Found submission date string: '{date_str}'")
                return date_str
        return None

    def _normalize_date(self, date_str: str | None) -> str | None:
        """
        Convert arbitrary date strings to YYYY-MM-DD. Returns None if parsing fails.

        Args:
            date_str (str | None): Date string.

        Returns:
            str | None: ISO date or None.
        """
        if not date_str:
            return None
        try:
            dt = dateparser.parse(date_str)
            if not dt:
                return None
            return dt.strftime("%Y-%m-%d")
        except Exception:
            return None

    def _save_results(self, results: list[dict[str, Any]]) -> None:
        """
        Save the results list to CSV and JSONL files in the output directory.

        Args:
            results (list[dict[str, Any]]): Collected records.
        """
        # Always write files, even if empty, for reproducibility
        csv_path = self.output_dir / "ml_finance_papers.csv"
        jsonl_path = self.output_dir / "ml_finance_papers.jsonl"

        try:
            import pandas as pd

            df = pd.DataFrame(results)
            # Ensure consistent column order: first 7 fields, then model flags
            cols = ["title", "journal", "publication_date", "submission_date", "doi", "url", "abstract"]
            model_cols = [f"contains_{m}" for m in self.model_patterns.keys()]
            for col in model_cols:
                if col not in df.columns:
                    df[col] = False
            ordered_cols = cols + model_cols
            # If empty results, still create columns
            for c in ordered_cols:
                if c not in df.columns:
                    df[c] = []  # will yield empty columns
            df.to_csv(csv_path, index=False, columns=ordered_cols)
        except ImportError:
            import csv

            fieldnames = [
                "title",
                "journal",
                "publication_date",
                "submission_date",
                "doi",
                "url",
                "abstract",
            ] + [f"contains_{m}" for m in self.model_patterns.keys()]

            with open(csv_path, mode="w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in results:
                    out = {k: row.get(k) for k in fieldnames}
                    for m in [f"contains_{x}" for x in self.model_patterns.keys()]:
                        out[m] = bool(row.get(m, False))
                    writer.writerow(out)

        with open(jsonl_path, "w", encoding="utf-8") as jf:
            for rec in results:
                jf.write(json.dumps(rec, ensure_ascii=False) + "\n")

        self.logger.info(f"Wrote CSV to {csv_path} and JSONL to {jsonl_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=f"Collect ML-related finance papers ({START_DATE}-present).")
    parser.add_argument(
        "--include-subtier",
        action="store_true",
        default=False,
        help="Include sub-tier finance journals in the search.",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Max number of records to fetch (for dry run or testing).",
    )
    args = parser.parse_args()

    collector = PaperCollector(include_subtier=args.include_subtier, max_records=args.max_records)
    collector.collect()
