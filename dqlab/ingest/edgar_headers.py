import os
import re
import time
from pathlib import Path

import pandas as pd
import polars as pl
import requests
from dotenv import load_dotenv

from .base import save_raw

load_dotenv()
ROOT = Path(__file__).resolve().parents[2]
# In the HTML the tag is escaped, so match both "<...>" and "&lt;...&gt;" forms
PAT = re.compile(r"ACCEPTANCE-DATETIME(?:>|&gt;)\s*(\d{14})")


def header_url(cik, accession):
    return (f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/"
            f"{accession.replace('-', '')}/{accession}-index-headers.html")


def run():
    f = pl.read_parquet(ROOT / "data/normalized/filings.parquet").select("symbol", "cik", "accession")
    ua = {"User-Agent": os.environ["SEC_USER_AGENT"]}
    rows = []
    for i, (sym, cik, acc) in enumerate(f.iter_rows()):
        try:
            r = requests.get(header_url(cik, acc), headers=ua, timeout=30)
            m = PAT.search(r.text) if r.status_code == 200 else None
            rows.append({"accession": acc, "symbol": sym, "status": r.status_code,
                         "header_acceptance": m.group(1) if m else None})
        except Exception:
            rows.append({"accession": acc, "symbol": sym, "status": -1,
                         "header_acceptance": None})
        time.sleep(0.12)  # stay under 10 requests/second
        if i % 200 == 0:
            print(f"{i}/{len(f)}")
    save_raw(pd.DataFrame(rows), "edgar_headers", "all", "per-filing index-headers.html")


if __name__ == "__main__":
    run()