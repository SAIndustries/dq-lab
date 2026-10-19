import os
import time

import pandas as pd

from .base import get_json, save_raw

BASE = "https://data.sec.gov/submissions/"


def _ua():
    return {"User-Agent": os.environ["SEC_USER_AGENT"]}


def cik_map():
    """Current ticker -> CIK map from the SEC (10-digit, zero-padded)."""
    data = get_json("https://www.sec.gov/files/company_tickers.json", headers=_ua())
    return {v["ticker"]: str(v["cik_str"]).zfill(10) for v in data.values()}


def fetch_filings(sym, cik):
    """Fetch every filing for one CIK, including older paged submission files."""
    sub = get_json(f"{BASE}CIK{cik}.json", headers=_ua())
    frames = [pd.DataFrame(sub["filings"]["recent"])]

    # The "recent" block holds only ~1000 filings; older ones live in extra pages.
    for extra in sub["filings"].get("files", []):
        time.sleep(0.2)  # SEC limit: 10 requests/second
        frames.append(pd.DataFrame(get_json(BASE + extra["name"], headers=_ua())))

    df = pd.concat(frames, ignore_index=True)
    df["cik"] = cik

    # Mixed-type object columns break Parquet; store them as strings.
    obj = df.select_dtypes("object").columns
    df[obj] = df[obj].astype(str)

    # One file per (symbol, CIK) so multiple CIKs for a ticker don't overwrite.
    save_raw(df, "edgar", f"{sym}_{cik}", f"{BASE}CIK{cik}.json")