import os, time
import pandas as pd
from .base import get_json, save_raw

BASE = "https://data.sec.gov/submissions/"

def _ua():
    return {"User-Agent": os.environ["SEC_USER_AGENT"]}

def cik_map():
    data = get_json("https://www.sec.gov/files/company_tickers.json", headers=_ua())
    return {v["ticker"]: str(v["cik_str"]).zfill(10) for v in data.values()}

def fetch_filings(sym, cik):
    sub = get_json(f"{BASE}CIK{cik}.json", headers=_ua())
    frames = [pd.DataFrame(sub["filings"]["recent"])]
    for extra in sub["filings"].get("files", []):     # older filings live in extra pages
        time.sleep(0.2)                                # SEC limit: 10 requests/second
        frames.append(pd.DataFrame(get_json(BASE + extra["name"], headers=_ua())))
    df = pd.concat(frames, ignore_index=True)
    df["cik"] = cik
    obj = df.select_dtypes("object").columns           # mixed types break Parquet
    df[obj] = df[obj].astype(str)
    save_raw(df, "edgar", sym, f"{BASE}CIK{cik}.json")