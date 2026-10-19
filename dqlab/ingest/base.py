# dqlab/ingest/base.py
import hashlib, json, time
from datetime import datetime, timezone
from pathlib import Path
import requests

RAW = Path("data/raw")
MANIFEST = RAW / "manifest.jsonl"
HEADERS = {"User-Agent": "data-quality-lab"}

def get_json(url, params=None, retries=4, headers=None):
    for i in range(retries):
        r = requests.get(url, params=params, headers=headers or HEADERS, timeout=30)
        if r.status_code == 200:
            return r.json()
        time.sleep(2 ** i)
    r.raise_for_status()

def save_raw(df, vendor, symbol, source=""):
    if len(df) == 0:
        raise ValueError(f"{vendor}/{symbol}: empty response, not saving")
    out = RAW / vendor / f"{symbol}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    entry = {
        "vendor": vendor, "symbol": symbol, "rows": len(df),
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
        "source": source,
    }
    RAW.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open("a") as f:
        f.write(json.dumps(entry) + "\n")