import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import polars as pl
import requests
import yaml
from dotenv import load_dotenv

load_dotenv()
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/raw/edgar_exhibits"
MANIFEST = ROOT / "data/raw/manifest.jsonl"
CFG = yaml.safe_load(open(ROOT / "config/universe.yaml"))

# Matches "<TYPE>EX-99.1 ... <FILENAME>name.htm" (tags may be HTML-escaped)
DOC = re.compile(r"TYPE(?:>|&gt;)\s*(EX-99[\w.]*)\s.*?FILENAME(?:>|&gt;)\s*([^\s<&]+)", re.S)

# When a filing has several EX-99 exhibits, prefer the press release
EXHIBIT_RANK = {"EX-99.1": 0, "EX-99.01": 1, "EX-99": 2}


def _get(url, ua):
    for i in range(4):
        r = requests.get(url, headers=ua, timeout=30)
        time.sleep(0.12)  # stay under SEC's 10 requests/second
        if r.status_code == 200:
            return r
        time.sleep(2 ** i)
    r.raise_for_status()


def _log(name, content, url):
    entry = {"vendor": "edgar_exhibits", "symbol": name, "bytes": len(content),
             "fetched_at": datetime.now(timezone.utc).isoformat(),
             "sha256": hashlib.sha256(content).hexdigest(), "source": url}
    with MANIFEST.open("a") as f:
        f.write(json.dumps(entry) + "\n")


def filing_url(cik, accession):
    return f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession.replace('-', '')}/"


def earnings_filings():
    """Original 8-Ks tagged Item 2.02, plus documented exceptions from config
    (filings whose item codes are wrong, e.g. CMG Q2 2016 filed as 9.01 only)."""
    extra = CFG.get("extra_earnings_filings") or []
    return (pl.read_parquet(ROOT / "data/normalized/filings.parquet")
              .filter((pl.col("form") == "8-K")
                      & (pl.col("items").str.contains("2.02", literal=True)
                         | pl.col("accession").is_in(extra)))
              .select("symbol", "cik", "accession", "primary_doc"))


def run():
    ua = {"User-Agent": os.environ["SEC_USER_AGENT"]}
    f = earnings_filings()
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"{f.height} earnings 8-Ks to process")

    rows = []
    for i, (sym, cik, acc, primary) in enumerate(f.iter_rows()):
        base = filing_url(cik, acc)
        try:
            hdr = _get(base + f"{acc}-index-headers.html", ua).text
            found = DOC.findall(hdr)  # [(type, filename), ...]
            found.sort(key=lambda d: EXHIBIT_RANK.get(d[0], 3))
            doc_type, fname = found[0] if found else ("PRIMARY", primary)
            dest = OUT / f"{acc}__{fname}"
            if not dest.exists():                      # resumable: skip what's already saved
                r = _get(base + fname, ua)
                dest.write_bytes(r.content)
                _log(dest.name, r.content, base + fname)
            status = "ok"
        except Exception as e:
            doc_type, fname, status = None, None, f"error: {e}"
        rows.append({"symbol": sym, "cik": cik, "accession": acc,
                     "doc_type": doc_type, "filename": fname, "status": status})
        if i % 100 == 0:
            print(f"{i}/{f.height}")

    idx = pl.DataFrame(rows)
    idx.write_parquet(OUT / "_index.parquet")
    print(idx.group_by("status").len())


if __name__ == "__main__":
    run()