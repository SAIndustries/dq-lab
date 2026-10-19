import re
from pathlib import Path

import polars as pl
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data/raw/edgar_exhibits"
OUT = ROOT / "data/normalized/earnings_text.parquet"


def html_to_text(raw: bytes) -> str:
    soup = BeautifulSoup(raw, "lxml")
    for tag in soup(["script", "style"]):
        tag.decompose()
    return re.sub(r"\s+", " ", soup.get_text(" ")).strip()


def run():
    idx = pl.read_parquet(SRC / "_index.parquet").filter(pl.col("status") == "ok")
    out = []
    for acc, fname in idx.select("accession", "filename").iter_rows():
        text = html_to_text((SRC / f"{acc}__{fname}").read_bytes())
        out.append({"accession": acc, "text": text, "n_words": len(text.split())})

    t = idx.join(pl.DataFrame(out), on="accession")
    t.write_parquet(OUT)
    print(f"Extracted {t.height} documents -> {OUT.name}")


if __name__ == "__main__":
    run()