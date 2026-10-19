import datetime as dt
from pathlib import Path
import polars as pl
import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load(open(ROOT / "config/universe.yaml"))
START = dt.date.fromisoformat(CFG["start"])

def run():
    frames = []
    for p in sorted((ROOT / "data/raw/edgar").glob("*.parquet")):
        frames.append(pl.read_parquet(p).select(
            pl.lit(p.stem).alias("symbol"), "cik",
            pl.col("accessionNumber").alias("accession"), "form",
            pl.col("filingDate").str.to_date().alias("filing_date"),
            pl.col("acceptanceDateTime").alias("acceptance_raw"),
            "items", pl.col("primaryDocument").alias("primary_doc")))
    f = (pl.concat(frames)
           .filter(pl.col("form").is_in(["8-K", "8-K/A"]) & (pl.col("filing_date") >= START))
           .sort("symbol", "filing_date"))
    f.write_parquet(ROOT / "data/normalized/filings.parquet")
    with pl.Config(tbl_rows=25):
        print(f.group_by("symbol").len().sort("symbol"))

if __name__ == "__main__":
    run()