import datetime as dt
from pathlib import Path

import polars as pl
import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load(open(ROOT / "config/universe.yaml"))
START = dt.date.fromisoformat(CFG["start"])
RAW = ROOT / "data/raw/edgar"
HDR = ROOT / "data/raw/edgar_headers/all.parquet"
OUT = ROOT / "data/normalized"
NY = "America/New_York"


def load_filings():
    """Read every raw EDGAR file and keep 8-Ks from the start date onward."""
    frames = []
    for p in sorted(RAW.glob("*.parquet")):
        symbol = p.stem.split("_")[0]  # "XOM_0000034088" -> "XOM"
        frames.append(pl.read_parquet(p).select(
            pl.lit(symbol).alias("symbol"),
            pl.col("cik").cast(pl.Utf8),
            pl.col("accessionNumber").alias("accession"),
            "form",
            pl.col("filingDate").str.to_date().alias("filing_date"),
            pl.col("acceptanceDateTime").alias("acceptance_raw"),
            "items",
            pl.col("primaryDocument").alias("primary_doc"),
        ))

    return (pl.concat(frames)
              .filter(pl.col("form").is_in(["8-K", "8-K/A"])
                      & (pl.col("filing_date") >= START))
              .unique("accession")              # accession numbers are globally unique
              .sort("symbol", "filing_date"))


def add_acceptance_times(f):
    """Attach the authoritative acceptance time from filing headers.

    The submissions API's acceptanceDateTime follows one of two conventions
    per CIK: genuine UTC, or Eastern Time with the UTC offset applied twice.
    Header ACCEPTANCE-DATETIME (Eastern Time) is ground truth. Where it is
    missing, the CIK's convention, learned from its other filings, is used.
    If a CIK has no header evidence at all, the timestamp is left null:
    never guessed.
    """
    hd = pl.read_parquet(HDR).select(
        "accession",
        pl.col("header_acceptance").str.to_datetime("%Y%m%d%H%M%S")
          .dt.replace_time_zone(NY, ambiguous="earliest").alias("acceptance_et"))
    f = f.join(hd, on="accession", how="left")

    # The two API conventions, both expressed in New York time
    api_utc = (pl.col("acceptance_raw").str.slice(0, 19)
                 .str.to_datetime("%Y-%m-%dT%H:%M:%S").dt.replace_time_zone("UTC"))
    h1 = api_utc.dt.convert_time_zone(NY)                       # genuine UTC
    h3 = (h1.dt.replace_time_zone(None).dt.replace_time_zone("UTC")
            .dt.convert_time_zone(NY))                          # offset applied twice

    # Learn each CIK's convention from filings that have header ground truth
    conv = (f.filter(pl.col("acceptance_et").is_not_null())
             .with_columns(((h1 - pl.col("acceptance_et")).dt.total_seconds().abs() <= 60)
                           .alias("_h1_ok"))
             .group_by("cik").agg(
                 (pl.col("_h1_ok").mean() > 0.5).alias("cik_true_utc"),
                 pl.col("_h1_ok").mean().round(3).alias("_h1_share")))

    # A CIK should follow one convention completely; flag any that don't
    mixed = conv.filter(pl.col("_h1_share").is_between(0.01, 0.99))
    if mixed.height:
        print("WARNING: CIKs with mixed timestamp conventions:")
        print(mixed)

    # CIKs with no header evidence at all cannot have their convention inferred
    no_evidence = (f.select("cik").unique()
                    .join(conv.select("cik"), on="cik", how="anti"))
    if no_evidence.height:
        print("WARNING: CIKs with no header timestamps (run edgar_headers again):")
        print(no_evidence)

    f = f.join(conv.select("cik", "cik_true_utc"), on="cik", how="left")

    # Header time where available; otherwise the CIK's learned convention;
    # if the convention is unknown, the timestamp stays null.
    f = f.with_columns(
        pl.when(pl.col("acceptance_et").is_not_null()).then(pl.lit("header"))
          .when(pl.col("cik_true_utc").is_not_null()).then(pl.lit("api_inferred"))
          .otherwise(pl.lit("missing")).alias("acceptance_source"),
        pl.coalesce("acceptance_et",
                    pl.when(pl.col("cik_true_utc")).then(h1)
                      .when(~pl.col("cik_true_utc")).then(h3))   # unknown CIK -> stays null
          .alias("acceptance_et"),
    ).with_columns(
        pl.col("acceptance_et").dt.convert_time_zone("UTC").alias("acceptance_utc"))

    return f


def run():
    f = load_filings()

    if HDR.exists():
        f = add_acceptance_times(f)
        print(f.group_by("acceptance_source").len().sort("acceptance_source"))
        missing = f["acceptance_et"].null_count()
        if missing:
            print(f"WARNING: {missing} filing(s) without a timestamp")
    else:
        print("NOTE: no header timestamps yet; run `python -m dqlab.ingest.edgar_headers`")

    OUT.mkdir(parents=True, exist_ok=True)
    f.write_parquet(OUT / "filings.parquet")

    with pl.Config(tbl_rows=30):
        print(f.group_by("symbol", "cik").len().sort("symbol", "cik"))


if __name__ == "__main__":
    run()