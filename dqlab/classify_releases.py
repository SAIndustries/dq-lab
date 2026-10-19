from pathlib import Path

import polars as pl
import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load(open(ROOT / "config/universe.yaml"))
T = ROOT / "data/normalized/earnings_text.parquet"
F = ROOT / "data/normalized/filings.parquet"

# Patterns are checked in order; the first match wins
NON_RELEASE = r"(?i)condensed consolidated|presentation and q&a|financial & operating review"
LETTER = r"(?i)quarter[^.]{0,40}(update|shareholder letter)|\bQ[1-4]\b[^.]{0,20}\bupdate\b"
DELIVERIES = r"(?i)production,? (and |& )?deliveries|delivers? [\d,]+ (vehicles|cars)"
PERIOD = r"(?i)(first|second|third|fourth)[- ]quarter|\bQ[1-4]\b|full[- ]year|\bFY ?\d{2,4}\b"
VERB = r"(?i)\b(report|announce|post|earn)\w*\b.{0,120}\b(quarter|fiscal|full[- ]year|results|earnings)"


def classify(t):
    top = pl.col("text").str.slice(0, 250)     # exhibit title region
    head = pl.col("text").str.slice(0, 600)    # headline region
    wide = pl.col("text").str.slice(0, 2000)   # allows for long letterheads (e.g. XOM)
    kind = (pl.when(pl.col("doc_type") == "PRIMARY").then(pl.lit("other"))
              .when(top.str.contains(NON_RELEASE)).then(pl.lit("other"))
              .when(head.str.contains(LETTER)).then(pl.lit("earnings"))
              .when(head.str.contains(DELIVERIES)).then(pl.lit("deliveries"))
              .when(head.str.contains(PERIOD) | wide.str.contains(VERB)).then(pl.lit("earnings"))
              .otherwise(pl.lit("other")))
    t = t.with_columns(kind.alias("release_kind"))

    # Same release filed under two CIKs (e.g. Google Inc. and Alphabet in Oct 2015).
    # Skip the first 200 chars, which contain the per-filing filename.
    t = t.with_columns(pl.col("text").str.slice(200).hash().alias("_h"))
    t = t.with_columns(
        pl.when(pl.col("accession") != pl.col("accession").min().over("symbol", "_h"))
          .then(pl.lit("duplicate")).otherwise(pl.col("release_kind"))
          .alias("release_kind")).drop("_h")

    # Manual overrides for cases no rule should guess (accession -> kind)
    ov = CFG.get("release_overrides") or {}
    if ov:
        t = t.with_columns(
            pl.when(pl.col("accession").is_in(list(ov)))
              .then(pl.col("accession").replace_strict(ov, default=None))
              .otherwise(pl.col("release_kind")).alias("release_kind"))
    return t


def run():
    t = classify(pl.read_parquet(T).drop("release_kind", strict=False))
    t.write_parquet(T)

    a = t.join(pl.read_parquet(F).select("accession", "filing_date"), on="accession")
    a = a.with_columns(pl.col("filing_date").dt.year().alias("year"))
    print(a.group_by("release_kind").len().sort("release_kind"))

    # Audit: full years should have exactly 4 scheduled earnings releases
    earn = a.filter(pl.col("release_kind") == "earnings")
    bad = (earn.group_by("symbol", "year").len()
               .filter((pl.col("len") != 4) & (pl.col("year") < 2026)))
    with pl.Config(tbl_rows=40):
        print("Company-years (before 2026) without exactly 4 earnings releases:")
        print(bad.sort("symbol", "year"))

    # Every release (any kind) in a bad company-year, to see what was missed or extra
    with pl.Config(tbl_rows=80, fmt_str_lengths=110):
        print(a.join(bad.select("symbol", "year"), on=["symbol", "year"])
               .select("symbol", "filing_date", "release_kind", "accession",
                       pl.col("text").str.slice(30, 110).alias("headline"))
               .sort("symbol", "filing_date"))


if __name__ == "__main__":
    run()