import datetime as dt
import polars as pl

f = pl.read_parquet("data/normalized/filings.parquet")
print(f.filter((pl.col("symbol") == "CMG")
               & pl.col("filing_date").is_between(dt.date(2016, 7, 1), dt.date(2016, 8, 31)))
       .select("filing_date", "form", "items", "accession"))