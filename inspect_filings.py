import polars as pl

f = pl.read_parquet("data/normalized/filings.parquet").with_columns(
    pl.col("acceptance_raw").str.slice(0, 19)
      .str.to_datetime("%Y-%m-%dT%H:%M:%S").alias("acc"))

# 1. What clock are these timestamps on?
with pl.Config(tbl_rows=30):
    print(f.group_by(pl.col("acc").dt.hour().alias("hour")).len().sort("hour"))

# 2. How often does the official filing date differ from the acceptance date?
print(f.with_columns((pl.col("filing_date") != pl.col("acc").dt.date()).alias("date_differs"))
       .group_by("date_differs").len())

# 3. Which 8-K item types do these companies file?
items = (f.with_columns(pl.col("items").str.split(","))
          .explode("items").group_by("items").len().sort("len", descending=True))
print(items.head(12))