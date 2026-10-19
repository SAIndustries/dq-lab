import polars as pl

f = pl.read_parquet("data/normalized/filings.parquet").with_columns(
    pl.col("acceptance_raw").str.slice(0, 19)
      .str.to_datetime("%Y-%m-%dT%H:%M:%S").alias("acc"))

# 1. Hour distribution of raw acceptance timestamps
with pl.Config(tbl_rows=30):
    print(f.group_by(pl.col("acc").dt.hour().alias("hour")).len().sort("hour"))

# 2. Filing date vs raw acceptance date
print(f.with_columns((pl.col("filing_date") != pl.col("acc").dt.date()).alias("date_differs"))
       .group_by("date_differs").len())

# 3. 8-K item types
items = (f.with_columns(pl.col("items").str.split(","))
          .explode("items").group_by("items").len().sort("len", descending=True))
print(items.head(12))

# A. Are hour-0 timestamps placeholders (exactly 00:00:00)?
print(f.filter(pl.col("acc").dt.hour() == 0)
       .group_by(pl.col("acc").dt.strftime("%H:%M:%S").alias("hms"))
       .len().sort("len", descending=True).head(5))

# B. Did the timestamp convention change over time?
with pl.Config(tbl_rows=15):
    print(f.with_columns(pl.col("acc").dt.year().alias("year"),
                         pl.col("acc").dt.hour().alias("h"))
           .group_by("year")
           .agg(pl.len().alias("n"),
                pl.col("h").is_between(0, 7).mean().round(2).alias("share_00_07"),
                (pl.col("acc").dt.strftime("%H:%M:%S") == "00:00:00").sum().alias("exact_midnight"))
           .sort("year"))

# C. Known anchor: Apple releases earnings at 4:30 pm ET
with pl.Config(fmt_str_lengths=40):
    print(f.filter((pl.col("symbol") == "AAPL")
                   & pl.col("items").str.contains("2.02", literal=True))
           .select("filing_date", "acceptance_raw", "accession").tail(8))