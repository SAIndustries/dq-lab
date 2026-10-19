import polars as pl

NY = "America/New_York"

def utc_to_ny(e):
    return e.dt.replace_time_zone("UTC").dt.convert_time_zone(NY).dt.replace_time_zone(None)

f = pl.read_parquet("data/normalized/filings.parquet").with_columns(
    pl.col("acceptance_raw").str.slice(0, 19).str.to_datetime("%Y-%m-%dT%H:%M:%S").alias("raw"))
hd = pl.read_parquet("data/raw/edgar_headers/all.parquet").select(
    "accession", "status",
    pl.col("header_acceptance").str.to_datetime("%Y%m%d%H%M%S").alias("truth_et"))

m = f.join(hd, on="accession").with_columns(utc_to_ny(pl.col("raw")).alias("H1"))
m = m.with_columns(utc_to_ny(pl.col("H1")).alias("H3"))
m = m.with_columns(
    ((pl.col("H1") - pl.col("truth_et")).dt.total_minutes().abs() <= 1).alias("H1_ok"),
    ((pl.col("H3") - pl.col("truth_et")).dt.total_minutes().abs() <= 1).alias("H3_ok"))

# 1. Coverage and overall accuracy
print(m.select(pl.len().alias("filings"),
               pl.col("truth_et").is_null().sum().alias("header_missing"),
               pl.col("H1_ok").mean().round(3).alias("H1_correct"),
               pl.col("H3_ok").mean().round(3).alias("H3_correct")))

# 2. Per company: which convention, and anything neither explains?
with pl.Config(tbl_rows=25):
    print(m.group_by("symbol").agg(
        pl.col("H1_ok").mean().round(2).alias("H1"),
        pl.col("H3_ok").mean().round(2).alias("H3"),
        (~pl.col("H1_ok") & ~pl.col("H3_ok")).sum().alias("neither"))
     .sort("symbol"))

# 3. Examples where neither hypothesis matches
with pl.Config(tbl_rows=10):
    print(m.filter(~pl.col("H1_ok") & ~pl.col("H3_ok"))
           .select("symbol", "accession", "acceptance_raw", "truth_et", "H1", "H3").head(10))

print(m.group_by("symbol", "cik").agg(pl.col("H1_ok").mean().round(2).alias("H1"),
                                      pl.col("H3_ok").mean().round(2).alias("H3"), pl.len())
       .filter(pl.col("symbol") == "XOM"))