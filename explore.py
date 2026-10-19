import polars as pl

iss = pl.read_parquet("data/validation/issues.parquet")
bars = pl.read_parquet("data/normalized/bars.parquet")
CRYPTO = ["binance", "coinbase", "kraken"]

# 1. Which vendor pairs and symbols drive the return gaps?
gaps = iss.filter(pl.col("check") == "cross_vendor_return_gap").with_columns(
    pl.col("detail").str.extract(r"^(\w+ vs \w+)", 1).alias("pair"))
print(gaps.group_by("pair").len())
print(gaps.group_by("pair", "symbol").len().sort("len", descending=True).head(15))

# 2. Do Stooq gaps land on ex-dividend dates?
exdiv = (bars.filter((pl.col("vendor") == "tiingo") & (pl.col("dividend") > 0))
             .select("symbol", "date").with_columns(pl.lit(True).alias("exdiv")))
sg = gaps.filter(pl.col("pair").str.starts_with("stooq")).join(
    exdiv, on=["symbol", "date"], how="left")
print(sg.group_by(pl.col("exdiv").fill_null(False)).len())
print(sg.sort("date").head(10))

# 3. The stale prices: same days at every vendor?
print(iss.filter(pl.col("check") == "stale_price").sort("symbol", "date", "vendor"))

# 4. Outliers: market event (all vendors) or data error (one vendor)?
out = iss.filter((pl.col("check") == "return_outlier") & ~pl.col("vendor").is_in(CRYPTO))
agree = out.group_by("symbol", "date").agg(
    pl.col("vendor").sort().str.join(",").alias("vendors"), pl.len().alias("n"))
print(agree.group_by("n").len().sort("n"))
print(agree.filter(pl.col("n") == 1).sort("date"))