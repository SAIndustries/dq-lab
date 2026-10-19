import polars as pl

iss = pl.read_parquet("data/validation/issues.parquet")
b = pl.read_parquet("data/normalized/bars.parquet")

# Every dividend-related issue for Costco
with pl.Config(tbl_rows=30, fmt_str_lengths=60):
    print(iss.filter((pl.col("symbol") == "COST") & pl.col("check").str.contains("dividend"))
             .select("check", "vendor", "date", "detail").sort("date"))

# Costco's special dividends according to Tiingo (regular ones are well under $2)
print(b.filter((pl.col("vendor") == "tiingo") & (pl.col("symbol") == "COST")
               & (pl.col("dividend") > 2))
       .select("date", "dividend", "close"))

# Sanity check: PEP and UNH both show -16.6% total return since 2023
d = b.filter((pl.col("vendor") == "tiingo") & pl.col("symbol").is_in(["PEP", "UNH"])
             & (pl.col("date") >= pl.date(2023, 1, 1)))
print(d.group_by("symbol").agg(
    (pl.col("adj_close").sort_by("date").last() / pl.col("adj_close").sort_by("date").first() - 1)
    .round(5).alias("total_return")))