import datetime as dt
import polars as pl

iss = pl.read_parquet("data/validation/issues.parquet")
bars = pl.read_parquet("data/normalized/bars.parquet")

dd = iss.filter(pl.col("check").str.ends_with("dividend_adjustment")
                & (pl.col("vendor") != "stooq"))
with pl.Config(fmt_str_lengths=80):
    print(dd)

for sym, d in dd.select("symbol", "date").iter_rows():
    window = bars.filter(
        (pl.col("symbol") == sym)
        & pl.col("date").is_between(d - dt.timedelta(days=4), d + dt.timedelta(days=4))
        & pl.col("vendor").is_in(["tiingo", "yfinance"])
    ).select("vendor", "date", "close", "adj_close", "dividend").sort("date", "vendor")
    print(f"\n{sym} around {d}")
    with pl.Config(tbl_rows=20):
        print(window)