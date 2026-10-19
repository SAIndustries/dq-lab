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

EQ = ["stooq", "tiingo", "yfinance"]
win = lambda sym, a, z: (bars.filter((pl.col("symbol") == sym) & pl.col("vendor").is_in(EQ)
                                     & pl.col("date").is_between(a, z))
                             .select("vendor", "date", "open", "high", "low", "close", "volume")
                             .sort("date", "vendor"))
print(win("AAPL", pl.date(2015, 3, 9), pl.date(2015, 3, 16)))
print(win("AMZN", pl.date(2023, 9, 25), pl.date(2023, 9, 29)))

p = (bars.filter(pl.col("vendor").is_in(["stooq", "tiingo"]))
         .with_columns(pl.coalesce("adj_close", "close").alias("px"))
         .sort("vendor", "symbol", "date")
         .with_columns((pl.col("px") / pl.col("px").shift(1)).over("vendor", "symbol").log().alias("ret"),
                       pl.col("close").shift(1).over("vendor", "symbol").alias("prev_close")))
s = p.filter(pl.col("vendor") == "stooq").select("symbol", "date", pl.col("ret").alias("ret_stooq"))
t = p.filter(pl.col("vendor") == "tiingo").select(
        "symbol", "date", pl.col("ret").alias("ret_tiingo"),
        (pl.col("dividend") / pl.col("prev_close")).alias("div_yield"))
d = (s.join(t, on=["symbol", "date"]).filter(pl.col("div_yield") > 0)
      .with_columns((pl.col("ret_stooq") - pl.col("ret_tiingo")).alias("gap")))
print(d.select(pl.corr("gap", "div_yield").alias("corr"),
               (pl.col("gap") / pl.col("div_yield")).median().alias("gap_over_yield")))

aapl = win("AAPL", pl.date(2015, 3, 9), pl.date(2015, 3, 16)).with_columns(
    pl.col("close").pct_change().over("vendor").round(4).alias("ret"))
with pl.Config(tbl_rows=30):
    print(aapl.select("vendor", "date", "close", "ret", "volume"))

flag = gaps.filter(pl.col("pair") == "stooq vs tiingo").select("symbol", "date")
f = d.join(flag, on=["symbol", "date"]).with_columns(
    (pl.col("gap") / pl.col("div_yield")).alias("ratio"))
print(f.select(pl.len().alias("n"), pl.corr("gap", "div_yield").alias("corr"),
               pl.col("ratio").quantile(0.1).alias("p10"),
               pl.col("ratio").median().alias("p50"),
               pl.col("ratio").quantile(0.9).alias("p90")))
with pl.Config(tbl_rows=15):
    print(f.select("symbol", "date", pl.col("ret_stooq").round(4),
                   pl.col("ret_tiingo").round(4), pl.col("div_yield").round(4),
                   pl.col("ratio").round(2)).sort("date").head(15))


for shift in (-1, 0, 1):
    t_sh = t.with_columns(pl.col("date").shift(shift).over("symbol"))  # align neighbouring day
    m = s.join(t_sh, on=["symbol", "date"]).filter(pl.col("div_yield") > 0)
    print(shift, m.select((pl.col("ret_stooq") - pl.col("ret_tiingo")).abs().median()))

ex = d.with_columns(((pl.col("gap") / pl.col("div_yield") + 1).abs() < 0.1).alias("missed"))
print(ex.group_by(pl.col("date").dt.year().alias("year"))
        .agg(pl.len().alias("ex_div_events"), pl.col("missed").sum().alias("missed"))
        .with_columns((pl.col("missed") / pl.col("ex_div_events")).round(2).alias("share"))
        .sort("year"))

def check_stale(b, min_run=3):
    cols = ["open", "high", "low", "close", "volume"]
    s = b.sort(*KEYS, "date").with_columns(
        pl.any_horizontal([pl.col(c) != pl.col(c).shift(1).over(KEYS) for c in cols])
          .fill_null(True).alias("chg"))
    s = s.with_columns(pl.col("chg").cum_sum().over(KEYS).alias("run_id"))
    runs = (s.group_by(*KEYS, "run_id")
             .agg(pl.col("date").min(), pl.len().alias("n"), pl.col("close").first())
             .filter(pl.col("n") >= min_run))
    return issues(runs, "stale_price", "medium",
                  pl.format("OHLCV unchanged for {} rows (close {})", "n", "close"))

def test_repeated_close_with_moving_ohlc_not_stale():
    b = make_bars([100.0, 102.0, 102.0, 102.0, 103.0]).with_columns(
        (pl.col("high") + pl.int_range(pl.len())).alias("high"))
    assert len(v.check_stale(b)) == 0

imp = (ex.filter(pl.col("missed"))
         .group_by("symbol")
         .agg(pl.len().alias("missed_events"),
              (pl.col("div_yield").sum() * 100).round(2).alias("understated_return_pct"))
         .sort("understated_return_pct", descending=True))
print(imp)