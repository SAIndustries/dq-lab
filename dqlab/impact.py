import datetime as dt
from pathlib import Path

import polars as pl
import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load(open(ROOT / "config/universe.yaml"))
BARS = ROOT / "data/normalized/bars.parquet"
ISSUES = ROOT / "data/validation/issues.parquet"
START = dt.date.fromisoformat(CFG["start"])
BREAK = dt.date(2023, 1, 1)     # Stooq's dividend adjustments degrade from here
CRYPTO = ["binance", "coinbase", "kraken"]
K = 5                           # stocks per leg


def load_daily():
    """Adjusted price per vendor, plus 'tiingo_raw' (raw close misused as adjusted)."""
    b = (pl.read_parquet(BARS)
           .filter(~pl.col("vendor").is_in(CRYPTO) & (pl.col("date") >= START)))
    adj = b.select("vendor", "symbol", "date", pl.coalesce("adj_close", "close").alias("px"))
    raw = (b.filter(pl.col("vendor") == "tiingo")
            .select(pl.lit("tiingo_raw").alias("vendor"), "symbol", "date",
                    pl.col("close").alias("px")))
    d = pl.concat([adj, raw])

    # Compare like with like: only symbols every source covers
    n_sources = d["vendor"].n_unique()
    common = (d.group_by("symbol").agg(pl.col("vendor").n_unique().alias("n"))
               .filter(pl.col("n") == n_sources)["symbol"].to_list())
    excluded = sorted(set(d["symbol"].unique().to_list()) - set(common))
    print(f"Universe: {len(common)} symbols in every source; excluded: {excluded or 'none'}")
    return d.filter(pl.col("symbol").is_in(common)).sort("vendor", "symbol", "date")


def monthly(d):
    m = (d.with_columns(pl.col("date").dt.truncate("1mo").alias("month"))
           .group_by("vendor", "symbol", "month")
           .agg(pl.col("px").sort_by("date").last())
           .sort("vendor", "symbol", "month"))
    return m.with_columns(
        (pl.col("px") / pl.col("px").shift(1) - 1).over("vendor", "symbol").alias("ret_m"),
        # 12-1 momentum: return from 12 months ago to 1 month ago
        (pl.col("px").shift(1) / pl.col("px").shift(12) - 1).over("vendor", "symbol").alias("mom_12_1"))


def dividend_yield():
    """Point-in-time trailing-12-month dividend yield from Tiingo, at each month-end."""
    t = (pl.read_parquet(BARS)
           .filter((pl.col("vendor") == "tiingo") & (pl.col("date") >= START))
           .sort("symbol", "date")
           .with_columns((pl.col("dividend").rolling_sum(252).over("symbol")
                          / pl.col("close")).alias("yld"),
                         pl.col("date").dt.truncate("1mo").alias("month")))
    return t.group_by("symbol", "month").agg(pl.col("yld").sort_by("date").last())


def long_short(m, sig):
    """Each month: long top-K, short bottom-K by last month's signal (no look-ahead)."""
    x = (m.sort("vendor", "symbol", "month")
          .with_columns(pl.col(sig).shift(1).over("vendor", "symbol").alias("s"))
          .filter(pl.col("s").is_not_null() & pl.col("ret_m").is_not_null())
          .with_columns(
              pl.col("s").rank("ordinal", descending=True).over("vendor", "month").alias("r_hi"),
              pl.col("s").rank("ordinal").over("vendor", "month").alias("r_lo")))
    return x.group_by("vendor", "month").agg(
        (pl.col("ret_m").filter(pl.col("r_hi") <= K).mean()
         - pl.col("ret_m").filter(pl.col("r_lo") <= K).mean()).alias("ls"),
        pl.col("symbol").filter(pl.col("r_hi") <= K).sort().str.join(",").alias("longs"),
        pl.col("symbol").filter(pl.col("r_lo") <= K).sort().str.join(",").alias("shorts"))


def summarize(ls, title):
    ls = ls.with_columns(
        pl.when(pl.col("month") < pl.lit(BREAK)).then(pl.lit("2015-2022"))
          .otherwise(pl.lit("2023-2026")).alias("period"))
    out = (ls.group_by("vendor", "period").agg(
              (pl.col("ls").mean() * 1200).round(2).alias("ann_ret_pct"),
              (pl.col("ls").std() * 12 ** 0.5 * 100).round(2).alias("ann_vol_pct"),
              (pl.col("ls").mean() / pl.col("ls").std() * 12 ** 0.5).round(2).alias("sharpe"),
              pl.len().alias("months"))
             .sort("period", "vendor"))
    print(f"\n{title}")
    with pl.Config(tbl_rows=20):
        print(out)


def agreement(ls, ref="tiingo"):
    r = ls.filter(pl.col("vendor") == ref).select(
        "month", pl.col("longs").alias("ref_longs"), pl.col("shorts").alias("ref_shorts"))
    out = (ls.filter(pl.col("vendor") != ref).join(r, on="month")
             .group_by("vendor").agg(
                 (pl.col("longs") == pl.col("ref_longs")).mean().round(3).alias("same_longs"),
                 (pl.col("shorts") == pl.col("ref_shorts")).mean().round(3).alias("same_shorts"))
             .sort("vendor"))
    print("Share of months with identical holdings to Tiingo:")
    print(out)


def total_return_since(d, since):
    x = (d.filter(pl.col("date") >= since)
          .group_by("vendor", "symbol")
          .agg((pl.col("px").sort_by("date").last()
                / pl.col("px").sort_by("date").first() - 1).alias("tr")))
    p = x.pivot(on="vendor", index="symbol", values="tr")
    missed = (pl.read_parquet(ISSUES)
                .filter((pl.col("check") == "missed_dividend_adjustment")
                        & (pl.col("vendor") == "stooq") & (pl.col("date") >= since))
                .group_by("symbol").len().rename({"len": "missed_divs"}))
    p = (p.join(missed, on="symbol", how="left")
          .with_columns(pl.col("missed_divs").fill_null(0),
                        (pl.col("tiingo") * 100).round(1).alias("tiingo_tr_pct"),
                        ((pl.col("stooq") - pl.col("tiingo")) * 100).round(2).alias("stooq_minus_tiingo_pp"),
                        ((pl.col("yfinance") - pl.col("tiingo")) * 100).round(2).alias("yf_minus_tiingo_pp")))
    print(f"\nTotal return since {since}, by source (percentage points vs Tiingo)")
    with pl.Config(tbl_rows=25):
        print(p.select("symbol", "tiingo_tr_pct", "stooq_minus_tiingo_pp",
                       "yf_minus_tiingo_pp", "missed_divs").sort("stooq_minus_tiingo_pp"))
    print(p.select(pl.corr("stooq_minus_tiingo_pp", "missed_divs").round(3)
                   .alias("corr(stooq gap, missed dividends)")))


def risk(d):
    x = d.with_columns(
        (pl.col("px") / pl.col("px").shift(1) - 1).over("vendor", "symbol").alias("r"),
        (pl.col("px") / pl.col("px").cum_max().over("vendor", "symbol") - 1).alias("dd"))
    agg = x.group_by("vendor", "symbol").agg(
        (pl.col("r").std() * 252 ** 0.5 * 100).round(1).alias("vol_pct"),
        (pl.col("dd").min() * 100).round(1).alias("max_drawdown_pct"),
        (pl.col("r").min() * 100).round(1).alias("worst_day_pct"))
    for metric in ["max_drawdown_pct", "worst_day_pct", "vol_pct"]:
        print(f"\n{metric} by source")
        with pl.Config(tbl_rows=25):
            print(agg.pivot(on="vendor", index="symbol", values=metric).sort("symbol"))


def run():
    d = load_daily()

    # 1. Direct cost of Stooq's missed dividend adjustments
    total_return_since(d, BREAK)

    m = monthly(d)

    # 2. Dividend-yield factor: same Tiingo signal for every source, only returns differ
    dy = m.join(dividend_yield(), on=["symbol", "month"], how="left")
    summarize(long_short(dy, "yld"), "Dividend-yield long/short (signal fixed, returns by source)")

    # 3. Momentum: each source computes its own signal from its own prices
    mom = long_short(m, "mom_12_1")
    summarize(mom, "12-1 momentum long/short (signal and returns by source)")
    agreement(mom)

    # 4. Risk statistics
    risk(d)


if __name__ == "__main__":
    run()