import datetime as dt
from pathlib import Path
import polars as pl
import yaml
import pandas_market_calendars as mcal

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load(open(ROOT / "config/universe.yaml"))
BARS = ROOT / "data/normalized/bars.parquet"
OUT = ROOT / "data/validation"
START = dt.date.fromisoformat(CFG["start"])
CRYPTO = ["binance", "coinbase", "kraken"]
KEYS = ["vendor", "symbol"]

# ---------- helpers ----------
def issues(df, check, severity, detail):
    return df.select(
        pl.lit(check).alias("check"), pl.lit(severity).alias("severity"),
        "vendor", "symbol", "date", detail.cast(pl.Utf8).alias("detail"))

def with_returns(b, col="close"):
    return b.sort(*KEYS, "date").with_columns(
        (pl.col(col) / pl.col(col).shift(1)).over(KEYS).log().alias("ret"))

def nyse_days(start, end):
    sched = mcal.get_calendar("XNYS").schedule(start_date=start, end_date=end)
    return pl.Series("date", list(sched.index.date), dtype=pl.Date)

# ---------- single-vendor checks ----------
def check_duplicates(b):
    d = b.filter(pl.struct("vendor", "symbol", "date").is_duplicated())
    return issues(d.unique(["vendor", "symbol", "date"]), "duplicate_row", "high",
                  pl.lit("duplicate (vendor, symbol, date)"))

def check_ohlc(b, eps=1e-6):
    bad = b.filter(
        (pl.col("high") < pl.max_horizontal("open", "close", "low") * (1 - eps)) |
        (pl.col("low") > pl.min_horizontal("open", "close", "high") * (1 + eps)) |
        (pl.min_horizontal("open", "high", "low", "close") <= 0))
    return issues(bad, "ohlc_inconsistent", "high",
                  pl.format("O={} H={} L={} C={}", "open", "high", "low", "close"))

def check_zero_volume(b):
    bad = b.filter(~pl.col("vendor").is_in(CRYPTO) & (pl.col("volume") <= 0))
    return issues(bad, "zero_volume", "medium", pl.format("volume={}", "volume"))

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

def check_calendar(b):
    missing, extra = [], []
    for (vendor, symbol), g in b.group_by(KEYS):
        first, last = max(g["date"].min(), START), g["date"].max()
        if vendor in CRYPTO:
            exp = pl.date_range(first, last, "1d", eager=True).alias("date")
        else:
            exp = nyse_days(first, last)
        exp_df = pl.DataFrame({"date": exp})
        have = g.select("date").filter(pl.col("date") >= first)
        tag = [pl.lit(vendor).alias("vendor"), pl.lit(symbol).alias("symbol")]
        missing.append(exp_df.join(have, on="date", how="anti").with_columns(tag))
        extra.append(have.join(exp_df, on="date", how="anti").with_columns(tag))
    return pl.concat([
        issues(pl.concat(missing), "missing_day", "medium",
               pl.lit("expected trading day absent")),
        issues(pl.concat(extra), "unexpected_day", "medium",
               pl.lit("row on a non-trading day")),
    ])

def check_return_outliers(b, z=8):
    b = b.with_columns(pl.coalesce("adj_close", "close").alias("px"))
    r = with_returns(b, "px").with_columns(
        (pl.col("ret") - pl.col("ret").median().over(KEYS)).abs().alias("dev"))
    r = r.with_columns(
        (pl.col("dev") / (1.4826 * pl.col("dev").median().over(KEYS))).alias("z"))
    bad = r.filter(pl.col("z") > z)
    return issues(bad, "return_outlier", "low",
                  pl.format("log_ret={} z={}", pl.col("ret").round(4), pl.col("z").round(1)))

# ---------- split / basis check ----------
def split_behaviour(b):
    events = (b.filter((pl.col("vendor") == "tiingo") & (pl.col("split_ratio") != 1))
               .select("symbol", "date", pl.col("split_ratio").alias("ratio")))
    s = (b.filter(~pl.col("vendor").is_in(CRYPTO)).sort(*KEYS, "date")
          .with_columns((pl.col("close").shift(1) / pl.col("close")).over(KEYS).alias("jump")))
    j = s.join(events, on=["symbol", "date"])
    return j.with_columns(
        pl.when((pl.col("jump") / pl.col("ratio") - 1).abs() < 0.25).then(pl.lit("raw"))
          .when((pl.col("jump") - 1).abs() < 0.25).then(pl.lit("adjusted"))
          .otherwise(pl.lit("unclear")).alias("observed")
    ).select("vendor", "symbol", "date", "ratio", pl.col("jump").round(3), "observed")

def check_split_basis(b):
    sb = split_behaviour(b)
    basis = CFG["price_basis"]
    bdf = pl.DataFrame({"vendor": list(basis),
                        "expected": ["raw" if p == "raw" else "adjusted" for p in basis.values()]})
    bad = sb.join(bdf, on="vendor").filter(pl.col("observed") != pl.col("expected"))
    return issues(bad, "split_basis_mismatch", "high",
                  pl.format("ratio={} jump={} observed={} declared={}",
                            "ratio", "jump", "observed", "expected")), sb

# ---------- cross-vendor checks ----------
def check_cross_vendor(b, tol=0.005):
    eq = b.filter(~pl.col("vendor").is_in(CRYPTO)).with_columns(
        pl.coalesce("adj_close", "close").alias("px"))
    r = with_returns(eq, "px").select("vendor", "symbol", "date", "ret").drop_nulls("ret")
    pairs = (r.join(r, on=["symbol", "date"], suffix="_b")
              .filter(pl.col("vendor") < pl.col("vendor_b")))
    bad = pairs.filter((pl.col("ret") - pl.col("ret_b")).abs() > tol)
    return issues(bad, "cross_vendor_return_gap", "medium",
                  pl.format("{} vs {}: {} vs {}", "vendor", "vendor_b",
                            pl.col("ret").round(4), pl.col("ret_b").round(4)))

def check_cross_exchange(b, tol=0.01):
    c = b.filter(pl.col("vendor").is_in(CRYPTO)).select("vendor", "symbol", "date", "close")
    pairs = (c.join(c, on=["symbol", "date"], suffix="_b")
              .filter(pl.col("vendor") < pl.col("vendor_b")))
    bad = pairs.filter((pl.col("close") / pl.col("close_b") - 1).abs() > tol)
    return issues(bad, "cross_exchange_price_gap", "medium",
                  pl.format("{} vs {}: {} vs {}", "vendor", "vendor_b", "close", "close_b"))

def check_coverage(b):
    eq_vendors = [v for v in b["vendor"].unique().to_list() if v not in CRYPTO]
    have = b.filter(pl.col("vendor").is_in(eq_vendors)).select(KEYS).unique()
    want = pl.DataFrame({"vendor": eq_vendors}).join(
        pl.DataFrame({"symbol": CFG["equities"]}), how="cross")
    miss = want.join(have, on=KEYS, how="anti").with_columns(
        pl.lit(None, dtype=pl.Date).alias("date"))
    return issues(miss, "missing_symbol", "high", pl.lit("symbol absent from vendor"))

def check_dividend_amounts(b, ref="tiingo", tol=0.05):
    eq = b.filter(~pl.col("vendor").is_in(CRYPTO))
    ref_df = eq.filter((pl.col("vendor") == ref) & (pl.col("dividend") > 0)).select(
        "symbol", "date", pl.col("dividend").alias("div_ref"), pl.col("close").alias("close_ref"))
    m = (eq.filter((pl.col("vendor") != ref) & (pl.col("dividend") > 0))
           .join(ref_df, on=["symbol", "date"])
           .with_columns((pl.col("dividend") * pl.col("close_ref") / pl.col("close")
                          / pl.col("div_ref")).alias("div_ratio")))
    bad = m.filter((pl.col("div_ratio") - 1).abs() > tol)
    return issues(bad, "dividend_amount_mismatch", "high",
                  pl.format("vendor div={} ref div={} ratio={}", "dividend", "div_ref",
                            pl.col("div_ratio").round(2)))

def check_dividend_adjustment(b, ref="tiingo", tol=0.1):
    px = b.filter(~pl.col("vendor").is_in(CRYPTO)).with_columns(
        pl.coalesce("adj_close", "close").alias("px"))
    r = with_returns(px, "px").with_columns(
        pl.col("close").shift(1).over(KEYS).alias("prev_close"))
    ref_df = r.filter((pl.col("vendor") == ref) & (pl.col("dividend") > 0)).select(
        "symbol", "date", pl.col("ret").alias("ret_ref"),
        (pl.col("dividend") / pl.col("prev_close")).alias("div_yield"))
    m = (r.filter(pl.col("vendor") != ref).select("vendor", "symbol", "date", "ret")
          .join(ref_df, on=["symbol", "date"])
          .with_columns(((pl.col("ret") - pl.col("ret_ref")) / pl.col("div_yield")).alias("ratio")))
    detail = pl.format("gap/yield={} yield={}", pl.col("ratio").round(2), pl.col("div_yield").round(4))
    return pl.concat([
        issues(m.filter((pl.col("ratio") + 1).abs() < tol),
               "missed_dividend_adjustment", "high", detail),
        issues(m.filter((pl.col("ratio") - 1).abs() < tol),
               "double_dividend_adjustment", "high", detail),
    ])

# ---------- runner ----------
def run():
    b = pl.read_parquet(BARS).filter(pl.col("date") >= START)
    split_issues, split_table = check_split_basis(b)
    all_issues = pl.concat([
        check_duplicates(b), check_ohlc(b), check_zero_volume(b), check_stale(b),
        check_calendar(b), check_return_outliers(b), split_issues,
        check_cross_vendor(b), check_cross_exchange(b), check_coverage(b), check_dividend_amounts(b), check_dividend_adjustment(b),
    ])
    OUT.mkdir(parents=True, exist_ok=True)
    all_issues.write_parquet(OUT / "issues.parquet")
    split_table.write_csv(OUT / "split_behaviour.csv")
    with pl.Config(tbl_rows=60, fmt_str_lengths=80):
        print(all_issues.group_by("check", "vendor").agg(pl.len().alias("n"))
                        .sort("check", "vendor"))
        print(split_table.sort("symbol", "date", "vendor"))



if __name__ == "__main__":
    run()