import datetime as dt
import polars as pl
from dqlab import validate as v

def make_bars(closes, vendor="tiingo", symbol="TEST"):
    n, d0 = len(closes), dt.date(2024, 1, 2)
    return pl.DataFrame({
        "vendor": [vendor] * n, "symbol": [symbol] * n,
        "date": [d0 + dt.timedelta(days=i) for i in range(n)],
        "open": closes, "high": [c * 1.01 for c in closes],
        "low": [c * 0.99 for c in closes], "close": closes,
        "adj_close": [None] * n, "volume": [1e6] * n,
        "dividend": [0.0] * n, "split_ratio": [1.0] * n,
    }, schema_overrides={"adj_close": pl.Float64})

def test_stale_price_detected():
    b = make_bars([100.0, 101.0, 102.0, 102.0, 102.0, 102.0, 103.0])
    assert len(v.check_stale(b)) == 1

def test_clean_series_not_stale():
    assert len(v.check_stale(make_bars([100.0, 101.0, 102.0, 103.0]))) == 0

def test_ohlc_violation_detected():
    b = make_bars([100.0, 101.0, 102.0]).with_columns(
        pl.when(pl.col("close") == 101.0).then(90.0)
          .otherwise(pl.col("high")).alias("high"))
    assert len(v.check_ohlc(b)) == 1

def test_duplicate_detected():
    b = make_bars([100.0, 101.0])
    assert len(v.check_duplicates(pl.concat([b, b.head(1)]))) == 1