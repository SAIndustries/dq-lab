from pathlib import Path
import polars as pl
import yaml

RAW, OUT = Path("data/raw"), Path("data/normalized")
SYMBOL_MAP = yaml.safe_load(open("config/universe.yaml")).get("symbol_map", {})

SCHEMA = {
    "vendor": pl.Utf8, "symbol": pl.Utf8, "date": pl.Date,
    "open": pl.Float64, "high": pl.Float64, "low": pl.Float64,
    "close": pl.Float64, "adj_close": pl.Float64, "volume": pl.Float64,
    "dividend": pl.Float64, "split_ratio": pl.Float64,
}

def _finish(df, vendor, symbol):
    df = df.with_columns(pl.lit(vendor).alias("vendor"),
                         pl.lit(SYMBOL_MAP.get(symbol, symbol)).alias("symbol"))
    for col, dtype in SCHEMA.items():
        if col not in df.columns:
            df = df.with_columns(pl.lit(None, dtype=dtype).alias(col))
    return df.select([pl.col(c).cast(t) for c, t in SCHEMA.items()]).sort("date")

def norm_yfinance(df):
    df = df.rename({"Date": "date", "Open": "open", "High": "high", "Low": "low",
                    "Close": "close", "Adj Close": "adj_close", "Volume": "volume",
                    "Dividends": "dividend", "Stock Splits": "split_ratio"})
    return df.with_columns(
        pl.col("date").dt.date(),
        # yfinance uses 0 for "no split"; Tiingo uses 1. Standardise on 1.
        pl.when(pl.col("split_ratio") == 0).then(1.0)
          .otherwise(pl.col("split_ratio")).alias("split_ratio"))

def norm_stooq(df):
    df = df.rename({c: c.lower() for c in df.columns})
    return df.with_columns(pl.col("date").str.to_date())

def norm_tiingo(df):
    df = df.rename({"adjClose": "adj_close", "divCash": "dividend",
                    "splitFactor": "split_ratio"})
    return df.with_columns(pl.col("date").str.slice(0, 10).str.to_date())

def norm_binance(df):
    return df.with_columns(
        pl.from_epoch(pl.col("open_time").cast(pl.Int64), time_unit="ms")
          .dt.date().alias("date"))

def norm_epoch_seconds(df):  # Coinbase and Kraken
    return df.with_columns(
        pl.from_epoch(pl.col("time").cast(pl.Int64), time_unit="s")
          .dt.date().alias("date"))

NORMALISERS = {
    "yfinance": norm_yfinance, "stooq": norm_stooq, "tiingo": norm_tiingo,
    "binance": norm_binance, "coinbase": norm_epoch_seconds,
    "kraken": norm_epoch_seconds,
}

def run():
    frames = []
    for vendor, fn in NORMALISERS.items():
        for path in sorted((RAW / vendor).glob("*.parquet")):
            try:
                frames.append(_finish(fn(pl.read_parquet(path)), vendor, path.stem))
            except Exception as e:
                print(f"FAIL {vendor}/{path.stem}: {e}")
    bars = pl.concat(frames)
    OUT.mkdir(parents=True, exist_ok=True)
    bars.write_parquet(OUT / "bars.parquet")
    print(bars.group_by("vendor").agg(pl.len().alias("rows"),
                                      pl.col("symbol").n_unique().alias("symbols")))

if __name__ == "__main__":
    run()