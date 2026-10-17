# dqlab/ingest/crypto.py
import pandas as pd
from .base import get_json, save_raw

def fetch_binance(pair):
    url = "https://api.binance.com/api/v3/klines"
    data = get_json(url, {"symbol": pair, "interval": "1d", "limit": 1000})
    cols = ["open_time", "open", "high", "low", "close", "volume", "close_time",
            "quote_volume", "trades", "taker_base", "taker_quote", "ignore"]
    save_raw(pd.DataFrame(data, columns=cols), "binance", pair, url)

def fetch_coinbase(pair):
    url = f"https://api.exchange.coinbase.com/products/{pair}/candles"
    data = get_json(url, {"granularity": 86400})
    cols = ["time", "low", "high", "open", "close", "volume"]  # note: LOW first!
    save_raw(pd.DataFrame(data, columns=cols), "coinbase", pair, url)

def fetch_kraken(pair):
    url = "https://api.kraken.com/0/public/OHLC"
    data = get_json(url, {"pair": pair, "interval": 1440})
    if data.get("error"):
        raise RuntimeError(data["error"])
    key = next(k for k in data["result"] if k != "last")  # XBTUSD -> "XXBTZUSD"
    cols = ["time", "open", "high", "low", "close", "vwap", "volume", "count"]
    save_raw(pd.DataFrame(data["result"][key], columns=cols), "kraken", pair, url)