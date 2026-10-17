# dqlab/ingest/equities.py
import io, os
import pandas as pd
import requests
import yfinance as yf
from .base import save_raw, get_json

def fetch_yfinance(sym, start):
    # auto_adjust=False keeps raw Close AND Adj Close, plus splits/dividends
    df = yf.Ticker(sym).history(start=start, auto_adjust=False, actions=True)
    save_raw(df.reset_index(), "yfinance", sym, "yfinance")

def fetch_stooq(sym, start):
    r = requests.get("https://stooq.com/q/d/l/",
                     params={"s": f"{sym.lower()}.us", "i": "d"}, timeout=30)
    df = pd.read_csv(io.StringIO(r.text))
    save_raw(df, "stooq", sym, r.url)

def fetch_tiingo(sym, start):
    url = f"https://api.tiingo.com/tiingo/daily/{sym}/prices"
    data = get_json(url, {"startDate": start, "token": os.environ["TIINGO_KEY"]})
    save_raw(pd.DataFrame(data), "tiingo", sym, url)