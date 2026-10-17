# run_ingest.py
import time, yaml
from dotenv import load_dotenv
from dqlab.ingest import equities, crypto

load_dotenv()
cfg = yaml.safe_load(open("config/universe.yaml"))

jobs = []
for sym in cfg["equities"]:
    for fn in (equities.fetch_yfinance, equities.fetch_stooq, equities.fetch_tiingo):
        jobs.append((fn, (sym, cfg["start"])))
for vendor, pairs in cfg["crypto"].items():
    fn = getattr(crypto, f"fetch_{vendor}")
    jobs += [(fn, (p,)) for p in pairs]

for fn, args in jobs:
    try:
        fn(*args)
        print(f"OK   {fn.__name__} {args[0]}")
    except Exception as e:
        print(f"FAIL {fn.__name__} {args[0]}: {e}")
    time.sleep(1)  # be polite to free APIs