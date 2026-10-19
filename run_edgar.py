import time

import yaml
from dotenv import load_dotenv

load_dotenv()

from dqlab.ingest.edgar import cik_map, fetch_filings

cfg = yaml.safe_load(open("config/universe.yaml"))
ciks = cik_map()
overrides = cfg.get("cik_overrides", {})

for sym in cfg["equities"]:
    # Use the configured CIK list if present, otherwise the current SEC mapping.
    sym_ciks = overrides.get(sym, [ciks.get(sym)])
    for cik in sym_ciks:
        if cik is None:
            print("FAIL", sym, "no CIK found in SEC ticker map")
            continue
        try:
            name = fetch_filings(sym, cik)
            print("OK  ", sym, cik, name)
        except Exception as e:
            print("FAIL", sym, cik, e)
        time.sleep(0.5)