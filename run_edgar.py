import time, yaml
from dotenv import load_dotenv
load_dotenv()
from dqlab.ingest.edgar import cik_map, fetch_filings

cfg = yaml.safe_load(open("config/universe.yaml"))
ciks = cik_map()
for sym in cfg["equities"]:
    try:
        fetch_filings(sym, ciks[sym])
        print("OK  ", sym, ciks[sym])
    except Exception as e:
        print("FAIL", sym, e)
    time.sleep(0.5)