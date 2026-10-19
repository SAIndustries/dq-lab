from pathlib import Path
import pandas as pd
from .base import save_raw

ROOT = Path(__file__).resolve().parents[2]
MANUAL = ROOT / "data/manual/stooq"

def run():
    for f in sorted(MANUAL.glob("*_us_d.csv")):
        sym = f.name.split("_")[0].upper()          # nvda_us_d.csv -> NVDA
        df = pd.read_csv(f)
        if list(df.columns[:2]) != ["Date", "Open"]:
            print(f"SKIP {f.name}: unexpected columns {list(df.columns)}")
            continue
        save_raw(df, "stooq", sym, source=f"manual download: {f.name}")
        print(f"OK   {sym}: {len(df)} rows")

if __name__ == "__main__":
    run()