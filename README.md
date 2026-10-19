# dq-lab: Market Data Quality Lab

> **Status: work in progress.** Ingestion, normalisation, validation and SEC filing timestamps are complete. Text features, the look-ahead bias experiment and the published quality report are in progress.

A pipeline that pulls the same market data from several vendors, normalises it into one schema, and systematically tests whether it can be trusted. The goal isn't another trading model. It's to answer the question every model depends on: **is the data right, and how would I know if it weren't?**

---

## Why this project

Research results are only as good as the data underneath them. Vendors disagree about what "close" means, apply corporate actions inconsistently, fail silently, and label timestamps in ways that don't match reality. Each of these can quietly invalidate a backtest.

This project treats data quality as a research problem:

- **Ingest raw vendor data unmodified**, with full lineage.
- **Normalise the format, not the values.**
- **Validate with explicit, tested checks.**
- **Investigate every discrepancy until it's explained**, including rejecting hypotheses that don't hold up.

---

## Pipeline

```
 Ingest (raw, with lineage)  →  Normalise (one schema)  →  Validate (tested checks)  →  Investigate & report
 yfinance · Tiingo · Stooq       Polars, canonical          11+ checks, synthetic        quantified findings
 Binance · Coinbase · Kraken     OHLCV + corporate          error-injection tests
 SEC EDGAR (8-K filings)         actions; verified
                                 filing timestamps
```

| Stage | What it does |
|---|---|
| **Ingest** | Fetches daily OHLCV for 20 US equities (2015 onward) from three vendors and daily candles for BTC/ETH from three exchanges. Every fetch is logged to a manifest with row count, SHA-256 hash and UTC timestamp. Empty or non-data responses are rejected at ingestion. |
| **Normalise** | Maps every vendor into one canonical schema (dates, OHLC, adjusted close, volume, dividends, split ratio) without changing values. Vendor quirks (millisecond vs second timestamps, column order, string-typed prices, split-ratio conventions) are handled explicitly. |
| **Validate** | Duplicate rows, OHLC consistency, zero volume, stale OHLCV runs, NYSE/crypto calendar gaps, robust return outliers, split-day price-basis verification, cross-vendor return reconciliation, cross-exchange price gaps, symbol coverage, and targeted dividend-adjustment and dividend-amount checks. |
| **SEC filings** | Ingests every 8-K for the universe from SEC EDGAR, including paged historical submissions, and assigns each filing a verified acceptance timestamp from its filing header. |

---

## Universe

**Equities (20):** chosen to stress-test corporate-action handling.

- **Recent stock splits:** NVDA, AAPL, TSLA, AMZN, GOOGL, WMT, CMG, AVGO
- **Ticker change:** META (formerly FB)
- **Control group:** MSFT, JPM, XOM, JNJ, PG, KO, V, HD, UNH, PEP, COST

**Crypto:** BTC and ETH on Binance (USDT), Coinbase (USD) and Kraken (USD).

---

## Key findings so far

1. **Three vendors, three meanings of "close".** Measured on 11 split events: Tiingo's `close` is raw, yfinance's is split-adjusted (even with `auto_adjust=False`), and Stooq's is split- and dividend-adjusted.

2. **Stooq's dividend adjustments broke down from 2023 onward.** Every 2015–2022 ex-dividend date is adjusted correctly. The miss rate then rises to 46% (2023), 56% (2024), 88% (2025) and 91% (2026, to date), with 152 missed adjustments in total. On those dates Stooq's return differs from the reference by almost exactly the dividend yield (median gap/yield ≈ −1.00).

3. **Generic tolerance checks under-detect systematic errors.** A 50 bp cross-vendor return check caught only 91 of the 152 missed adjustments, because low-yield stocks stay under the threshold. A targeted check found the rest.

4. **yfinance records two AVGO dividends (March and June 2018) at double their correct value**: $0.35 split-adjusted instead of $0.175. AVGO's adjusted history before June 2018 sits 1.38% below the reference, exactly the sum of the two excess adjustments.

5. **A corrupted Stooq bar on 2015-03-12.** Seven symbols disagree with the other vendors on the same date. The two-day return matches, so it's a single bad close that reverts the next day.

6. **Stooq scales volume by its dividend adjustment factor**, inflating historical share volume (about 12% for AAPL in March 2015) where the usual convention adjusts volume for splits only.

7. **yfinance and Tiingo reconcile almost perfectly**: only 4 adjusted-return gaps above 50 bp across about 59,000 bars.

8. **SEC's submissions API mislabels acceptance times for most filers.** `data.sec.gov` reports `acceptanceDateTime` with a UTC "Z" suffix. For 13 of 20 companies (about two-thirds of 3,202 8-Ks) the value is actually Eastern Time with the UTC offset applied twice (+8h in summer, +10h in winter relative to ET). The rest are genuine UTC. Verified against the authoritative `ACCEPTANCE-DATETIME` in every filing header: each CIK follows exactly one convention, with zero unexplained filings. ExxonMobil's old and new registrant CIKs follow *different* conventions. A tested "filing agent" explanation was rejected.

9. **Ticker-to-entity mappings change over time.** The SEC's current ticker map points XOM to a new registrant with 5 filings. The 170 historical 8-Ks live under the original CIK.

10. **Ingestion failures can look like success.** A Stooq API change caused HTML error pages to be saved as data for every ticker. This was caught at normalisation, and a non-empty-response guard now blocks it at ingestion.

### Validating the validation

Several checks initially produced wrong answers, and fixing them was part of the work:

- **Return outliers on raw prices** flagged stock splits as crashes. The check now uses adjusted prices.
- **A close-only stale-price check** flagged a genuine AMZN coincidence (three identical closes in September 2023 with different open, high, low and volume). The check now requires all OHLCV fields to repeat.
- **An integer overflow** (`dt.hour()` returns an 8-bit integer in Polars, so `hour * 60` wrapped around) produced plausible but completely wrong timezone results until values were cast to `Int32`.
- **yfinance stores prices at float32 precision** (e.g. `125.9800033569336`), so exact equality comparisons on its prices are unreliable.

---

## Project structure

```
dq-lab/
├── config/
│   └── universe.yaml          # tickers, start date, price bases, CIK overrides
├── dqlab/
│   ├── ingest/
│   │   ├── base.py            # HTTP with retries, raw storage, manifest
│   │   ├── equities.py        # yfinance, Stooq, Tiingo adapters
│   │   ├── crypto.py          # Binance, Coinbase, Kraken adapters
│   │   ├── stooq_manual.py    # loads manually downloaded Stooq CSVs
│   │   ├── edgar.py           # SEC submissions (all pages)
│   │   └── edgar_headers.py   # per-filing ACCEPTANCE-DATETIME
│   ├── normalize.py           # canonical OHLCV schema
│   ├── edgar_normalize.py     # 8-K table with verified timestamps
│   └── validate.py            # validation suite
├── tests/                     # unit tests with synthetic injected errors
├── explore.py                 # investigation queries behind the findings
├── run_ingest.py
├── run_edgar.py
└── requirements.txt
```

Raw and processed data live in `data/` and are not committed.

---

## Quickstart

Requires Python 3.10+.

```bash
git clone https://github.com/SAIndustries/dq-lab.git
cd dq-lab
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file:

```
TIINGO_KEY=your_tiingo_key
SEC_USER_AGENT="Your Name your.email@example.com"
```

Stooq now requires an API key for automated downloads, so its CSVs are downloaded manually from `https://stooq.com/q/d/?s=<ticker>.us` into `data/manual/stooq/`.

Run the pipeline:

```bash
python run_ingest.py                     # equities and crypto
python -m dqlab.ingest.stooq_manual      # Stooq CSVs into the raw layer
python -m dqlab.normalize                # canonical schema
python -m dqlab.validate                 # validation suite
python run_edgar.py                      # SEC 8-K filings
python -m dqlab.ingest.edgar_headers     # authoritative acceptance times
python -m dqlab.edgar_normalize          # verified filing table
pytest                                   # tests
```

---

## Roadmap

- [x] Multi-vendor ingestion with lineage
- [x] Canonical normalisation
- [x] Validation suite with tests
- [x] Investigation of every major discrepancy
- [x] SEC 8-K ingestion with verified acceptance timestamps
- [ ] Earnings-release text features (Loughran–McDonald sentiment)
- [ ] Look-ahead bias experiment: filing date vs raw API time vs true acceptance time
- [ ] Downstream impact analysis: how data errors change research conclusions
- [ ] Published data quality report and dashboard

---

## Known limitations

- Stooq data for COST and PEP is pending, so the coverage check currently flags them.
- Tiingo is used as the reference vendor for dividend checks, justified by its near-perfect agreement with yfinance. It isn't treated as infallible.
- Crypto candles cover roughly the most recent 300–1,000 days, depending on each exchange's default history.
- The cause of SEC's two timestamp conventions is not known. The project documents the behaviour, not the mechanism.
