# dq-lab: How Much Can You Trust Market Data?

**dq-lab** pulls the same stock, crypto and SEC filing data from several sources, checks it systematically, and measures what the errors do to real research results.

**[Read the full report and findings →](https://YOUR-SITE.netlify.app)**

---

## The idea

Every trading model, backtest and research paper rests on one assumption: that the data is right. This project tests that assumption.

It takes 20 large US stocks (2015–2026), downloads their prices from three vendors, adds three crypto exchanges and every SEC 8-K filing for those companies, and asks three questions:

1. **Where do the sources disagree, and why?**
2. **Which disagreements are real errors?**
3. **Do those errors actually change the conclusions a researcher would reach?**

The answer to the third question is the point of the project: some errors are harmless, and some quietly flip results.

---

## What we found

**Data vendors disagree about what a price even means.** Across 11 stock splits, one vendor's "close" was raw, one was split-adjusted, and one was split- *and* dividend-adjusted, all under the same column name.

**One vendor stopped adjusting for dividends, silently.** Stooq adjusted almost every dividend correctly until 2023, then missed 54% of them in 2023, 61% in 2024, 90% in 2025 and 92% in 2026 (184 in total). Nothing announced it. Only cross-checking caught it.

**The SEC's own API mislabels filing times.** For 13 of 20 companies, the submissions API reports acceptance times that are 4–5 hours off, with the timezone offset effectively applied twice. This was verified against the timestamp in every one of 3,289 filing headers, with zero filings left unexplained.

**Careless timestamps create fake trading profits.** Using a filing's date instead of its exact time lets a backtest "trade" before after-hours news was released, for 52% of earnings announcements. A controlled experiment showed this alone can turn a realistic signal into an apparent 82 bp per trade at a t-statistic of 5.3, when the correctly timed result is 1.3 bp at t = 0.2.

**Errors change real research conclusions, but not equally.**
- Treating raw prices as adjusted **flips the sign** of a dividend-yield strategy (−3.8% → +4.0% a year) and shows fake crashes of up to −99%.
- Stooq's missing dividends understate JPMorgan's 2023–2026 return by **24.6 percentage points** and bias a dividend strategy by 2.7 points a year.
- Isolated glitches, such as a single corrupted trading day, barely matter.

**The lesson:** data errors have a severity hierarchy, and validation effort should follow it.

Several of the project's own first hypotheses turned out to be wrong, and the report documents each one and how it was tested.

---

## How it works

```
Ingest  →  Normalise  →  Validate  →  Investigate  →  Measure impact
```

| Stage | What it does |
|---|---|
| **Ingest** | Raw data from yfinance, Tiingo, Stooq, Binance, Coinbase, Kraken and SEC EDGAR, stored unmodified. Every fetch is logged with a row count, SHA-256 hash and timestamp. |
| **Normalise** | One schema for all sources. Format is standardised; values are never altered. |
| **Validate** | 14 checks with unit tests: OHLC consistency, stale prices, trading-calendar gaps, return outliers, split-day price basis, cross-vendor reconciliation, and targeted dividend checks. |
| **SEC filings** | Every 8-K since 2015, including predecessor registrants (Google Inc., Avago, Broadcom Ltd, ExxonMobil). Timestamps come from filing headers; 944 earnings releases are classified and audited to exactly 4 per company per year. |
| **Text features** | Loughran–McDonald sentiment, with tables, legal boilerplate and accounting terms removed, scored point-in-time with no look-ahead. |
| **Experiments** | Look-ahead bias test with positive and negative controls; downstream impact of each vendor's errors on total returns, factor strategies, momentum and risk statistics. |

**Stack:** Python 3.12, Polars, pandas, pytest, BeautifulSoup, pandas-market-calendars.

---

## Run it yourself

```bash
git clone https://github.com/SAIndustries/dq-lab.git && cd dq-lab
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file:

```
TIINGO_KEY=your_tiingo_key
SEC_USER_AGENT="Your Name your.email@example.com"
```

Two manual downloads, both free:
- **Stooq CSVs** for each ticker from `https://stooq.com/q/d/?s=<ticker>.us`, saved to `data/manual/stooq/`
- **Loughran–McDonald Master Dictionary** from `https://sraf.nd.edu/loughranmcdonald-master-dictionary/`, saved to `data/manual/lm/`

Run the pipeline from the project root:

```bash
# Prices
python run_ingest.py
python -m dqlab.ingest.stooq_manual
python -m dqlab.normalize
python -m dqlab.validate

# SEC filings
python run_edgar.py
python -m dqlab.edgar_normalize
python -m dqlab.ingest.edgar_headers
python -m dqlab.edgar_normalize          # second pass, now with header timestamps
python -m dqlab.ingest.edgar_exhibits
python -m dqlab.text_extract
python -m dqlab.classify_releases

# Features and experiments
python -m dqlab.sentiment
python -m dqlab.lookahead
python -m dqlab.positive_control
python -m dqlab.impact

pytest
```

---

## Project structure

```
dq-lab/
├── config/universe.yaml      # tickers, price bases, CIK and release overrides
├── dqlab/
│   ├── ingest/               # vendor and SEC adapters
│   ├── normalize.py          # canonical price schema
│   ├── validate.py           # validation suite
│   ├── edgar_normalize.py    # verified filing timestamps
│   ├── classify_releases.py  # earnings / deliveries / preannouncements
│   ├── sentiment.py          # cleaned, point-in-time tone
│   ├── lookahead.py          # timestamp alignment experiment
│   ├── positive_control.py   # controlled test of look-ahead bias
│   └── impact.py             # effect of vendor errors on research results
├── tests/
└── requirements.txt
```

Data files are not committed. Every result can be regenerated with the commands above.

---

## Limitations

- Tiingo serves as the reference vendor, justified by its near-perfect agreement with yfinance, but it isn't treated as infallible.
- Costco is excluded from the sentiment analysis, because its press releases are almost entirely numeric.
- The cause of the SEC's two timestamp conventions is unknown. The project documents the behaviour, not the mechanism.
- Results cover 20 large US stocks and may not generalise to smaller or less liquid companies.

---

**Author:** Vishal Sujatha Achimuthu · [LinkedIn](https://www.linkedin.com/in/saindustries/) · [GitHub](https://github.com/SAIndustries)