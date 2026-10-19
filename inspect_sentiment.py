import polars as pl

s = pl.read_parquet("data/features/sentiment.parquet")
w = pl.read_csv("data/features/lm_word_counts.csv")
ws = pl.read_csv("data/features/lm_word_counts_by_symbol.csv")
t = pl.read_parquet("data/normalized/earnings_text.parquet").select("accession", "text")

# 1. Effect of each cleaning step (all variants now smoothed)
V = ["tone_full", "tone_body", "tone_prose", "tone_clean_v1", "tone_clean", "tone"]
print(pl.DataFrame({
    "variant": V,
    "mean": [s[c].mean() for c in V],
    "std": [s[c].std() for c in V],
    "corr_with_final": [s.select(pl.corr(c, "tone")).item() for c in V],
}).with_columns(pl.col(pl.Float64).round(3)))

# 2. Coverage per company
with pl.Config(tbl_rows=25):
    print(s.group_by("symbol").agg(
        pl.col("tone").mean().round(3).alias("mean_tone"),
        pl.col("tone").std().round(3).alias("std_tone"),
        pl.col("low_evidence").sum().alias("n_low"),
        pl.col("n_hits").median().alias("median_hits"),
        pl.col("scoreable").first(),
        pl.len()).sort("mean_tone"))

# 3. Face validity: quarterly average tone (scoreable companies only)
sc = s.filter(pl.col("scoreable"))
with pl.Config(tbl_rows=50):
    print(sc.group_by(pl.col("filing_date").dt.year().alias("year"),
                      pl.col("filing_date").dt.quarter().alias("q"))
            .agg(pl.col("tone").mean().round(3).alias("avg_tone"), pl.len())
            .sort("year", "q"))

# 4. Extremes: should no longer be saturated at exactly +/-1
x = sc.join(t, on="accession").with_columns(pl.col("text").str.slice(30, 100).alias("headline"))
with pl.Config(fmt_str_lengths=100):
    print(x.sort("tone").select("symbol", "filing_date", pl.col("tone").round(3),
                                "n_hits", "headline").head(6))
    print(x.sort("tone", descending=True).select("symbol", "filing_date", pl.col("tone").round(3),
                                                 "n_hits", "headline").head(6))

# 5. What drives Visa and PepsiCo negatives?
with pl.Config(tbl_rows=10):
    for sym in ["V", "PEP"]:
        print(sym, "negative words")
        print(ws.filter((pl.col("symbol") == sym) & (pl.col("side") == "negative"))
                .select("word", "count").head(10))

# 6. Seasonality: Visa tone by calendar quarter of filing
print(s.filter(pl.col("symbol") == "V")
       .group_by(pl.col("filing_date").dt.quarter().alias("filing_q"))
       .agg(pl.col("tone").mean().round(3).alias("mean_tone"), pl.len())
       .sort("filing_q"))

# 7. Surprise coverage (scoreable companies; expect 4 per company without history)
print(sc.select(pl.col("tone_surprise").is_null().sum().alias("no_history"),
                pl.col("symbol").n_unique().alias("companies"), pl.len()))