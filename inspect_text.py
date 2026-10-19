import polars as pl

t = pl.read_parquet("data/normalized/earnings_text.parquet")
f = pl.read_parquet("data/normalized/filings.parquet").select("accession", "filing_date")
t = t.join(f, on="accession")

# 1. What kind of document did we get for each filing?
print(t.group_by("doc_type").len().sort("len", descending=True))

# 2. Length per company: very short texts are suspicious
with pl.Config(tbl_rows=25):
    print(t.group_by("symbol").agg(
        pl.len().alias("n"),
        pl.col("n_words").min().alias("min_words"),
        pl.col("n_words").median().alias("median_words"),
        pl.col("n_words").max().alias("max_words")).sort("symbol"))

# 3. Does it read like an earnings release?
kw = t.with_columns(
    pl.col("text").str.to_lowercase().str.contains(r"quarter|fiscal|results").alias("earnings_like"))
print(kw.group_by("earnings_like").len())

# 4. Coverage: roughly 4 releases per company per year?
with pl.Config(tbl_rows=25):
    print(t.group_by("symbol").agg(
        pl.col("filing_date").dt.year().value_counts().alias("per_year"))
     .with_columns(pl.col("per_year").list.eval(pl.element().struct.field("count")).list.mean()
                   .round(2).alias("avg_per_year"))
     .select("symbol", "avg_per_year").sort("symbol"))

# 5. Shortest documents, for a manual look
with pl.Config(fmt_str_lengths=120):
    print(t.sort("n_words").select("symbol", "filing_date", "doc_type", "n_words",
                                   pl.col("text").str.slice(0, 120)).head(8))