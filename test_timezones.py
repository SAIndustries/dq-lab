import polars as pl

NY = "America/New_York"

f = pl.read_parquet("data/normalized/filings.parquet").with_columns(
    pl.col("acceptance_raw").str.slice(0, 19)
      .str.to_datetime("%Y-%m-%dT%H:%M:%S").alias("raw"))

def utc_to_ny(col):
    return (pl.col(col).dt.replace_time_zone("UTC")
              .dt.convert_time_zone(NY).dt.replace_time_zone(None))

h = f.with_columns(
    utc_to_ny("raw").alias("H1_true_utc"),      # raw is genuine UTC
    pl.col("raw").alias("H2_et_mislabelled"),   # raw is ET with a misleading Z
)
h = h.with_columns(utc_to_ny("H1_true_utc").alias("H3_double_offset"))  # offset applied twice

def predicted_filing_date(col):
    t = pl.col(col)
    mins = t.dt.hour().cast(pl.Int32) * 60 + t.dt.minute().cast(pl.Int32)
    late = mins > 17 * 60 + 30
    d = pl.when(late).then(t.dt.date() + pl.duration(days=1)).otherwise(t.dt.date())
    wd = d.dt.weekday()  # Mon=1 ... Sun=7
    return (pl.when(wd == 6).then(d + pl.duration(days=2))
              .when(wd == 7).then(d + pl.duration(days=1))
              .otherwise(d))

HYPS = ["H1_true_utc", "H2_et_mislabelled", "H3_double_offset"]

# 1. Overall: which hypothesis reproduces EDGAR's official filing dates?
print(h.select([(predicted_filing_date(c) == pl.col("filing_date")).mean().round(3).alias(c)
                for c in HYPS]))

# 2. Per company: is the data one population or two?
with pl.Config(tbl_rows=25):
    print(h.group_by("symbol").agg(
        [(predicted_filing_date(c) == pl.col("filing_date")).mean().round(2).alias(c) for c in HYPS]
    ).sort("symbol"))

# 3. Under each hypothesis, what ET hours do filings land in?
with pl.Config(tbl_rows=30):
    print(h.select([pl.col(c).dt.hour().alias(c) for c in HYPS])
           .unpivot(variable_name="hyp", value_name="et_hour")
           .group_by("hyp", "et_hour").len()
           .sort("hyp", "et_hour"))

# 4. Per company: share of timestamps inside EDGAR's 06:00-22:00 ET window
def plausible(c):
    return pl.col(c).dt.hour().cast(pl.Int32).is_between(6, 21).mean().round(2)

with pl.Config(tbl_rows=25):
    print(h.group_by("symbol").agg([plausible(c).alias(c) for c in HYPS]).sort("symbol"))

# 5. Per company: most common ET hour of earnings (Item 2.02) filings
earn = h.filter(pl.col("items").str.contains("2.02", literal=True))
with pl.Config(tbl_rows=25):
    print(earn.group_by("symbol").agg(
        [pl.col(c).dt.hour().mode().first().alias(c) for c in ["H1_true_utc", "H3_double_offset"]]
    ).sort("symbol"))

# 6. A KO earnings filing for a second manual check
print(earn.filter(pl.col("symbol") == "KO")
          .select("filing_date", "acceptance_raw", "accession").tail(2))

# 7. Does the convention follow the submitter, not the company?
s = h.with_columns(pl.col("accession").str.slice(0, 10).alias("submitter")).with_columns(
    (pl.col("submitter") == pl.col("cik")).alias("self_filed"))

print(s.group_by("self_filed").agg(
    pl.len().alias("n"),
    plausible("H1_true_utc").alias("H1_plausible"),
    plausible("H3_double_offset").alias("H3_plausible")))

with pl.Config(tbl_rows=20, fmt_str_lengths=60):
    print(s.group_by("submitter").agg(
        pl.len().alias("n"),
        plausible("H1_true_utc").alias("H1"),
        plausible("H3_double_offset").alias("H3"),
        pl.col("symbol").unique().sort().str.join(",").alias("symbols"))
     .sort("n", descending=True).head(15))