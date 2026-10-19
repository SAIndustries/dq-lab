from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
NY = "America/New_York"
OUT = ROOT / "data/features/lookahead_events.parquet"
METHODS = ["filing_date", "raw_api", "true"]


def price_points():
    """Every tradable price: each trading day's open (09:30 ET) and close (16:00 ET),
    split- and dividend-adjusted, from Tiingo (the validated reference vendor)."""
    b = (pl.read_parquet(ROOT / "data/normalized/bars.parquet")
           .filter(pl.col("vendor") == "tiingo")
           .with_columns((pl.col("adj_close") / pl.col("close")).alias("adj")))
    opens = b.select("symbol", "date", pl.lit(0).alias("slot"),
                     (pl.col("open") * pl.col("adj")).alias("px"))
    closes = b.select("symbol", "date", pl.lit(1).alias("slot"),
                      pl.col("adj_close").alias("px"))
    pts = pl.concat([opens, closes]).sort("symbol", "date", "slot")
    pts = pts.with_columns(
        (pl.col("date").cast(pl.Datetime("us"))
         + pl.when(pl.col("slot") == 0).then(pl.duration(hours=9, minutes=30))
             .otherwise(pl.duration(hours=16)))
        .dt.replace_time_zone(NY).alias("ts"),
        # same slot one trading day later -> 1-day holding return
        pl.col("px").shift(-2).over("symbol").alias("px_next"))
    pts = pts.with_columns((pl.col("px_next") / pl.col("px") - 1).alias("ret"))
    # abnormal return: minus the universe average over the same interval
    return pts.with_columns(
        (pl.col("ret") - pl.col("ret").mean().over("date", "slot")).alias("abn"))


def load_events():
    s = pl.read_parquet(ROOT / "data/features/sentiment.parquet").filter(pl.col("scoreable"))
    f = pl.read_parquet(ROOT / "data/normalized/filings.parquet").select("accession", "acceptance_raw")
    raw_as_utc = (pl.col("acceptance_raw").str.slice(0, 19)
                    .str.to_datetime("%Y-%m-%dT%H:%M:%S")
                    .dt.replace_time_zone("UTC").dt.convert_time_zone(NY))
    e = s.join(f, on="accession").with_columns(
        pl.col("filing_date").cast(pl.Datetime("us")).dt.replace_time_zone(NY).alias("t_filing_date"),
        raw_as_utc.alias("t_raw_api"),
        pl.col("acceptance_et").alias("t_true"))

    # Timing bucket from the TRUE time (cast before multiplying: dt.hour() is Int8)
    mins = pl.col("t_true").dt.hour().cast(pl.Int32) * 60 + pl.col("t_true").dt.minute().cast(pl.Int32)
    return e.with_columns(
        pl.when(pl.col("t_true").dt.weekday() >= 6).then(pl.lit("weekend"))
          .when(mins < 9 * 60 + 30).then(pl.lit("pre_open"))
          .when(mins < 16 * 60).then(pl.lit("intraday"))
          .otherwise(pl.lit("after_close")).alias("timing"))


def attach(e, pts, col, name, strategy="forward"):
    """First price at/after (forward) or last price before (backward) the given time."""
    left = e.select("accession", "symbol", pl.col(col).alias("t")).sort("t")
    right = pts.select("symbol", pl.col("ts").alias("t"), "ts", "px", "ret", "abn").sort("t")
    j = left.join_asof(right, on="t", by="symbol", strategy=strategy, check_sortedness=False)
    return j.select("accession",
                    pl.col("ts").alias(f"entry_{name}"), pl.col("px").alias(f"px_{name}"),
                    pl.col("ret").alias(f"ret_{name}"), pl.col("abn").alias(f"abn_{name}"))


def stats(df, sig, ret):
    d = df.filter(pl.col(sig).is_not_null() & pl.col(ret).is_not_null())
    ls = pl.col(sig).sign() * pl.col(ret)
    return d.select(
        pl.len().alias("n"),
        (ls.mean() * 1e4).round(1).alias("mean_bp"),
        (ls.mean() / (ls.std() / pl.len().cast(pl.Float64).sqrt())).round(2).alias("t_stat"),
        (ls > 0).mean().round(3).alias("hit_rate"),
        pl.corr(sig, ret, method="spearman").round(3).alias("rank_ic"))


def run():
    pts = price_points()
    e = load_events()
    for m in METHODS:
        e = e.join(attach(e, pts, f"t_{m}", m), on="accession", how="left")

    # Reaction: last price strictly before the true time -> true entry price
    before = (e.select("accession", "symbol",
                       (pl.col("t_true") - pl.duration(microseconds=1)).alias("t_before")))
    b = attach(before, pts, "t_before", "before", strategy="backward")
    e = e.join(b.select("accession", "px_before"), on="accession", how="left").with_columns(
        (pl.col("px_true") / pl.col("px_before") - 1).alias("reaction"))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    e.write_parquet(OUT)

    # 1. Timing mix of the events
    print(e.group_by("timing").len().sort("timing"))

    # 2. How often does each method's entry differ from the true entry?
    for m in ["filing_date", "raw_api"]:
        print(f"{m} vs true entry:")
        print(e.select(
            (pl.col(f"entry_{m}") < pl.col("entry_true")).mean().round(3).alias("too_early"),
            (pl.col(f"entry_{m}") == pl.col("entry_true")).mean().round(3).alias("same"),
            (pl.col(f"entry_{m}") > pl.col("entry_true")).mean().round(3).alias("too_late")))

    # 3. Headline result: apparent predictability under each method
    for sig in ["tone_surprise", "tone_surprise_yoy"]:
        for kind in ["ret", "abn"]:
            print(f"\n{sig} | {'raw' if kind == 'ret' else 'abnormal'} 1-day returns")
            print(pl.concat([stats(e, sig, f"{kind}_{m}").with_columns(pl.lit(m).alias("method"))
                             for m in METHODS]).select("method", "n", "mean_bp", "t_stat",
                                                       "hit_rate", "rank_ic"))

    # 4. Where does the gap come from? By timing bucket (abnormal returns)
    print("\nBy release timing (tone_surprise, abnormal returns):")
    for m in ["filing_date", "true"]:
        print(m)
        print(pl.concat([stats(e.filter(pl.col("timing") == tb), "tone_surprise", f"abn_{m}")
                           .with_columns(pl.lit(tb).alias("timing"))
                         for tb in ["pre_open", "intraday", "after_close"]])
                .select("timing", "n", "mean_bp", "t_stat", "rank_ic"))

    # 5. Does tone explain the (untradable) reaction?
    print("\nTone surprise vs announcement reaction (not tradable):")
    print(e.filter(pl.col("tone_surprise").is_not_null() & pl.col("reaction").is_not_null())
           .select(pl.len().alias("n"),
                   pl.corr("tone_surprise", "reaction", method="spearman").round(3).alias("rank_ic"),
                   (pl.col("reaction").abs().mean() * 1e4).round(1).alias("mean_abs_reaction_bp")))


if __name__ == "__main__":
    run()