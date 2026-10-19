import re
from collections import Counter, defaultdict
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
LM_DIR = ROOT / "data/manual/lm"
T = ROOT / "data/normalized/earnings_text.parquet"
F = ROOT / "data/normalized/filings.parquet"
OUT_DIR = ROOT / "data/features"

TOKEN = re.compile(r"[A-Za-z]+")
NEGATORS = {"NO", "NOT", "NONE", "NEITHER", "NEVER", "NOBODY"}
SAFE_HARBOR = re.compile(r"(?i)forward[- ]looking statements|safe harbor")
SPLIT = re.compile(r"(?<=[.!?;])\s+")
BOILER = re.compile(r"(?i)forward[- ]looking|safe harbor|undue reliance|differ materially|risk factors"
                    r"|(duty|obligation) to (update|revise)|private securities litigation reform act")

# Accounting line items that appear in reconciliation tables and non-GAAP notes
# every quarter regardless of performance (identified from LM word counts)
ACCOUNTING = frozenset({"LOSS", "LOSSES", "GAIN", "GAINS", "RESTRUCTURING", "IMPAIRMENT",
                        "IMPAIRMENTS", "DIVESTITURE", "DIVESTITURES", "DISCONTINUED"})
K_SMOOTH = 5                 # tone = (p - n) / (p + n + K): sparse texts shrink toward 0
LOW_EVIDENCE_HITS = 5        # fewer sentiment words than this -> flagged low_evidence
MAX_LOW_EVIDENCE_SHARE = 0.5 # companies above this are flagged unscoreable


def load_lm():
    files = sorted(LM_DIR.glob("*MasterDictionary*.csv"))
    if not files:
        raise FileNotFoundError(f"Put the LM Master Dictionary CSV in {LM_DIR}")
    d = pl.read_csv(files[-1], infer_schema_length=0)
    cols = {c.lower(): c for c in d.columns}

    def words(cat):
        c = cols[cat]
        return set(d.filter(pl.col(c).cast(pl.Int64, strict=False) > 0)[cols["word"]]
                    .str.to_uppercase().to_list())

    return words("positive"), words("negative"), words("uncertainty")


def body(text):
    """Old method: text before the first safe-harbor mention after the first 20%."""
    m = SAFE_HARBOR.search(text, len(text) // 5)
    return text[:m.start()] if m else text


def is_prose(seg):
    """A sentence-like segment, not a table row: at least 6 words, at most 10% digits."""
    letters = sum(c.isalpha() for c in seg)
    digits = sum(c.isdigit() for c in seg)
    return len(TOKEN.findall(seg)) >= 6 and digits <= 0.1 * (letters + digits)


def variants(text):
    segs = SPLIT.split(text)
    prose = [s for s in segs if is_prose(s)]

    # Boilerplate section: from the first boilerplate sentence past the opening 20%
    # of prose, everything onward (catches risk lists split on semicolons)
    start = len(prose) // 5
    cut = next((i for i, s in enumerate(prose) if i >= start and BOILER.search(s)), len(prose))
    clean = [s for s in prose[:cut] if not BOILER.search(s)]
    tail = prose[cut:]

    return {"full": text, "body": body(text),
            "prose": " ".join(prose),
            "clean_v1": " ".join(s for s in prose if not BOILER.search(s)),
            "clean": " ".join(clean), "tail": " ".join(tail)}


def score(text, pos, neg, unc, exclude=frozenset(), pos_ctr=None, neg_ctr=None):
    toks = [t.upper() for t in TOKEN.findall(text)]
    p = n = u = 0
    for i, t in enumerate(toks):
        if t in exclude:
            continue
        if t in pos:
            if NEGATORS.isdisjoint(toks[max(0, i - 3):i]):   # skip negated positives
                p += 1
                if pos_ctr is not None:
                    pos_ctr[t] += 1
        elif t in neg:
            n += 1
            if neg_ctr is not None:
                neg_ctr[t] += 1
        if t in unc:
            u += 1
    return len(toks), p, n, u


def tone(p, n):
    """Smoothed tone: near (p-n)/(p+n) for long texts, shrinks toward 0 for sparse ones."""
    return (p - n) / (p + n + K_SMOOTH)


def counts_frame(ctr, side, symbol=None):
    d = {"word": list(ctr), "count": list(ctr.values()), "side": side}
    schema = {"word": pl.Utf8, "count": pl.Int64, "side": pl.Utf8}
    if symbol is not None:
        d["symbol"] = symbol
        schema["symbol"] = pl.Utf8
    return pl.DataFrame(d, schema=schema)


def run():
    pos, neg, unc = load_lm()
    print(f"LM dictionary: {len(pos)} positive, {len(neg)} negative, {len(unc)} uncertainty words")

    t = pl.read_parquet(T).filter(pl.col("release_kind") == "earnings")
    pos_ctr, neg_ctr, tail_pos, tail_neg = Counter(), Counter(), Counter(), Counter()
    sym_pos, sym_neg = defaultdict(Counter), defaultdict(Counter)
    rows = []
    for acc, sym, text in t.select("accession", "symbol", "text").iter_rows():
        v = variants(text)
        row = {"accession": acc, "symbol": sym}
        for name in ("full", "body", "prose", "clean_v1", "clean"):
            _, p, n, _ = score(v[name], pos, neg, unc)
            row[f"tone_{name}"] = tone(p, n)

        # Diagnostic: what did the boilerplate cut remove?
        score(v["tail"], pos, neg, unc, ACCOUNTING, tail_pos, tail_neg)

        # Final score: clean prose before the boilerplate section, no accounting terms
        rp, rn = Counter(), Counter()
        nt, p, n, u = score(v["clean"], pos, neg, unc, ACCOUNTING, rp, rn)
        pos_ctr.update(rp); neg_ctr.update(rn)
        sym_pos[sym].update(rp); sym_neg[sym].update(rn)

        prose_len = len(v["prose"])
        row.update({"n_tokens": nt, "pos": p, "neg": n, "unc": u, "n_hits": p + n,
                    "low_evidence": (p + n) < LOW_EVIDENCE_HITS,
                    "pos_rate": p / nt if nt else None,
                    "neg_rate": n / nt if nt else None,
                    "unc_rate": u / nt if nt else None,
                    "tone": tone(p, n),
                    "prose_share": len(v["clean"]) / len(text) if text else None,
                    "tail_share": len(v["tail"]) / prose_len if prose_len else None,
                    "tail_head": v["tail"][:150]})
        rows.append(row)

    s = (pl.DataFrame(rows)
           .join(pl.read_parquet(F).select("accession", "filing_date", "acceptance_et"),
                 on="accession")
           .sort("symbol", "acceptance_et"))

    # Coverage rule: companies where most releases have too little evidence are flagged
    cov = (s.group_by("symbol")
             .agg(pl.col("low_evidence").mean().alias("low_share"))
             .with_columns((pl.col("low_share") <= MAX_LOW_EVIDENCE_SHARE).alias("scoreable")))
    s = s.join(cov.select("symbol", "scoreable"), on="symbol")
    excluded = cov.filter(~pl.col("scoreable"))["symbol"].to_list()
    print(f"Flagged unscoreable (> {MAX_LOW_EVIDENCE_SHARE:.0%} low-evidence): {excluded or 'none'}")

        # Two surprise definitions, both past-only (no look-ahead):
    #   tone_surprise     : vs the mean of the previous 4 releases
    #   tone_surprise_yoy : vs the same quarter one year earlier (removes seasonality)
    s = s.with_columns(
        (pl.col("tone") - pl.col("tone").shift(1).rolling_mean(4).over("symbol"))
          .alias("tone_surprise"),
        (pl.col("tone") - pl.col("tone").shift(4).over("symbol"))
          .alias("tone_surprise_yoy"))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    s.write_parquet(OUT_DIR / "sentiment.parquet")
    pl.concat([counts_frame(pos_ctr, "positive"), counts_frame(neg_ctr, "negative"),
               counts_frame(tail_pos, "tail_positive"), counts_frame(tail_neg, "tail_negative")]
              ).sort("count", descending=True).write_csv(OUT_DIR / "lm_word_counts.csv")
    pl.concat([counts_frame(c, side, sym)
               for side, d in (("positive", sym_pos), ("negative", sym_neg))
               for sym, c in d.items()]
              ).sort("symbol", "side", "count", descending=[False, False, True]
              ).write_csv(OUT_DIR / "lm_word_counts_by_symbol.csv")
    print(f"Scored {s.height} earnings releases -> sentiment.parquet")


if __name__ == "__main__":
    run()