from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
E = ROOT / "data/features/lookahead_events.parquet"
N_SEEDS = 200


def ls_stats(sig, ret):
    ls = np.sign(sig) * ret
    return ls.mean() * 1e4, ls.mean() / (ls.std(ddof=1) / np.sqrt(len(ls)))


def run():
    e = pl.read_parquet(E).filter(pl.col("reaction").is_not_null()
                                  & pl.col("abn_true").is_not_null()
                                  & pl.col("abn_filing_date").is_not_null())

    # 1. Mechanism: for after-close releases, the naive window contains the reaction
    ac = e.filter(pl.col("timing") == "after_close")
    print("After-close releases: correlation of each method's return with the reaction")
    print(ac.select(pl.len().alias("n"),
                    pl.corr("reaction", "abn_filing_date").round(3).alias("naive_window"),
                    pl.corr("reaction", "abn_true").round(3).alias("true_window")))

    # 2. Positive control: synthetic signals with a chosen correlation to the reaction.
    # Reaction is rank-transformed so a few huge moves don't dominate.
    z = e["reaction"].rank().to_numpy().astype(float)
    z = (z - z.mean()) / z.std()
    r_naive = e["abn_filing_date"].to_numpy()
    r_true = e["abn_true"].to_numpy()
    reaction = e["reaction"].to_numpy()

    rng = np.random.default_rng(42)
    rows = []
    for rho in [0.0, 0.1, 0.2, 0.3, 0.5]:
        res = []
        for _ in range(N_SEEDS):
            sig = rho * z + np.sqrt(1 - rho ** 2) * rng.standard_normal(len(z))
            ic = pl.DataFrame({"s": sig, "r": reaction}).select(
                pl.corr("s", "r", method="spearman")).item()
            res.append((ic, *ls_stats(sig, r_naive), *ls_stats(sig, r_true)))
        a = np.array(res)
        rows.append({"target_ic": rho,
                     "measured_ic": a[:, 0].mean(),
                     "naive_bp": a[:, 1].mean(), "naive_t": a[:, 2].mean(),
                     "true_bp": a[:, 3].mean(), "true_t": a[:, 4].mean()})

    print(f"\nPositive control ({N_SEEDS} synthetic signals per level, abnormal 1-day returns)")
    print(pl.DataFrame(rows).with_columns(pl.col(pl.Float64).round(2)))

    real = e.select(pl.corr("tone_surprise", "reaction", method="spearman")).item()
    print(f"\nReal tone_surprise sits at measured IC = {real:.3f} on this curve")


if __name__ == "__main__":
    run()