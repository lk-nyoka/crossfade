"""Ranking metrics computed per (snapshot week, song) group: 'of the countries this song could enter next,
did the model rank the ones it actually entered near the top?'. Groups with no positive are skipped for
ranking metrics (nothing to find) but are kept for pooled PR-AUC."""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss


def group_metrics(df: pd.DataFrame, score_col: str, ks=(1, 3, 5, 10)) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    d = df[["gid", "t", "y", score_col]].copy()
    d["_s"] = d[score_col].astype("float64") + rng.uniform(0, 1e-9, len(d))  # random tie-break
    d = d[d.groupby("gid")["y"].transform("sum") > 0]
    d = d.sort_values(["gid", "_s"], ascending=[True, False])
    d["rank"] = d.groupby("gid").cumcount() + 1
    pos = d[d.y == 1]
    g = pos.groupby("gid")
    out = pd.DataFrame({"t": g["t"].first(), "n_pos": g.size()})
    for k in ks:
        out[f"recall@{k}"] = g["rank"].apply(lambda r: (r <= k).sum()) / out["n_pos"]
        out[f"hit@{k}"] = g["rank"].apply(lambda r: float((r <= k).any()))
    out["mrr"] = 1.0 / g["rank"].min()
    dcg = pos.assign(g=1 / np.log2(pos["rank"] + 1)).loc[lambda x: x["rank"] <= 10].groupby("gid")["g"].sum()
    ideal = out["n_pos"].clip(upper=10).map(lambda n: sum(1 / np.log2(i + 2) for i in range(n)))
    out["ndcg@10"] = dcg.reindex(out.index).fillna(0) / ideal
    return out


def summarize(df, score_col, n_boot=500, seed=42):
    gm = group_metrics(df, score_col)
    res = {m: gm[m].mean() for m in ["hit@1", "hit@3", "recall@5", "recall@10", "ndcg@10", "mrr"]}
    res["pr_auc"] = average_precision_score(df["y"], df[score_col])
    res["groups"] = len(gm)
    # bootstrap CI for recall@5 by resampling snapshot weeks (keeps within-week correlation)
    rng = np.random.default_rng(seed)
    wk = gm.groupby("t")["recall@5"].agg(["sum", "count"])
    s, c = wk["sum"].values, wk["count"].values
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(wk), len(wk))
        boots.append(s[i].sum() / c[i].sum())
    res["recall@5_lo"], res["recall@5_hi"] = np.percentile(boots, [2.5, 97.5])
    return res


def calibration_table(y, p, bins=10):
    q = pd.qcut(p, bins, duplicates="drop")
    t = pd.DataFrame({"y": y, "p": p, "bin": q}).groupby("bin", observed=True).agg(
        mean_pred=("p", "mean"), observed=("y", "mean"), n=("y", "size")).reset_index(drop=True)
    ece = (abs(t.mean_pred - t.observed) * t.n).sum() / t.n.sum()
    return t, ece, brier_score_loss(y, p)
