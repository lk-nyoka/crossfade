"""Decision value on the held-out test: a label or distributor can only work a few (song, market) pairs a week.
Each week, every model picks its top-K pairs across all rising songs; we count how many actually entered within
4 weeks. Same rows and same K for every model. Also reported for an Africa-focused roster."""
import json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from ml.paths import INTERIM, REPORTS, COUNTRY_META
from ml.models.train import CFG

BF = ["mkt_prior", "tr_max", "tr_noisy_or", "n_same_lang", "n_same_region"]


def main():
    te = pd.read_parquet(INTERIM / "test_predictions.parquet")
    te["t"] = pd.to_datetime(te["t"])
    te["song_id"] = te.song_id.astype(str)
    # B5 refit exactly as in train.py (logistic blend on the training rows)
    tr = pq.read_table(INTERIM / "dataset.parquet", columns=BF + ["y", "w", "in_train", "period", "t"],
                       filters=[("in_train", "=", True), ("period", "=", "dev")]).to_pandas()
    tr = tr[pd.to_datetime(tr.t) <= CFG["splits"]["final_train_end"]]
    sc = StandardScaler().fit(tr[BF])
    lr = LogisticRegression(max_iter=500).fit(sc.transform(tr[BF]), tr["y"], sample_weight=tr["w"])
    te["B5_logistic_blend"] = lr.predict_proba(sc.transform(te[BF].astype("float64")))[:, 1]
    te["B3_transition_noisy_or"] = te["tr_noisy_or"]
    te["B1_market_popularity"] = te["mkt_prior"]
    te["B0_random"] = np.random.default_rng(0).uniform(size=len(te))
    models = ["Crossfade_LGBM", "B5_logistic_blend", "B3_transition_noisy_or", "B1_market_popularity", "B0_random"]

    meta = pd.read_csv(COUNTRY_META)
    afr = set(meta[meta.region == "Africa"].cc)
    charts = pd.read_parquet(INTERIM / "charts.parquet")
    charts["week_end"] = pd.to_datetime(charts["week_end"])
    af_now = set(map(tuple, charts[charts.cc.isin(afr)][["song_id", "week_end"]].astype({"song_id": str}).values))
    te["af_roster"] = [(s, t) in af_now for s, t in zip(te.song_id, te.t)]

    rows = []
    for roster, d in [("all rising songs", te), ("songs charting in Africa", te[te.af_roster])]:
        weeks = d.t.nunique()
        for k in [10, 25, 50, 100]:
            for m in models:
                top = d.sort_values(m, ascending=False).groupby("t").head(k)
                rows.append({"roster": roster, "k_per_week": k, "model": m, "weeks": weeks,
                             "bets": len(top), "hits": int(top.y.sum()), "precision": top.y.mean(),
                             "base_rate": d.y.mean(), "global_launch_share": top.is_simul_launch.mean(),
                             "distinct_songs_median": top.groupby("t").song_id.nunique().median()})
    # robustness: emerging songs only, and a diversified slate (each song's single best market, top 10 songs)
    for m in models:
        em = te[te.is_simul_launch == 0]
        top = em.sort_values(m, ascending=False).groupby("t").head(10)
        rows.append({"roster": "emerging songs only", "k_per_week": 10, "model": m, "weeks": em.t.nunique(),
                     "bets": len(top), "hits": int(top.y.sum()), "precision": top.y.mean(), "base_rate": em.y.mean(),
                     "distinct_songs_median": top.groupby("t").song_id.nunique().median()})
        best = te.sort_values(m, ascending=False).drop_duplicates(["t", "song_id"]).groupby("t").head(10)
        rows.append({"roster": "10 songs x their #1 market", "k_per_week": 10, "model": m, "weeks": te.t.nunique(),
                     "bets": len(best), "hits": int(best.y.sum()), "precision": best.y.mean(), "base_rate": te.y.mean(),
                     "distinct_songs_median": 10.0})
    out = pd.DataFrame(rows)
    out["lift_vs_random_pick"] = out.precision / out.base_rate
    out.to_csv(REPORTS / "decision_value.csv", index=False)
    pd.set_option("display.width", 200)
    print(out.pivot_table(index=["roster", "k_per_week"], columns="model", values="precision").round(3)
          [models].to_string())
    print(out[out.model == "Crossfade_LGBM"][["roster", "k_per_week", "bets", "hits", "precision", "base_rate", "lift_vs_random_pick"]].round(4).to_string(index=False))
    # B5 sanity: PR-AUC must match benchmark.csv
    from sklearn.metrics import average_precision_score
    print("B5 PR-AUC check:", round(average_precision_score(te.y, te.B5_logistic_blend), 4))


if __name__ == "__main__":
    main()
