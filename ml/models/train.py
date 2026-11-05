"""Step 3 - baselines vs. LightGBM, walk-forward validation, held-out 2025-26 test, explanations."""
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from ml.paths import INTERIM, REPORTS, MODELS, CONFIG
from ml.evaluation.metrics import summarize, calibration_table

CFG = yaml.safe_load(open(CONFIG))
FEATURES = [
    # song momentum
    "age_weeks", "is_simul_launch", "debut_markets", "n_markets_now", "best_rank_now", "mean_rank_now",
    "log_views_now", "rank_gain_1w", "views_growth_1w", "markets_gain_1w", "n_markets_ever",
    "new_markets_4w", "weeks_since_last_entry", "on_global", "global_rank",
    # song -> target diffusion evidence
    "tr_max", "tr_noisy_or", "tr_mean", "tr_max_now", "tr_max_recent",
    "n_same_lang", "n_same_subregion", "n_same_region",
    # target market
    "mkt_prior", "mkt_turnover_26w", "mkt_imports_26w",
    # artist track record in target
    "artist_hits_in_target", "artist_prior_songs", "artist_target_rate",
]
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_child_samples=200,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0,
              verbose=-1, num_threads=2, seed=7)


def load():
    """Memory-lean load: float32 features, categorical strings, integer group id."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    cols = ["song_id", "t", "cc", "y", "period", "in_eval", "in_train", "w", "top_source"] + FEATURES
    tbl = pq.read_table(INTERIM / "dataset.parquet", columns=cols)
    for c in FEATURES + ["y", "w"]:
        i = tbl.schema.get_field_index(c)
        tbl = tbl.set_column(i, c, tbl.column(c).cast(pa.float32()))
    df = tbl.to_pandas(strings_to_categorical=True, self_destruct=True)
    del tbl
    df["t"] = pd.to_datetime(df["t"])
    week = ((df["t"] - pd.Timestamp("2018-01-01")).dt.days // 7).astype("int64")
    df["gid"] = df["song_id"].cat.codes.astype("int64") * 1000 + week
    return df


def add_baselines(ev, tr):
    rng = np.random.default_rng(1)
    ev["B0_random"] = rng.uniform(size=len(ev))
    ev["B1_market_popularity"] = ev["mkt_prior"]
    ev["B2_transition_max"] = ev["tr_max"]
    ev["B3_transition_noisy_or"] = ev["tr_noisy_or"]
    ev["B4_cultural_proximity"] = 0.5 * ev["n_same_lang"] + 0.3 * ev["n_same_subregion"] + 0.2 * ev["n_same_region"]
    bf = ["mkt_prior", "tr_max", "tr_noisy_or", "n_same_lang", "n_same_region"]
    sc = StandardScaler().fit(tr[bf])
    lr = LogisticRegression(max_iter=500).fit(sc.transform(tr[bf]), tr["y"], sample_weight=tr["w"])
    ev["B5_logistic_blend"] = lr.predict_proba(sc.transform(ev[bf]))[:, 1]
    return ["B0_random", "B1_market_popularity", "B2_transition_max", "B3_transition_noisy_or",
            "B4_cultural_proximity", "B5_logistic_blend"]


def fit(train, valid=None, rounds=None):
    dtr = lgb.Dataset(train[FEATURES], train["y"], weight=train["w"], free_raw_data=False)
    if valid is not None:
        dva = lgb.Dataset(valid[FEATURES], valid["y"], reference=dtr)
        m = lgb.train(PARAMS, dtr, num_boost_round=2000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
    else:
        m = lgb.train(PARAMS, dtr, num_boost_round=rounds)
    return m


def main():
    df = load()
    dev_train = df[df.in_train & (df.period == "dev")]
    rows, rounds = [], []

    # ---- walk-forward validation ----
    for f in CFG["splits"]["folds"]:
        tr = dev_train[dev_train.t <= f["train_end"]]
        va = df[df.in_eval & (df.t >= f["val_start"]) & (df.t <= f["val_end"])].copy()
        m = fit(tr, va)
        rounds.append(m.best_iteration)
        va["Crossfade_LGBM"] = m.predict(va[FEATURES], num_iteration=m.best_iteration)
        names = add_baselines(va, tr) + ["Crossfade_LGBM"]
        for n in names:
            r = summarize(va, n, n_boot=200)
            rows.append({"split": f["name"], "model": n, **r})
        print(f["name"], "best_iter", m.best_iteration,
              {n: round(rows[-len(names) + i]["recall@5"], 3) for i, n in enumerate(names)}, flush=True)

    # ---- final model: train <= final_train_end, test 2025-01-01 .. 2026-09-17 (never touched before) ----
    n_rounds = int(np.mean(rounds))
    tr = dev_train[dev_train.t <= CFG["splits"]["final_train_end"]]
    model = fit(tr, rounds=n_rounds)
    model.save_model(str(MODELS / "crossfade_lgbm.txt"))
    te = df[df.period == "test"].copy()
    te["Crossfade_LGBM"] = model.predict(te[FEATURES])
    names = add_baselines(te, tr) + ["Crossfade_LGBM"]
    emerging = te[te.is_simul_launch == 0]
    simul = te[te.is_simul_launch == 1]
    for n in names:
        rows.append({"split": "test_all", "model": n, **summarize(te, n)})
        rows.append({"split": "test_emerging", "model": n, **summarize(emerging, n)})
        rows.append({"split": "test_simultaneous_launch", "model": n, **summarize(simul, n, n_boot=200)})
    res = pd.DataFrame(rows)
    res.to_csv(REPORTS / "benchmark.csv", index=False)

    # calibration on test
    cal, ece, brier = calibration_table(te["y"].values, te["Crossfade_LGBM"].values)
    cal.to_csv(REPORTS / "calibration_test.csv", index=False)

    # explanations: exact TreeSHAP contributions from LightGBM
    shap_sample = te.sample(n=min(300_000, len(te)), random_state=0)  # exact TreeSHAP on a 300k-row sample
    contrib = model.predict(shap_sample[FEATURES], pred_contrib=True)
    shap_imp = pd.Series(np.abs(contrib[:, :-1]).mean(0), index=FEATURES).sort_values(ascending=False)
    shap_imp.to_csv(REPORTS / "shap_importance.csv", header=["mean_abs_shap"])

    te[["song_id", "t", "cc", "y", "Crossfade_LGBM", "B5_logistic_blend", "top_source"]].to_parquet(
        INTERIM / "test_predictions.parquet", index=False)

    # live week (2026-10-15) for the demo
    live = df[df.period == "live"].copy()
    live["p"] = model.predict(live[FEATURES])
    lc = model.predict(live[FEATURES], pred_contrib=True)
    live_contrib = pd.DataFrame(lc[:, :-1], columns=FEATURES, index=live.index)
    live.to_parquet(INTERIM / "live_predictions.parquet", index=False)
    live_contrib.to_parquet(INTERIM / "live_contrib.parquet", index=False)

    # showcase: test-period predictions with contributions for the top-scoring emerging songs
    summary = {
        "trained_rounds": n_rounds, "fold_best_iterations": rounds,
        "test_rows": int(len(te)), "test_positives": int(te.y.sum()),
        "test_base_rate": float(te.y.mean()), "test_groups_with_entry": int(
            te.groupby("gid").y.max().sum()), "test_groups_total": int(te.gid.nunique()),
        "calibration_ece": float(ece), "brier": float(brier),
        "train_rows": int(len(tr)), "train_positives": int(tr.y.sum()),
    }
    json.dump(summary, open(REPORTS / "run_summary.json", "w"), indent=2)
    pd.set_option("display.width", 200)
    show = res[res.split.isin(["test_all", "test_emerging"])][
        ["split", "model", "hit@1", "hit@3", "recall@5", "recall@5_lo", "recall@5_hi", "recall@10", "ndcg@10", "mrr", "pr_auc"]]
    print(show.round(3).to_string(index=False))
    print(json.dumps(summary, indent=1))
    print(shap_imp.head(12).round(4))


if __name__ == "__main__":
    main()
