"""Regenerate step-3 outputs from the saved model without retraining:
SHAP importance (300k-row sample), test/live predictions with contributions, run summary."""
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from ml.paths import INTERIM, REPORTS, MODELS
from ml.models.train import load, FEATURES, CFG
from ml.evaluation.metrics import calibration_table


def main():
    model = lgb.Booster(model_file=str(MODELS / "crossfade_lgbm.txt"))
    df = load()
    te = df[df.period == "test"].copy()
    te["Crossfade_LGBM"] = model.predict(te[FEATURES], num_threads=2)
    smp = te.sample(n=300_000, random_state=0)
    contrib = model.predict(smp[FEATURES], pred_contrib=True)
    imp = pd.Series(np.abs(contrib[:, :-1]).mean(0), index=FEATURES).sort_values(ascending=False)
    imp.to_csv(REPORTS / "shap_importance.csv", header=["mean_abs_shap"])
    te[["song_id", "t", "cc", "y", "Crossfade_LGBM", "top_source", "gid"] + FEATURES].to_parquet(
        INTERIM / "test_predictions.parquet", index=False)

    live = df[df.period == "live"].copy()
    live["p"] = model.predict(live[FEATURES])
    lc = model.predict(live[FEATURES], pred_contrib=True)
    pd.DataFrame(lc[:, :-1], columns=FEATURES).assign(song_id=live.song_id.astype(str).values,
                                                     cc=live.cc.astype(str).values).to_parquet(
        INTERIM / "live_contrib.parquet", index=False)
    live.to_parquet(INTERIM / "live_predictions.parquet", index=False)

    _, ece, brier = calibration_table(te["y"].values, te["Crossfade_LGBM"].values)
    tr = df[df.in_train & (df.period == "dev") & (df.t <= CFG["splits"]["final_train_end"])]
    summary = {
        "trained_rounds": model.num_trees(), "test_rows": int(len(te)), "test_positives": int(te.y.sum()),
        "test_base_rate": float(te.y.mean()), "test_groups_total": int(te.gid.nunique()),
        "test_groups_with_entry": int(te.groupby("gid").y.max().sum()),
        "calibration_ece": float(ece), "brier": float(brier),
        "train_rows": int(len(tr)), "train_positives": int(tr.y.sum()),
        "live_snapshot": str(live.t.max().date()), "live_songs": int(live.song_id.nunique()),
    }
    json.dump(summary, open(REPORTS / "run_summary.json", "w"), indent=2)
    print(json.dumps(summary, indent=1)); print(imp.round(4).to_string())


if __name__ == "__main__":
    main()
