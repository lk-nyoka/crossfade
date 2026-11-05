"""Held-out test performance by target region and for songs whose journey started in an African market."""
import pandas as pd
from ml.paths import INTERIM, REPORTS, COUNTRY_META
from ml.evaluation.metrics import summarize


def main():
    te = pd.read_parquet(INTERIM / "test_predictions.parquet")
    te["t"] = pd.to_datetime(te["t"])
    meta = pd.read_csv(COUNTRY_META)
    songs = pd.read_parquet(INTERIM / "songs.parquet")
    region = dict(zip(meta.cc, meta.region))
    te["target_region"] = te["cc"].astype(str).map(region)
    debut_africa = songs.set_index("song_id")["debut_ccs"].map(lambda l: any(region.get(c) == "Africa" for c in l))
    te["debut_in_africa"] = te["song_id"].astype(str).map(debut_africa).fillna(False)
    te["tr_noisy_or_baseline"] = te["tr_noisy_or"]
    rows = []
    for name, d in [("all", te), ("debuted_in_africa", te[te.debut_in_africa])] + \
                   [(f"target_{r}", te[te.target_region == r]) for r in sorted(te.target_region.dropna().unique())]:
        for col in ["Crossfade_LGBM", "tr_noisy_or_baseline"]:
            r = summarize(d, col, n_boot=200)
            rows.append({"slice": name, "model": col, **{k: r[k] for k in ["groups", "hit@1", "recall@5", "recall@10", "pr_auc"]}})
    out = pd.DataFrame(rows)
    out.to_csv(REPORTS / "slices_test.csv", index=False)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
