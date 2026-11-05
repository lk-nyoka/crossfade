import os
import sys
import json
from pathlib import Path
from collections import defaultdict
import duckdb
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = Path(r"C:\Users\ASUS\Desktop\Coding Projects\Build or Die\Week 3\Hackathon\Crossfade")
sys.path.insert(0, str(BASE_DIR))

from ml.models.baselines import (
    CountryPairTransitionModel,
    CulturalProximityModel,
    compute_ranking_metrics
)

parquet_glob = (BASE_DIR / "data" / "processed" / "youtube" / "year=*" / "*.parquet").as_posix()
country_meta_path = BASE_DIR / "data" / "reference" / "country_meta.csv"
con = duckdb.connect()

print("=" * 75, flush=True)
print("CROSSFADE PHASE 1 & 2: WALK-FORWARD FOLDS & BASELINE BENCHMARK", flush=True)
print("=" * 75, flush=True)

# --- 1. Walk-Forward Fold Table ---
print("\n--- 1. Computing Walk-Forward Folds Statistics ---", flush=True)
# We compute validation candidate stats for 2022, 2023, 2024, and final test (2025-2026)
fold_specs = [
    ("fold_2022", "2018-11-01", "2021-12-02", "2022-01-01", "2022-12-31"),
    ("fold_2023", "2018-11-01", "2022-12-01", "2023-01-01", "2023-12-31"),
    ("fold_2024", "2018-11-01", "2023-11-30", "2024-01-01", "2024-12-31"),
    ("final_test", "2018-11-01", "2024-12-05", "2025-01-01", "2026-09-17")
]

# Create indexed views or tables for lightning fast query
con.execute(f"""
    CREATE OR REPLACE TEMPORARY TABLE history_chart AS
    SELECT DISTINCT key, cc, CAST(week_end AS DATE) as week_end
    FROM read_parquet('{parquet_glob}')
""")

con.execute(f"""
    CREATE OR REPLACE TEMPORARY TABLE song_first_entry AS
    SELECT key, cc, MIN(week_end) as entry_date
    FROM history_chart
    GROUP BY key, cc
""")

fold_results = []
for fname, tr_s, tr_e, v_s, v_e in fold_specs:
    print(f"  Calculating {fname} ({v_s} to {v_e})...", flush=True)
    f_stats = con.execute(f"""
        WITH 
        active_songs AS (
            SELECT DISTINCT week_end as snapshot_date, key
            FROM history_chart
            WHERE week_end BETWEEN DATE '{v_s}' AND DATE '{v_e}'
        ),
        all_countries AS (
            SELECT DISTINCT cc as target_cc FROM history_chart
        ),
        candidates AS (
            SELECT a.snapshot_date, a.key, c.target_cc
            FROM active_songs a
            CROSS JOIN all_countries c
            WHERE NOT EXISTS (
                SELECT 1 FROM history_chart h
                WHERE h.key = a.key AND h.cc = c.target_cc AND h.week_end <= a.snapshot_date
            )
        )
        SELECT 
            COUNT(*) as candidates,
            SUM(CASE WHEN fe.entry_date IS NOT NULL 
                     AND fe.entry_date > c.snapshot_date 
                     AND fe.entry_date <= c.snapshot_date + INTERVAL 28 DAYS THEN 1 ELSE 0 END) as positives
        FROM candidates c
        LEFT JOIN song_first_entry fe ON c.key = fe.key AND c.target_cc = fe.cc
    """).fetchdf().iloc[0].to_dict()
    
    cand = int(f_stats["candidates"])
    pos = int(f_stats["positives"])
    rate = (pos / cand) * 100 if cand else 0.0
    
    fold_results.append({
        "Fold": fname,
        "Train Range": f"{tr_s} to {tr_e}",
        "Eval Range": f"{v_s} to {v_e}",
        "Candidates": cand,
        "Positives": pos,
        "Base Rate (%)": f"{rate:.3f}%"
    })

fold_df = pd.DataFrame(fold_results)
print("\n" + fold_df.to_string(index=False), flush=True)

# --- 2. Fit Models Strictly Prior to Test Period ---
train_cutoff = "2024-12-05" # 4-week purge gap before 2025-01-01
print(f"\n--- 2. Fitting Baseline Models with Strict Cutoff at {train_cutoff} ---", flush=True)

print("  Fitting CountryPairTransitionModel (B3)...", flush=True)
b3_model = CountryPairTransitionModel()
b3_model.fit_from_duckdb(con, f"(SELECT * FROM history_chart WHERE week_end <= DATE '{train_cutoff}')")
b1_prior = b3_model.country_prior

print("  Fitting CulturalProximityModel (B4)...", flush=True)
b4_model = CulturalProximityModel(country_meta_path)
hist_training_df = con.execute(f"""
    SELECT DISTINCT key, cc 
    FROM history_chart 
    WHERE week_end <= DATE '{train_cutoff}'
""").fetchdf()
b4_model.fit_chart_overlap(hist_training_df)

# Fit B5 (Logistic blend) on a 2024 validation sample
print("  Fitting B5 Logistic Blend on 2024 validation sample...", flush=True)
val_sample = con.execute(f"""
    WITH 
    active_songs AS (
        SELECT DISTINCT week_end as snapshot_date, key
        FROM history_chart
        WHERE week_end BETWEEN DATE '2024-01-01' AND DATE '2024-06-01'
    ),
    all_countries AS (
        SELECT DISTINCT cc as target_cc FROM history_chart
    ),
    candidates AS (
        SELECT a.snapshot_date, a.key, c.target_cc
        FROM active_songs a
        CROSS JOIN all_countries c
        WHERE NOT EXISTS (
            SELECT 1 FROM history_chart h
            WHERE h.key = a.key AND h.cc = c.target_cc AND h.week_end <= a.snapshot_date
        )
    )
    SELECT 
        c.snapshot_date, c.key, c.target_cc,
        CASE WHEN fe.entry_date IS NOT NULL 
             AND fe.entry_date > c.snapshot_date 
             AND fe.entry_date <= c.snapshot_date + INTERVAL 28 DAYS THEN 1 ELSE 0 END as y
    FROM candidates c
    LEFT JOIN song_first_entry fe ON c.key = fe.key AND c.target_cc = fe.cc
    USING SAMPLE 50000
""").fetchdf()

val_markets = con.execute(f"""
    SELECT DISTINCT key, week_end as snapshot_date, cc
    FROM history_chart
    WHERE week_end BETWEEN DATE '2024-01-01' AND DATE '2024-06-01'
""").fetchdf().groupby(["key", "snapshot_date"])["cc"].apply(list).to_dict()

X_val = []
for k, d, tgt in zip(val_sample["key"].values, val_sample["snapshot_date"].values, val_sample["target_cc"].values):
    cm = val_markets.get((k, d), [])
    s_b1 = b1_prior.get(tgt, 0.0)
    s_b3 = b3_model.predict_pair(cm, tgt, method="max")
    s_b4 = b4_model.predict_pair(cm, tgt)
    X_val.append([s_b1, s_b3, s_b4])

blend_lr = LogisticRegression(class_weight="balanced", max_iter=200)
blend_lr.fit(np.array(X_val), val_sample["y"].values)
print(f"  B5 Logistic Blend fit: Intercept={blend_lr.intercept_[0]:.3f}, B1={blend_lr.coef_[0][0]:.3f}, B3={blend_lr.coef_[0][1]:.3f}, B4={blend_lr.coef_[0][2]:.3f}", flush=True)

# --- 3. Evaluate on Final Test Set (2025-01-01 to 2026-09-17) ---
print("\n--- 3. Extracting Test Period Cohort with Positives ---", flush=True)
# To evaluate macro-ranking metrics over groups with >= 1 positive, extract directly in SQL
test_cohort_df = con.execute(f"""
    WITH 
    active_songs AS (
        SELECT DISTINCT week_end as snapshot_date, key
        FROM history_chart
        WHERE week_end BETWEEN DATE '2025-01-01' AND DATE '2026-09-17'
    ),
    all_countries AS (
        SELECT DISTINCT cc as target_cc FROM history_chart
    ),
    candidates AS (
        SELECT a.snapshot_date, a.key, c.target_cc
        FROM active_songs a
        CROSS JOIN all_countries c
        WHERE NOT EXISTS (
            SELECT 1 FROM history_chart h
            WHERE h.key = a.key AND h.cc = c.target_cc AND h.week_end <= a.snapshot_date
        )
    ),
    labeled AS (
        SELECT 
            c.snapshot_date, c.key, c.target_cc,
            CASE WHEN fe.entry_date IS NOT NULL 
                 AND fe.entry_date > c.snapshot_date 
                 AND fe.entry_date <= c.snapshot_date + INTERVAL 28 DAYS THEN 1 ELSE 0 END as y
        FROM candidates c
        LEFT JOIN song_first_entry fe ON c.key = fe.key AND c.target_cc = fe.cc
    ),
    positive_groups AS (
        SELECT snapshot_date, key
        FROM labeled
        GROUP BY snapshot_date, key
        HAVING SUM(y) > 0
    )
    SELECT l.snapshot_date, l.key, l.target_cc, l.y
    FROM labeled l
    JOIN positive_groups pg ON l.snapshot_date = pg.snapshot_date AND l.key = pg.key
""").fetchdf()

print(f"  Total candidates in positive groups cohort: {len(test_cohort_df):,}", flush=True)
num_groups = test_cohort_df.groupby(["snapshot_date", "key"]).ngroups
total_positives = int(test_cohort_df["y"].sum())
unique_songs = test_cohort_df["key"].nunique()
unique_snapshots = test_cohort_df["snapshot_date"].nunique()
base_rate = (total_positives / len(test_cohort_df)) * 100

print(f"  Test Cohort Summary:", flush=True)
print(f"    Snapshot Weeks: {unique_snapshots}", flush=True)
print(f"    (Snapshot, Song) Groups: {num_groups:,}", flush=True)
print(f"    Unique Songs: {unique_songs:,}", flush=True)
print(f"    Total Positives: {total_positives:,}", flush=True)
print(f"    Cohort Base Rate: {base_rate:.2f}%", flush=True)

# Get current markets for songs in test cohort
test_markets = con.execute(f"""
    SELECT DISTINCT key, week_end as snapshot_date, cc
    FROM history_chart
    WHERE week_end BETWEEN DATE '2025-01-01' AND DATE '2026-09-17'
""").fetchdf().groupby(["key", "snapshot_date"])["cc"].apply(list).to_dict()

# Predict all baselines with fast array iteration
print("  Scoring candidates across baselines B0, B1, B2, B3 (Max & Noisy-OR), B4, B5...", flush=True)
n_rows = len(test_cohort_df)
np.random.seed(42)
scores_b0 = np.random.uniform(0, 1, size=n_rows)
scores_b1 = np.empty(n_rows, dtype=np.float32)
scores_b2 = np.empty(n_rows, dtype=np.float32)
scores_b3_max = np.empty(n_rows, dtype=np.float32)
scores_b3_nor = np.empty(n_rows, dtype=np.float32)
scores_b4 = np.empty(n_rows, dtype=np.float32)

keys_arr = test_cohort_df["key"].values
dates_arr = test_cohort_df["snapshot_date"].values
targets_arr = test_cohort_df["target_cc"].values

for i in range(n_rows):
    k = keys_arr[i]
    d = dates_arr[i]
    tgt = targets_arr[i]
    cm = test_markets.get((k, d), [])
    
    s1 = b1_prior.get(tgt, 0.0)
    scores_b1[i] = s1
    scores_b2[i] = len(cm)
    scores_b3_max[i] = b3_model.predict_pair(cm, tgt, method="max")
    scores_b3_nor[i] = b3_model.predict_pair(cm, tgt, method="noisy_or")
    scores_b4[i] = b4_model.predict_pair(cm, tgt)

# B5 blend score
X_test = np.column_stack([scores_b1, scores_b3_max, scores_b4])
scores_b5 = blend_lr.predict_proba(X_test)[:, 1]

test_cohort_df["score_b0"] = scores_b0
test_cohort_df["score_b1"] = scores_b1
test_cohort_df["score_b2"] = scores_b2
test_cohort_df["score_b3_max"] = scores_b3_max
test_cohort_df["score_b3_nor"] = scores_b3_nor
test_cohort_df["score_b4"] = scores_b4
test_cohort_df["score_b5"] = scores_b5

models_dict = {
    "B0: Random": "score_b0",
    "B1: Country Popularity Prior": "score_b1",
    "B2: Song Current Markets Count": "score_b2",
    "B3: Transition (Max)": "score_b3_max",
    "B3: Transition (Noisy-OR)": "score_b3_nor",
    "B4: Cultural Proximity": "score_b4",
    "B5: Logistic Blend (B3+B4)": "score_b5"
}

# Fast group evaluation
print("  Computing group ranking metrics...", flush=True)
group_keys = list(test_cohort_df.groupby(["snapshot_date", "key"]).groups.keys())
group_indices = test_cohort_df.groupby(["snapshot_date", "key"]).indices

group_metrics = {m: defaultdict(list) for m in models_dict}
group_summary = {}

y_all = test_cohort_df["y"].values

for gk in group_keys:
    idx = group_indices[gk]
    y_grp = y_all[idx]
    group_summary[gk] = {}
    
    for m_name, col in models_dict.items():
        sc = test_cohort_df[col].values[idx] + np.random.uniform(0, 1e-7, size=len(idx))
        met = compute_ranking_metrics(y_grp, sc)
        group_summary[gk][m_name] = met
        for k, v in met.items():
            group_metrics[m_name][k].append(v)

# --- 4. Bootstrap Confidence Intervals (1,000 Resamples by Snapshot Week & Song) ---
print("\n--- 4. Computing 95% Bootstrap CIs (1,000 resamples by snapshot week & song) ---", flush=True)
unique_w = np.array(list(set(k[0] for k in group_keys)))
week_groups = defaultdict(list)
for gk in group_keys:
    week_groups[gk[0]].append(gk)

n_boot = 1000
boot_r10 = {m: [] for m in models_dict}

np.random.seed(42)
for _ in range(n_boot):
    sampled_w = np.random.choice(unique_w, size=len(unique_w), replace=True)
    sampled_g = [g for w in sampled_w for g in week_groups[w]]
    for m in models_dict:
        r10_val = np.mean([group_summary[g][m]["recall_at_10"] for g in sampled_g])
        boot_r10[m].append(r10_val)

benchmark_rows = []
for m_name, col in models_dict.items():
    r5 = np.mean(group_metrics[m_name]["recall_at_5"])
    r10 = np.mean(group_metrics[m_name]["recall_at_10"])
    ndcg10 = np.mean(group_metrics[m_name]["ndcg_at_10"])
    mrr = np.mean(group_metrics[m_name]["mrr"])
    pooled_ap = average_precision_score(y_all, test_cohort_df[col].values)
    
    ci_l = np.percentile(boot_r10[m_name], 2.5)
    ci_h = np.percentile(boot_r10[m_name], 97.5)
    
    benchmark_rows.append({
        "Model": m_name,
        "Recall@5": f"{r5*100:.2f}%",
        "Recall@10": f"{r10*100:.2f}%",
        "Recall@10 (95% CI)": f"[{ci_l*100:.1f}%, {ci_h*100:.1f}%]",
        "NDCG@10": f"{ndcg10:.3f}",
        "MRR": f"{mrr:.3f}",
        "Pooled AP": f"{pooled_ap:.4f}"
    })

bench_df = pd.DataFrame(benchmark_rows)
print("\n" + "=" * 90, flush=True)
print("FINAL BASELINE BENCHMARK RESULTS (Test Period 2025-01-01 to 2026-09-17)", flush=True)
print("=" * 90, flush=True)
print(bench_df.to_string(index=False), flush=True)
print("=" * 90, flush=True)

out_csv = BASE_DIR / "ml" / "reports" / "baseline_benchmark.csv"
bench_df.to_csv(out_csv, index=False)
print(f"Results saved to: {out_csv}", flush=True)
