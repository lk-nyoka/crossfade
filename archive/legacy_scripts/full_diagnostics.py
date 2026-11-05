import os
import sys
from pathlib import Path
import json
import duckdb
import pandas as pd
import numpy as np
from collections import defaultdict

# Project base directory
BASE_DIR = Path(r"C:\Users\ASUS\Desktop\Coding Projects\Build or Die\Week 3\Hackathon\Crossfade")
sys.path.insert(0, str(BASE_DIR))

# Import the conservative normaliser
from ml.pipelines.song_identity import normalize_song_identity

# Paths
parquet_glob = (BASE_DIR / "data" / "processed" / "youtube" / "year=*" / "*.parquet").as_posix()
report_dir = BASE_DIR / "ml" / "reports"
report_dir.mkdir(parents=True, exist_ok=True)

con = duckdb.connect()

# ---------------------------------------------------------------------------
# 1. Load data and create helper tables
# ---------------------------------------------------------------------------
con.execute(f"""
CREATE OR REPLACE TEMPORARY TABLE history_chart AS
SELECT DISTINCT key, cc, CAST(week_end AS DATE) AS week_end, title, artists
FROM read_parquet('{parquet_glob}')
""")

# Canonical mapping: for each key pick the most frequent (title, artists) pair
key_meta = con.execute("""
WITH cnt AS (
    SELECT key, title, artists, COUNT(*) AS obs
    FROM history_chart
    GROUP BY key, title, artists
)
SELECT key, title, artists
FROM (
    SELECT *, ROW_NUMBER() OVER (PARTITION BY key ORDER BY obs DESC) AS rn
    FROM cnt
) t
WHERE rn = 1
""").fetchdf()

# Build canonical dict in Python
canon_dict = {}
for _, row in key_meta.iterrows():
    canon_id, _, _ = normalize_song_identity(row['title'], row['artists'])
    canon_dict[row['key']] = canon_id

# Register as DuckDB view for joins
con.register('canon_map', pd.DataFrame(list(canon_dict.items()), columns=['key', 'canon_id']))

# Add canonical column to history_chart
con.execute("""
CREATE OR REPLACE TEMPORARY TABLE chart_canonical AS
SELECT h.*, c.canon_id
FROM history_chart h
LEFT JOIN canon_map c USING (key)
""")

# First chart week per key (overall) and per key per country
con.execute("""
CREATE OR REPLACE TEMPORARY TABLE first_entry_overall AS
SELECT key, MIN(week_end) AS first_week
FROM chart_canonical
GROUP BY key
""")

con.execute("""
CREATE OR REPLACE TEMPORARY TABLE first_entry_country AS
SELECT key, cc, MIN(week_end) AS entry_date
FROM chart_canonical
GROUP BY key, cc
""")

# ---------------------------------------------------------------------------
# A. Active songs per snapshot
# ---------------------------------------------------------------------------
# Define snapshot weeks (eligible weeks for candidates) – use same range as label generation
snapshot_start = '2018-11-01'
snapshot_end = '2026-09-17'
con.execute(f"""
CREATE OR REPLACE TEMPORARY TABLE snapshots AS
SELECT DISTINCT week_end AS snapshot_date
FROM chart_canonical
WHERE week_end BETWEEN DATE '{snapshot_start}' AND DATE '{snapshot_end}'
""")

# Active songs per snapshot (canonical)
active_per_snap = con.execute("""
SELECT s.snapshot_date, COUNT(DISTINCT canon_id) AS active_canonical_songs
FROM snapshots s
JOIN chart_canonical c ON c.week_end = s.snapshot_date
GROUP BY s.snapshot_date
ORDER BY s.snapshot_date
""").fetchdf()

# Compute stats
active_stats = {
    'average': active_per_snap['active_canonical_songs'].mean(),
    'median': active_per_snap['active_canonical_songs'].median(),
    'min': int(active_per_snap['active_canonical_songs'].min()),
    'max': int(active_per_snap['active_canonical_songs'].max()),
    'total_unique_canonical_songs': int(con.execute('SELECT COUNT(DISTINCT canon_id) FROM chart_canonical').fetchone()[0])
}

# Candidate rows per snapshot (before applying target country exclusion)
# Number of distinct target countries (excluding any global placeholder – we treat 'global' as cc='global' if exists)
con.execute("""
CREATE OR REPLACE TEMPORARY TABLE distinct_countries AS
SELECT DISTINCT cc FROM chart_canonical WHERE cc != 'global'
""")

num_countries = con.execute('SELECT COUNT(*) FROM distinct_countries').fetchone()[0]

candidates_per_snap = con.execute(f"""
SELECT s.snapshot_date, COUNT(DISTINCT c.key) AS active_keys, COUNT(*) * {num_countries} AS candidate_rows_est
FROM snapshots s
JOIN chart_canonical c ON c.week_end = s.snapshot_date
GROUP BY s.snapshot_date
""").fetchdf()

# ---------------------------------------------------------------------------
# B. Positive outcomes near first chart appearance
# ---------------------------------------------------------------------------
# Build candidate+label table like label.py (but keep only needed columns)
con.execute(f"""
CREATE OR REPLACE TEMPORARY TABLE candidates AS
WITH active_songs AS (
    SELECT DISTINCT week_end AS snapshot_date, key, canon_id
    FROM chart_canonical
    WHERE week_end BETWEEN DATE '{snapshot_start}' AND DATE '{snapshot_end}'
    AND DATE_DIFF('day', (SELECT first_week FROM first_entry_overall WHERE key = chart_canonical.key), week_end) <= 26*7
),
all_countries AS (SELECT DISTINCT cc FROM distinct_countries)
SELECT a.snapshot_date, a.key, a.canon_id, c.target_cc
FROM active_songs a
CROSS JOIN all_countries c
WHERE NOT EXISTS (
    SELECT 1 FROM chart_canonical h
    WHERE h.key = a.key AND h.cc = c.target_cc AND h.week_end <= a.snapshot_date
)
""")

# Attach future entry dates
con.execute("""
CREATE OR REPLACE TEMPORARY TABLE labeled_candidates AS
SELECT cand.*, fe.entry_date,
CASE WHEN fe.entry_date IS NOT NULL AND fe.entry_date > cand.snapshot_date AND fe.entry_date <= cand.snapshot_date + INTERVAL 28 DAYS THEN 1 ELSE 0 END AS label
FROM candidates cand
LEFT JOIN first_entry_country fe ON cand.key = fe.key AND cand.target_cc = fe.cc
""")

# Positive rows
positives = con.execute('SELECT * FROM labeled_candidates WHERE label = 1').fetchdf()

# First overall chart week per canonical song
first_overall = con.execute('SELECT canon_id, MIN(first_week) AS first_week FROM first_entry_overall JOIN canon_map USING (key) GROUP BY canon_id').fetchdf()
first_overall_dict = dict(zip(first_overall['canon_id'], pd.to_datetime(first_overall['first_week'])))

# Map each positive to its canon first week
positives['snapshot_date'] = pd.to_datetime(positives['snapshot_date'])
positives['canon_id'] = positives['canon_id']
positives['first_week'] = positives['canon_id'].map(first_overall_dict)

# Within first week or next week (<= first_week + 7 days)
within_1w = positives[positives['snapshot_date'] <= positives['first_week'] + pd.Timedelta(days=6)]
within_2w = positives[positives['snapshot_date'] <= positives['first_week'] + pd.Timedelta(days=13)]

pos_near_counts = {
    'within_1w_rows': int(len(within_1w)),
    'within_2w_rows': int(len(within_2w)),
    'total_positive_rows': int(len(positives)),
    'within_1w_pct_of_positives': len(within_1w) / len(positives) * 100 if len(positives) else 0,
    'within_2w_pct_of_positives': len(within_2w) / len(positives) * 100 if len(positives) else 0,
}

# Unique positive events (canon, target_cc, entry week) within those windows
unique_events_1w = within_1w.drop_duplicates(subset=['canon_id', 'target_cc', 'entry_date']).shape[0]
unique_events_2w = within_2w.drop_duplicates(subset=['canon_id', 'target_cc', 'entry_date']).shape[0]
pos_near_counts.update({
    'unique_events_within_1w': unique_events_1w,
    'unique_events_within_2w': unique_events_2w,
})

# ---------------------------------------------------------------------------
# C. Simultaneous-launch songs
# ---------------------------------------------------------------------------
# First appearance week per (canon, cc) already in first_entry_country via join to canon_map
con.execute("""
CREATE OR REPLACE TEMPORARY TABLE canon_first_country AS
SELECT c.canon_id, fe.cc, fe.entry_date
FROM first_entry_country fe
JOIN canon_map c USING (key)
""")

# For each canon, count distinct countries where entry_date == overall first week for that canon
# First overall week per canon
con.execute("""
CREATE OR REPLACE TEMPORARY TABLE canon_overall_first AS
SELECT canon_id, MIN(entry_date) AS canon_first_week
FROM canon_first_country
GROUP BY canon_id
""")

simul = con.execute("""
SELECT c.canon_id,
       COUNT(DISTINCT c.cc) AS country_count,
       COUNT(DISTINCT c.key) AS source_key_count
FROM canon_first_country c
JOIN canon_overall_first o USING (canon_id)
WHERE c.entry_date = o.canon_first_week
GROUP BY c.canon_id
HAVING COUNT(DISTINCT c.cc) >= 3
""").fetchdf()

simul_stats = {
    'num_simultaneous_launch_songs': int(simul.shape[0]),
    'total_source_keys_in_simul': int(simul['source_key_count'].sum()),
    'average_countries_per_song': float(simul['country_count'].mean()) if simul.shape[0] else 0,
}

# Positive breakout events for these songs (from positives dataframe)
simul_canon_set = set(simul['canon_id'])
simul_positives = positives[positives['canon_id'].isin(simul_canon_set)]
simul_positives_count = int(simul_positives.shape[0])
simul_stats['positive_breakout_rows_for_simul'] = simul_positives_count

# ---------------------------------------------------------------------------
# D. Candidates excluded for incomplete outcome windows / other reasons
# ---------------------------------------------------------------------------
# Determine label_valid flag as in label.py (snapshot_date + 28 days <= 2026-10-15)
con.execute("""
CREATE OR REPLACE TEMPORARY TABLE labeled_candidates_full AS
SELECT *, (snapshot_date + INTERVAL 28 DAYS <= DATE '2026-10-15') AS label_valid
FROM labeled_candidates
""")

# Count exclusions
total_candidates = con.execute('SELECT COUNT(*) FROM candidates').fetchone()[0]
valid_candidates = con.execute('SELECT COUNT(*) FROM labeled_candidates_full WHERE label_valid').fetchone()[0]
invalid_future = total_candidates - valid_candidates

# Insufficient historical eligibility already filtered (song_age_weeks > 26) – we can compute how many rows were dropped before candidate creation by comparing with a broader set
# For simplicity, report as NOT DONE for that subcategory (requires re‑generation of broader set).

exclusion_counts = {
    'total_candidates_generated': total_candidates,
    'excluded_incomplete_future_window': int(invalid_future),
    'excluded_due_to_insufficient_history': 'NOT DONE',
    'excluded_due_to_data_quality_or_identity': 'NOT DONE'
}

# ---------------------------------------------------------------------------
# Split statistics (validation folds and final test)
# ---------------------------------------------------------------------------
fold_specs = [
    ("fold_2022", "2018-11-01", "2021-12-02", "2022-01-01", "2022-12-31"),
    ("fold_2023", "2018-11-01", "2022-12-01", "2023-01-01", "2023-12-31"),
    ("fold_2024", "2018-11-01", "2023-11-30", "2024-01-01", "2024-12-31"),
    ("final_test", "2018-11-01", "2024-12-05", "2025-01-01", "2026-09-17")
]

split_rows = []
for fname, tr_s, tr_e, v_s, v_e in fold_specs:
    # Training candidates (using same logic but limited to training snapshot range)
    train_cands = con.execute(f"""
    WITH active_songs AS (
        SELECT DISTINCT week_end AS snapshot_date, key, canon_id
        FROM chart_canonical
        WHERE week_end BETWEEN DATE '{tr_s}' AND DATE '{tr_e}'
        AND DATE_DIFF('day', (SELECT first_week FROM first_entry_overall WHERE key = chart_canonical.key), week_end) <= 26*7
    ), all_countries AS (SELECT DISTINCT cc FROM distinct_countries)
    SELECT a.snapshot_date, a.key, a.canon_id, c.target_cc
    FROM active_songs a
    CROSS JOIN all_countries c
    WHERE NOT EXISTS (
        SELECT 1 FROM chart_canonical h
        WHERE h.key = a.key AND h.cc = c.target_cc AND h.week_end <= a.snapshot_date
    )
    """).fetchdf()
    train_total = len(train_cands)
    train_unique_songs = train_cands['canon_id'].nunique()
    train_unique_countries = train_cands['target_cc'].nunique()
    # Evaluation candidates (validation or test)
    eval_cands = con.execute(f"""
    WITH active_songs AS (
        SELECT DISTINCT week_end AS snapshot_date, key, canon_id
        FROM chart_canonical
        WHERE week_end BETWEEN DATE '{v_s}' AND DATE '{v_e}'
        AND DATE_DIFF('day', (SELECT first_week FROM first_entry_overall WHERE key = chart_canonical.key), week_end) <= 26*7
    ), all_countries AS (SELECT DISTINCT cc FROM distinct_countries)
    SELECT a.snapshot_date, a.key, a.canon_id, c.target_cc
    FROM active_songs a
    CROSS JOIN all_countries c
    WHERE NOT EXISTS (
        SELECT 1 FROM chart_canonical h
        WHERE h.key = a.key AND h.cc = c.target_cc AND h.week_end <= a.snapshot_date
    )
    """).fetchdf()
    # Attach labels
    eval_labeled = con.execute(f"""
    SELECT e.*, fe.entry_date,
        CASE WHEN fe.entry_date IS NOT NULL AND fe.entry_date > e.snapshot_date AND fe.entry_date <= e.snapshot_date + INTERVAL 28 DAYS THEN 1 ELSE 0 END AS label
    FROM ({eval_cands.query}) e
    LEFT JOIN first_entry_country fe ON e.key = fe.key AND e.target_cc = fe.cc
    """).fetchdf()
    eval_total = len(eval_labeled)
    eval_positives = eval_labeled['label'].sum()
    eval_base_rate = (eval_positives / eval_total) * 100 if eval_total else 0
    eval_unique_songs = eval_labeled['canon_id'].nunique()
    eval_unique_countries = eval_labeled['target_cc'].nunique()
    # Groups with at least one positive (snapshot,key)
    pos_groups = eval_labeled[eval_labeled['label'] == 1].groupby(['snapshot_date', 'canon_id']).ngroups
    # Country-snapshot groups with at least one positive
    pos_country_groups = eval_labeled[eval_labeled['label'] == 1].groupby(['snapshot_date', 'target_cc']).ngroups
    split_rows.append({
        'Fold': fname,
        'Train_Range': f"{tr_s} to {tr_e}",
        'Eval_Range': f"{v_s} to {v_e}",
        'Train_Candidates': train_total,
        'Eval_Candidates': eval_total,
        'Eval_Positives': int(eval_positives),
        'Eval_Base_Rate_%': f"{eval_base_rate:.3f}%",
        'Eval_Unique_Songs': int(eval_unique_songs),
        'Eval_Unique_Countries': int(eval_unique_countries),
        'Eval_Pos_Groups_Song': int(pos_groups),
        'Eval_Pos_Groups_Country': int(pos_country_groups)
    })

split_df = pd.DataFrame(split_rows)
split_df_path = report_dir / "split_statistics.csv"
split_df.to_csv(split_df_path, index=False)

# ---------------------------------------------------------------------------
# Assemble final report dictionary
# ---------------------------------------------------------------------------
report = {
    'active_song_stats': active_stats,
    'candidates_per_snapshot_estimate': {
        'num_countries': int(num_countries),
        'sample_row': candidates_per_snap.head(1).to_dict(orient='records')[0]
    },
    'positive_near_first_chart': pos_near_counts,
    'simultaneous_launch_stats': simul_stats,
    'candidate_exclusion_counts': exclusion_counts,
    'split_statistics_path': str(split_df_path)
}

# Write JSON report
json_path = report_dir / "candidate_diagnostics.json"
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(report, f, indent=2)

print('Diagnostics completed. JSON saved to', json_path)
print('Split CSV saved to', split_df_path)
