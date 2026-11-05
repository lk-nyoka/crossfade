import os
import sys
from pathlib import Path
import duckdb
import pandas as pd
import yaml

BASE_DIR = Path(r"C:\Users\ASUS\Desktop\Coding Projects\Build or Die\Week 3\Hackathon\Crossfade")
sys.path.insert(0, str(BASE_DIR))

CONFIG_PATH = BASE_DIR / "ml" / "config" / "config.yaml"
OUTPUT_DIR = BASE_DIR / "data" / "processed" / "labels"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def generate_candidates_and_labels(start_date: str = "2018-11-01", end_date: str = "2026-09-17", max_age_weeks: int = 26):
    """
    Generates leakage-safe candidate pairs (snapshot_date, song, target_cc) and labels.
    - Snapshot dates strictly between start_date and end_date.
    - Excludes target countries where song has charted on or before snapshot_date.
    - Labels: 1 if song enters target_cc Top 100 in (snapshot, snapshot + 28 days], else 0.
    """
    cfg = load_config()
    parquet_glob = (BASE_DIR / cfg["data"]["parquet_glob"]).as_posix()
    con = duckdb.connect()

    print(f"Building candidate table from {start_date} to {end_date} (max_age_weeks={max_age_weeks})...", flush=True)

    # 1. Identify first appearance and simultaneous launch per song
    print("Computing song first appearance and launch strata...", flush=True)
    con.execute(f"""
        CREATE OR REPLACE TEMPORARY TABLE song_meta AS
        WITH ranked AS (
            SELECT key, CAST(week_end AS DATE) as week_end, cc,
                   DENSE_RANK() OVER (PARTITION BY key ORDER BY week_end ASC) as rk
            FROM read_parquet('{parquet_glob}')
        )
        SELECT 
            key, 
            MIN(week_end) as first_chart_week,
            COUNT(DISTINCT cc) as initial_countries,
            (COUNT(DISTINCT cc) >= {cfg.get('simultaneous_launch_threshold_countries', 3)}) as is_simul_launch
        FROM ranked
        WHERE rk = 1
        GROUP BY key
    """)

    # 2. Extract active songs at each snapshot week
    print("Identifying active songs per snapshot week...", flush=True)
    con.execute(f"""
        CREATE OR REPLACE TEMPORARY TABLE active_snapshot_songs AS
        SELECT DISTINCT 
            CAST(p.week_end AS DATE) as snapshot_date, 
            p.key,
            m.first_chart_week,
            m.is_simul_launch,
            CAST(DATE_DIFF('day', m.first_chart_week, CAST(p.week_end AS DATE)) / 7.0 AS INTEGER) as song_age_weeks
        FROM read_parquet('{parquet_glob}') p
        JOIN song_meta m ON p.key = m.key
        WHERE p.week_end BETWEEN DATE '{start_date}' AND DATE '{end_date}'
          AND DATE_DIFF('day', m.first_chart_week, CAST(p.week_end AS DATE)) <= ({max_age_weeks} * 7)
    """)

    # 3. Build eligible candidate (snapshot_date, key, target_cc) pairs
    # Eligible target_cc: has NOT charted on or before snapshot_date
    print("Cross-joining active songs with unentered target countries...", flush=True)
    con.execute(f"""
        CREATE OR REPLACE TEMPORARY TABLE candidates AS
        WITH all_countries AS (
            SELECT DISTINCT cc as target_cc FROM read_parquet('{parquet_glob}')
        ),
        charted_history AS (
            SELECT DISTINCT key, cc, CAST(week_end AS DATE) as week_end
            FROM read_parquet('{parquet_glob}')
        )
        SELECT 
            a.snapshot_date,
            a.key,
            a.song_age_weeks,
            a.is_simul_launch,
            c.target_cc
        FROM active_snapshot_songs a
        CROSS JOIN all_countries c
        WHERE NOT EXISTS (
            SELECT 1 FROM charted_history h 
            WHERE h.key = a.key AND h.cc = c.target_cc AND h.week_end <= a.snapshot_date
        )
    """)

    # 4. Attach ground truth 4-week breakout label
    print("Evaluating 4-week breakout labels...", flush=True)
    con.execute(f"""
        CREATE OR REPLACE TEMPORARY TABLE candidates_labeled AS
        WITH future_entries AS (
            SELECT 
                key, 
                cc as target_cc, 
                MIN(CAST(week_end AS DATE)) as first_future_entry
            FROM read_parquet('{parquet_glob}')
            GROUP BY key, cc
        )
        SELECT 
            c.snapshot_date,
            c.key,
            c.song_age_weeks,
            c.is_simul_launch,
            c.target_cc,
            (c.snapshot_date + INTERVAL 28 DAYS <= DATE '2026-10-15') as label_valid,
            CASE 
                WHEN fe.first_future_entry IS NOT NULL 
                 AND fe.first_future_entry > c.snapshot_date 
                 AND fe.first_future_entry <= c.snapshot_date + INTERVAL 28 DAYS 
                THEN 1 ELSE 0 
            END as label_top100_4w
        FROM candidates c
        LEFT JOIN future_entries fe ON c.key = fe.key AND c.target_cc = fe.target_cc
    """)

    summary = con.execute("""
        SELECT 
            COUNT(*) as total_candidates,
            SUM(label_top100_4w) as total_positives,
            AVG(label_top100_4w) * 100 as positive_rate_pct,
            COUNT(DISTINCT snapshot_date) as num_snapshots,
            COUNT(DISTINCT key) as num_unique_songs
        FROM candidates_labeled
    """).fetchdf().iloc[0].to_dict()

    print(f"Generated {int(summary['total_candidates']):,} candidate rows.")
    print(f"Total Positives: {int(summary['total_positives']):,} ({summary['positive_rate_pct']:.3f}%)")
    print(f"Snapshots: {int(summary['num_snapshots'])}, Unique Songs: {int(summary['num_unique_songs']):,}")

    return con

if __name__ == "__main__":
    generate_candidates_and_labels()
