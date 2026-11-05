import sys
from pathlib import Path
import duckdb
import pandas as pd

BASE_DIR = Path(r"C:\Users\ASUS\Desktop\Coding Projects\Build or Die\Week 3\Hackathon\Crossfade")
parquet_glob = (BASE_DIR / "data" / "processed" / "youtube" / "year=*" / "*.parquet").as_posix()
con = duckdb.connect()

print("=" * 70)
print("CANDIDATE DEFINITION & SCALING SIMULATION")
print("=" * 70)

# 1. First chart week per key (song) across all data
print("Computing first chart week per song...")
con.execute(f"""
    CREATE OR REPLACE TEMPORARY TABLE song_first_chart AS
    SELECT key, MIN(week_end) as first_chart_week, COUNT(DISTINCT cc) as initial_countries
    FROM read_parquet('{parquet_glob}')
    WHERE week_end = (SELECT MIN(week_end) FROM read_parquet('{parquet_glob}') as r2 WHERE r2.key = read_parquet.key)
    GROUP BY key
""")

# Actually let's do a much cleaner and faster query in DuckDB
con.execute(f"""
    CREATE OR REPLACE TEMPORARY TABLE song_first_appearance AS
    WITH ranked AS (
        SELECT key, week_end, cc,
               DENSE_RANK() OVER (PARTITION BY key ORDER BY week_end ASC) as rk
        FROM read_parquet('{parquet_glob}')
    )
    SELECT key, MIN(week_end) as first_week, COUNT(DISTINCT cc) as first_week_countries
    FROM ranked
    WHERE rk = 1
    GROUP BY key
""")

# Let's inspect simultaneous launches (first_week_countries >= 3)
simul_stats = con.execute("""
    SELECT 
        COUNT(*) as total_songs,
        SUM(CASE WHEN first_week_countries >= 3 THEN 1 ELSE 0 END) as simul_launch_songs,
        AVG(CASE WHEN first_week_countries >= 3 THEN 1.0 ELSE 0.0 END) as simul_launch_share
    FROM song_first_appearance
""").fetchdf().iloc[0].to_dict()

print(f"Total unique songs: {int(simul_stats['total_songs']):,}")
print(f"Simultaneous launch songs (>=3 countries in week 1): {int(simul_stats['simul_launch_songs']):,} ({simul_stats['simul_launch_share']*100:.2f}%)")

# Sample snapshots calculation across 20 sample weeks distributed across 2018-2026 to measure exact candidate row sizes
sample_weeks = con.execute(f"""
    SELECT DISTINCT week_end 
    FROM read_parquet('{parquet_glob}')
    WHERE week_end >= '2018-11-01' AND week_end <= '2026-09-17'
    ORDER BY week_end
""").fetchdf()["week_end"].tolist()

total_snapshots = len(sample_weeks)
print(f"\nTotal snapshot weeks (2018-11-01 to 2026-09-17): {total_snapshots}")

# Measure on 10 evenly spaced snapshot dates
test_dates = [sample_weeks[int(i * (total_snapshots - 1) / 9)] for i in range(10)]
print(f"Testing 10 representative snapshot dates: {[str(d)[:10] for d in test_dates]}")

results = []
for d in test_dates:
    d_str = str(d)[:10]
    # Active songs at d: appeared in [d - 28 days, d]
    row = con.execute(f"""
        WITH 
        active_4w AS (
            SELECT DISTINCT key 
            FROM read_parquet('{parquet_glob}')
            WHERE week_end BETWEEN DATE '{d_str}' - INTERVAL 28 DAYS AND DATE '{d_str}'
        ),
        active_current AS (
            SELECT DISTINCT key
            FROM read_parquet('{parquet_glob}')
            WHERE week_end = DATE '{d_str}'
        ),
        active_aged26 AS (
            SELECT a.key
            FROM active_current a
            JOIN song_first_appearance f ON a.key = f.key
            WHERE f.first_week >= DATE '{d_str}' - INTERVAL 182 DAYS
        ),
        -- Prior charting in target country up to snapshot d
        charted_ever AS (
            SELECT DISTINCT key, cc 
            FROM read_parquet('{parquet_glob}')
            WHERE week_end <= DATE '{d_str}'
        ),
        -- Entries in next 4 weeks: (d, d + 28 days]
        future_entries AS (
            SELECT DISTINCT key, cc 
            FROM read_parquet('{parquet_glob}')
            WHERE week_end > DATE '{d_str}' AND week_end <= DATE '{d_str}' + INTERVAL 28 DAYS
        ),
        all_countries AS (
            SELECT DISTINCT cc FROM read_parquet('{parquet_glob}')
        )
        SELECT 
            COUNT(DISTINCT a4.key) as active_4w_cnt,
            COUNT(DISTINCT ac.key) as active_curr_cnt,
            COUNT(DISTINCT a26.key) as active_26w_cnt,
            -- Candidates under current-week active:
            COUNT(DISTINCT ac.key) * (SELECT COUNT(*) FROM all_countries) - 
                (SELECT COUNT(*) FROM charted_ever ce JOIN active_current ac2 ON ce.key = ac2.key) as candidates_curr,
            -- Positives under current-week active:
            (SELECT COUNT(*) 
             FROM future_entries fe 
             JOIN active_current ac2 ON fe.key = ac2.key 
             WHERE NOT EXISTS (SELECT 1 FROM charted_ever ce WHERE ce.key = fe.key AND ce.cc = fe.cc)) as positives_curr,
            -- Candidates under aged <= 26w:
            COUNT(DISTINCT a26.key) * (SELECT COUNT(*) FROM all_countries) - 
                (SELECT COUNT(*) FROM charted_ever ce JOIN active_aged26 a26_2 ON ce.key = a26_2.key) as candidates_26w,
            -- Positives under aged <= 26w:
            (SELECT COUNT(*) 
             FROM future_entries fe 
             JOIN active_aged26 a26_2 ON fe.key = a26_2.key 
             WHERE NOT EXISTS (SELECT 1 FROM charted_ever ce WHERE ce.key = fe.key AND ce.cc = fe.cc)) as positives_26w
        FROM active_4w a4
        LEFT JOIN active_current ac ON a4.key = ac.key
        LEFT JOIN active_aged26 a26 ON a4.key = a26.key
    """).fetchdf().iloc[0].to_dict()
    results.append(row)

res_df = pd.DataFrame(results)
print("\n--- Averages Across Representative Snapshots ---")
print(f"Mean active songs (last 4w): {res_df['active_4w_cnt'].mean():.1f}")
print(f"Mean active songs (current week): {res_df['active_curr_cnt'].mean():.1f}")
print(f"Mean active songs (current week & age <= 26w): {res_df['active_26w_cnt'].mean():.1f}")

total_cands_curr = res_df['candidates_curr'].mean() * total_snapshots
total_pos_curr = res_df['positives_curr'].mean() * total_snapshots
pos_rate_curr = (total_pos_curr / total_cands_curr) * 100

total_cands_26w = res_df['candidates_26w'].mean() * total_snapshots
total_pos_26w = res_df['positives_26w'].mean() * total_snapshots
pos_rate_26w = (total_pos_26w / total_cands_26w) * 100

print(f"\n[Option A: Active at snapshot week t, unentered country]")
print(f"  Estimated total candidate rows: {total_cands_curr:,.0f} (~{total_cands_curr/1e6:.1f}M rows)")
print(f"  Estimated total positive entries: {total_pos_curr:,.0f}")
print(f"  Estimated positive rate: {pos_rate_curr:.3f}% (1 in {100/pos_rate_curr:.0f})")

print(f"\n[Option B: Active at snapshot week t, song age <= 26 weeks]")
print(f"  Estimated total candidate rows: {total_cands_26w:,.0f} (~{total_cands_26w/1e6:.1f}M rows)")
print(f"  Estimated total positive entries: {total_pos_26w:,.0f} (retains {total_pos_26w/total_pos_curr*100:.1f}% of all breakouts!)")
print(f"  Estimated positive rate: {pos_rate_26w:.3f}% (1 in {100/pos_rate_26w:.0f})")
