import sys
from pathlib import Path
import duckdb

BASE_DIR = Path(r"C:\Users\ASUS\Desktop\Coding Projects\Build or Die\Week 3\Hackathon\Crossfade")
parquet_glob = (BASE_DIR / "data" / "processed" / "youtube" / "year=*" / "*.parquet").as_posix()
con = duckdb.connect()

print("Calculating launch timing of positive entries...")
timing_df = con.execute(f"""
    WITH song_first AS (
        SELECT key, MIN(week_end) as song_first_week
        FROM read_parquet('{parquet_glob}')
        GROUP BY key
    ),
    country_first AS (
        SELECT key, cc, MIN(week_end) as country_first_week
        FROM read_parquet('{parquet_glob}')
        GROUP BY key, cc
    ),
    entry_diffs AS (
        SELECT 
            c.key, 
            c.cc, 
            c.country_first_week, 
            s.song_first_week,
            DATE_DIFF('day', s.song_first_week, c.country_first_week) as days_since_launch
        FROM country_first c
        JOIN song_first s ON c.key = s.key
    )
    SELECT 
        COUNT(*) as total_first_entries,
        SUM(CASE WHEN days_since_launch = 0 THEN 1 ELSE 0 END) as launch_week_entries,
        SUM(CASE WHEN days_since_launch > 0 THEN 1 ELSE 0 END) as subsequent_breakouts,
        SUM(CASE WHEN days_since_launch BETWEEN 1 AND 7 THEN 1 ELSE 0 END) as week1_breakouts,
        SUM(CASE WHEN days_since_launch BETWEEN 1 AND 28 THEN 1 ELSE 0 END) as month1_breakouts
    FROM entry_diffs
""").fetchdf()

row = timing_df.iloc[0].to_dict()
total = int(row["total_first_entries"])
launch = int(row["launch_week_entries"])
subseq = int(row["subsequent_breakouts"])
w1 = int(row["week1_breakouts"])
m1 = int(row["month1_breakouts"])

print(f"Total country first-entry events: {total:,}")
print(f"Entries occurring at week 0 (initial launch): {launch:,} ({launch/total*100:.2f}%)")
print(f"Subsequent market breakout events (after launch week): {subseq:,} ({subseq/total*100:.2f}%)")
print(f"Subsequent breakouts occurring within 1 week of initial launch: {w1:,} ({w1/subseq*100:.2f}% of subsequent breakouts)")
print(f"Subsequent breakouts occurring within 4 weeks of initial launch: {m1:,} ({m1/subseq*100:.2f}% of subsequent breakouts)")
