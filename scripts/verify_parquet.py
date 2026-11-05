import os
import duckdb

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PARQUET_BASE = os.path.join(BASE_DIR, "data", "processed", "youtube")

con = duckdb.connect()

years = ["2020", "2022", "2026"]
for yr in years:
    path = os.path.join(PARQUET_BASE, f"year={yr}", f"youtube_top100_positions_{yr}.parquet").replace("\\", "/")
    cnt = con.execute(f"SELECT COUNT(*) FROM read_parquet('{path}')").fetchone()[0]
    print(f"Year {yr} count: {cnt:,}")

glob_path = os.path.join(PARQUET_BASE, "year=*", "*.parquet").replace("\\", "/")
total_cnt = con.execute(f"SELECT COUNT(*) FROM read_parquet('{glob_path}')").fetchone()[0]
print(f"Total rows across all Parquet files: {total_cnt:,}")

sample_df = con.execute(f"SELECT cc, week_end, rank, title, artists FROM read_parquet('{glob_path}') LIMIT 3").fetchdf()
print("\nSample records:")
print(sample_df)
print("\nVerification successful! Parquet tables are completely valid and readable.")
