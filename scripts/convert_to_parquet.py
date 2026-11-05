import os
import duckdb
from pathlib import Path

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')
PROCESSED_YT_DIR = os.path.join(BASE_DIR, 'data', 'processed', 'youtube')

os.makedirs(PROCESSED_YT_DIR, exist_ok=True)

con = duckdb.connect()

# List available YouTube CSV.GZ files
yt_files = sorted([p for p in Path(RAW_DIR).glob('youtube_top100_positions_*.csv.gz')])
if not yt_files:
    raise RuntimeError('No YouTube CSV files found in raw folder.')

for f in yt_files:
    year = f.name.split('_')[-1].split('.')[0]  # e.g. 2020
    out_dir = os.path.join(PROCESSED_YT_DIR, f"year={year}")
    os.makedirs(out_dir, exist_ok=True)
    parquet_path = os.path.join(out_dir, f"youtube_top100_positions_{year}.parquet")
    # DuckDB COPY from CSV (gzip) to Parquet directly
    con.execute(f"COPY (SELECT * FROM read_csv_auto('{f.as_posix()}', compression='gzip')) TO '{parquet_path}' (FORMAT 'parquet')")
    print(f"Converted {f.name} -> {parquet_path}")

print('All available YouTube files have been converted to Parquet (year-partitioned).')
