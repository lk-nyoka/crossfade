import os, json
import pandas as pd

# Define base directories
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')
DOCS_DIR = os.path.join(BASE_DIR, 'docs')
ML_CONFIG_DIR = os.path.join(BASE_DIR, 'ml', 'config')

os.makedirs(DOCS_DIR, exist_ok=True)
os.makedirs(ML_CONFIG_DIR, exist_ok=True)

REPORT_MD = os.path.join(DOCS_DIR, 'data_coverage_report.md')
JSON_OUT = os.path.join(ML_CONFIG_DIR, 'data_coverage.json')

# Collect CSV.GZ files
files = [f for f in os.listdir(RAW_DIR) if f.endswith('.csv.gz')]
records = []
for fname in sorted(files):
    fpath = os.path.join(RAW_DIR, fname)
    try:
        # Read a few rows to infer columns
        df = pd.read_csv(fpath, compression='gzip', nrows=5)
        cols = list(df.columns)
    except Exception as e:
        cols = []
    records.append({
        'file': fname,
        'size_bytes': os.path.getsize(fpath),
        'columns': cols
    })

summary = {
    'total_files': len(files),
    'total_size_mb': round(sum(r['size_bytes'] for r in records) / 1e6, 2),
    'files': records
}

# Write JSON report
with open(JSON_OUT, 'w') as jf:
    json.dump(summary, jf, indent=2)

# Write Markdown report
lines = [
    '# Data Coverage Report',
    '',
    f'- Total files: {summary["total_files"]}',
    f'- Total size: {summary["total_size_mb"]} MB',
    '',
    '## Files',
    ''
]
for r in records:
    size_mb = round(r['size_bytes'] / 1e6, 2)
    cols = ', '.join(r['columns']) if r['columns'] else 'N/A'
    lines.append(f"- **{r['file']}**: {size_mb} MB, columns: {cols}")

with open(REPORT_MD, 'w') as mf:
    mf.write('\n'.join(lines))

print('Data coverage audit completed. Reports written to:')
print(REPORT_MD)
print(JSON_OUT)
