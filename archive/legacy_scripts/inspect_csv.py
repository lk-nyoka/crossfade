import pandas as pd, sys, os
path = r'C:\\Users\\ASUS\\.gemini\\antigravity\\scratch\\crossfade\\data\\raw\\youtube_top100_positions_2026.csv.gz'
print('Reading preview...')
try:
    df = pd.read_csv(path, nrows=5)
    print('Columns:', df.columns.tolist())
    print(df.head())
except Exception as e:
    print('Error:', e)
