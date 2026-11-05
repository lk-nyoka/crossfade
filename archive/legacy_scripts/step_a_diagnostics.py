import re
import sys
from pathlib import Path
from collections import defaultdict
import duckdb
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = Path(r"C:\Users\ASUS\Desktop\Coding Projects\Build or Die\Week 3\Hackathon\Crossfade")
sys.path.insert(0, str(BASE_DIR))

from ml.pipelines.song_identity import normalize_song_identity

parquet_glob = (BASE_DIR / "data" / "processed" / "youtube" / "year=*" / "*.parquet").as_posix()
con = duckdb.connect()

print("=" * 70)
print("STEP A: PHASE 0 RECONCILIATION & DIAGNOSTICS")
print("=" * 70)

# --- 1. Key Pattern Analysis ---
print("\n--- 1. Keys Matching vs Not Matching 11-Char YouTube Pattern ---")
keys_df = con.execute(f"SELECT DISTINCT key FROM read_parquet('{parquet_glob}')").fetchdf()
yt_pattern = re.compile(r"^[a-zA-Z0-9_-]{11}$")
non_matching = keys_df[~keys_df["key"].str.match(yt_pattern, na=False)]

print(f"Total distinct keys: {len(keys_df):,}")
print(f"Matching 11-char pattern: {len(keys_df) - len(non_matching):,}")
print(f"Non-matching keys count: {len(non_matching):,}")

# Breakdown of non-matching prefixes/patterns
non_match_types = defaultdict(list)
for k in non_matching["key"]:
    if ":" in k:
        prefix = k.split(":")[0]
        non_match_types[f"Colon prefix '{prefix}:'"].append(k)
    else:
        non_match_types[f"Other (length {len(k)})"].append(k)

for n_type, k_list in non_match_types.items():
    print(f"  {n_type}: {len(k_list):,} keys (e.g. {k_list[:3]})")

# --- 2. Country-Weeks != 100 Rows & Rank Ties ---
print("\n--- 2. Analysis of Country-Weeks with Row Count != 100 ---")
cw_df = con.execute(f"""
    SELECT cc, CAST(week_end AS DATE) as week_end, COUNT(*) as cnt 
    FROM read_parquet('{parquet_glob}') 
    GROUP BY cc, week_end 
    HAVING cnt != 100
""").fetchdf()

print(f"Total non-100 country-weeks: {len(cw_df)}")
print("Distribution of row counts:")
print(cw_df["cnt"].value_counts().to_string())

# Verify rank ties
ties_df = con.execute(f"""
    WITH ranked AS (
        SELECT cc, CAST(week_end AS DATE) as week_end, rank, COUNT(*) as rank_cnt
        FROM read_parquet('{parquet_glob}')
        GROUP BY cc, week_end, rank
        HAVING COUNT(*) > 1
    )
    SELECT cc, week_end, rank, rank_cnt
    FROM ranked
""").fetchdf()
print(f"\nTotal instances of tied ranks across dataset: {len(ties_df)}")
print(f"Distinct country-weeks with tied ranks: {len(ties_df.groupby(['cc', 'week_end']))}")
print(f"Every one of the {len(cw_df)} non-100 country-weeks has exact rank ties matching the extra row count: {len(ties_df.groupby(['cc', 'week_end'])) == len(cw_df)}")
print("Sample tied ranks:")
print(ties_df.head(5).to_string())

# --- 3. Song Identity Discrepancy & Granular Category Breakdown ---
print("\n--- 3. Reconciliation of Key Counts & Canonical Songs ---")
df_all = con.execute(f"""
    SELECT DISTINCT key, title, artists, cc 
    FROM read_parquet('{parquet_glob}')
""").fetchdf()

# Primary key-to-canonical mapping using most frequent metadata per key
key_meta = con.execute(f"""
    WITH counts AS (
        SELECT key, title, artists, COUNT(*) as obs
        FROM read_parquet('{parquet_glob}')
        GROUP BY key, title, artists
    ),
    ranked AS (
        SELECT key, title, artists,
               ROW_NUMBER() OVER (PARTITION BY key ORDER BY obs DESC) as rn
        FROM counts
    )
    SELECT key, title, artists FROM ranked WHERE rn = 1
""").fetchdf()

print(f"Primary key metadata rows (1 per distinct key): {len(key_meta):,}")

# Test variants breakdown
# (a) feat/case/whitespace
# (b) live or remaster variants
# (c) different artist sets
# (d) cross-country presence of merged keys

canon_groups = defaultdict(list)
for _, r in key_meta.iterrows():
    cid, norm_title, norm_artists = normalize_song_identity(r["title"], r["artists"])
    canon_groups[cid].append({
        "key": r["key"],
        "title": r["title"] or "",
        "artists": r["artists"] or "",
        "norm_title": norm_title,
        "norm_artists": norm_artists
    })

total_canonical = len(canon_groups)
multikey_groups = {cid: members for cid, members in canon_groups.items() if len(members) > 1}

print(f"Distinct keys: {len(key_meta):,}")
print(f"Canonical songs (with current normalizer): {total_canonical:,}")
print(f"Canonical songs with >1 key: {len(multikey_groups):,}")
total_keys_in_multikey = sum(len(m) for m in multikey_groups.values())
net_keys_collapsed = total_keys_in_multikey - len(multikey_groups)
print(f"Total keys across multikey songs: {total_keys_in_multikey:,}")
print(f"Net redundant keys collapsed: {net_keys_collapsed:,}")
print(f"Check math: Distinct Keys ({len(key_meta):,}) - Redundant Keys ({net_keys_collapsed:,}) = {len(key_meta) - net_keys_collapsed:,} == Canonical Songs ({total_canonical:,}) -> {len(key_meta) - net_keys_collapsed == total_canonical}")

# Granular category breakdown
live_remaster_regex = re.compile(r"\b(live|remaster|remastered|acoustic|remix|version|deluxe)\b", re.IGNORECASE)
feat_regex = re.compile(r"\b(feat|ft|featuring)\b", re.IGNORECASE)

cat_counts = {
    "live_or_remaster_variant": 0,
    "feat_case_whitespace_only": 0,
    "different_artist_sets": 0
}

diff_country_count = 0

# Check country distribution per key
key_countries = df_all.groupby("key")["cc"].apply(set).to_dict()

for cid, members in multikey_groups.items():
    titles = [m["title"] for m in members]
    artists = [m["artists"] for m in members]
    norm_artists_list = [m["norm_artists"] for m in members]
    
    # Check if keys appear in different countries
    keys = [m["key"] for m in members]
    group_country_sets = [key_countries.get(k, set()) for k in keys]
    # Are the country sets strictly non-identical or distinct?
    has_diff_countries = any(s1 != s2 for s1 in group_country_sets for s2 in group_country_sets)
    if has_diff_countries:
        diff_country_count += 1
        
    has_live_remaster = any(live_remaster_regex.search(t) for t in titles)
    has_feat = any(feat_regex.search(t) for t in titles)
    diff_norm_artists = len(set(norm_artists_list)) > 1
    
    if has_live_remaster:
        cat_counts["live_or_remaster_variant"] += 1
    elif diff_norm_artists:
        cat_counts["different_artist_sets"] += 1
    else:
        cat_counts["feat_case_whitespace_only"] += 1

print("\n--- Breakdown of the Merged Groups ---")
for cat, cnt in cat_counts.items():
    print(f"  {cat}: {cnt:,} groups ({cnt / len(multikey_groups) * 100:.1f}%)")

print(f"\nMerged groups with keys appearing in DIFFERENT countries: {diff_country_count:,} / {len(multikey_groups):,} ({diff_country_count / len(multikey_groups) * 100:.1f}%)")
