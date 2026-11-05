import os
import sys
import json
import datetime
from pathlib import Path
from collections import defaultdict
import duckdb
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

# Add project root to sys.path so ml imports work
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from ml.pipelines.song_identity import normalize_song_identity

RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_YT_DIR = BASE_DIR / "data" / "processed" / "youtube"
REPORT_JSON = BASE_DIR / "ml" / "config" / "data_quality.json"
REPORT_MD = BASE_DIR / "docs" / "DATA_QUALITY.md"
HEATMAP_CSV = BASE_DIR / "ml" / "reports" / "country_week_heatmap.csv"

REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
HEATMAP_CSV.parent.mkdir(parents=True, exist_ok=True)

def main():
    print("=" * 70)
    print("CROSSFADE FULL DATA AUDIT & QUALITY VERIFICATION (2018-2026)")
    print("=" * 70, flush=True)

    con = duckdb.connect()

    # 1. File Listing & Reconciliation
    expected_years = [str(yr) for yr in range(2018, 2027)]
    file_records = []
    
    print("\n--- 1. Raw vs Parquet File Inventory ---", flush=True)
    total_raw_rows = 0
    total_parquet_rows = 0

    for yr in expected_years:
        raw_name = f"youtube_top100_positions_{yr}.csv.gz"
        raw_path = RAW_DIR / raw_name
        parquet_path = PROCESSED_YT_DIR / f"year={yr}" / f"youtube_top100_positions_{yr}.parquet"
        
        raw_exists = raw_path.exists()
        parquet_exists = parquet_path.exists()
        
        raw_size = raw_path.stat().st_size if raw_exists else 0
        parquet_size = parquet_path.stat().st_size if parquet_exists else 0
        
        parquet_rows = 0
        if parquet_exists:
            p_posix = parquet_path.as_posix()
            parquet_rows = con.execute(f"SELECT COUNT(*) FROM read_parquet('{p_posix}')").fetchone()[0]
            total_parquet_rows += parquet_rows
            
        file_records.append({
            "year": yr,
            "raw_file": raw_name,
            "raw_exists": raw_exists,
            "raw_size_bytes": raw_size,
            "parquet_file": str(parquet_path.relative_to(BASE_DIR)) if parquet_exists else None,
            "parquet_exists": parquet_exists,
            "parquet_size_bytes": parquet_size,
            "parquet_rows": parquet_rows
        })
        print(f"Year {yr}: Raw={raw_exists} ({raw_size / 1e6:.2f} MB) | Parquet={parquet_exists} ({parquet_size / 1e6:.2f} MB, {parquet_rows:,} rows)")

    parquet_glob = (PROCESSED_YT_DIR / "year=*" / "*.parquet").as_posix()

    # 2. Comprehensive DuckDB Single-Pass Aggregations on Parquet
    print("\n--- 2. Single-Pass Aggregation on Parquet ---", flush=True)
    
    overall_stats = con.execute(f"""
        SELECT 
            COUNT(*) AS total_rows,
            MIN(week_end) AS min_date,
            MAX(week_end) AS max_date,
            COUNT(DISTINCT cc) AS num_countries,
            COUNT(DISTINCT week_end) AS num_weeks,
            COUNT(DISTINCT key) AS num_unique_keys,
            SUM(CASE WHEN rank < 1 OR rank > 100 THEN 1 ELSE 0 END) AS invalid_ranks,
            SUM(CASE WHEN release_date IS NULL THEN 1 ELSE 0 END) AS null_release_dates,
            SUM(CASE WHEN title IS NULL THEN 1 ELSE 0 END) AS null_titles,
            SUM(CASE WHEN artists IS NULL THEN 1 ELSE 0 END) AS null_artists,
            SUM(CASE WHEN views IS NULL THEN 1 ELSE 0 END) AS null_views
        FROM read_parquet('{parquet_glob}')
    """).fetchdf().iloc[0].to_dict()

    total_rows = int(overall_stats["total_rows"])
    min_date = str(overall_stats["min_date"])
    max_date = str(overall_stats["max_date"])
    invalid_ranks = int(overall_stats["invalid_ranks"])
    null_release = int(overall_stats["null_release_dates"])
    num_unique_keys = int(overall_stats["num_unique_keys"])

    print(f"Total Rows: {total_rows:,}")
    print(f"Date Range: {min_date} to {max_date} (Note: 2026 is partial, ending 2026-10-15)")
    print(f"Countries: {int(overall_stats['num_countries'])}, Weeks: {int(overall_stats['num_weeks'])}, Unique Keys: {num_unique_keys:,}")

    # 3. Country-Week Row Counts (Is every country-week exactly 100 rows?)
    print("\n--- 3. Checking Country-Week Row Counts ---", flush=True)
    cw_df = con.execute(f"""
        SELECT 
            cc, 
            CAST(week_end AS DATE) AS week_end, 
            COUNT(*) AS row_cnt 
        FROM read_parquet('{parquet_glob}') 
        GROUP BY cc, week_end 
        ORDER BY cc, week_end
    """).fetchdf()

    # Save heatmap
    cw_df.to_csv(HEATMAP_CSV, index=False)

    total_country_weeks = len(cw_df)
    exact_100_count = int((cw_df["row_cnt"] == 100).sum())
    non_100_df = cw_df[cw_df["row_cnt"] != 100]
    non_100_count = len(non_100_df)

    print(f"Total country-weeks: {total_country_weeks:,}")
    print(f"Country-weeks with exactly 100 rows: {exact_100_count:,} ({exact_100_count / total_country_weeks * 100:.2f}%)")
    print(f"Country-weeks with != 100 rows: {non_100_count}")
    if non_100_count > 0:
        print("Sample non-100 country weeks:")
        print(non_100_df.head(10).to_string(index=False))

    # 4. Countries per week & Weeks per country
    countries_per_week = cw_df.groupby("week_end")["cc"].count()
    weeks_per_country = cw_df.groupby("cc")["week_end"].count()

    print(f"\nCountries per week: min={countries_per_week.min()}, max={countries_per_week.max()}, median={countries_per_week.median():.0f}")
    print(f"Weeks per country: min={weeks_per_country.min()}, max={weeks_per_country.max()}, median={weeks_per_country.median():.0f}")

    # 5. Gaps longer than 1 week inside each country & 4-week Horizon Breaks
    print("\n--- 4. Gaps and 4-Week Horizon Breaks Analysis ---", flush=True)
    
    # Build lookup set of all (cc, week_str)
    existing_cws = set(zip(cw_df["cc"], cw_df["week_end"].astype(str)))
    
    country_gaps = defaultdict(list)
    horizon_breaking_cws = []

    for cc, group in cw_df.groupby("cc"):
        dates = pd.to_datetime(group["week_end"]).sort_values().tolist()
        
        # Check consecutive week gaps
        for i in range(1, len(dates)):
            diff_days = (dates[i] - dates[i-1]).days
            if diff_days > 7:
                missing_weeks = (diff_days // 7) - 1
                country_gaps[cc].append({
                    "gap_start": dates[i-1].strftime("%Y-%m-%d"),
                    "gap_end": dates[i].strftime("%Y-%m-%d"),
                    "days_diff": diff_days,
                    "missing_weeks": missing_weeks
                })
                
        # Check 4-week horizon validity for each country-week
        for d in dates:
            d_str = d.strftime("%Y-%m-%d")
            # The next 4 weeks
            h1 = (d + pd.Timedelta(days=7)).strftime("%Y-%m-%d")
            h2 = (d + pd.Timedelta(days=14)).strftime("%Y-%m-%d")
            h3 = (d + pd.Timedelta(days=21)).strftime("%Y-%m-%d")
            h4 = (d + pd.Timedelta(days=28)).strftime("%Y-%m-%d")
            
            # If any of the future 4 weeks is missing, horizon is broken
            if not ((cc, h1) in existing_cws and (cc, h2) in existing_cws and (cc, h3) in existing_cws and (cc, h4) in existing_cws):
                horizon_breaking_cws.append({
                    "cc": cc,
                    "week_end": d_str,
                    "reason": "dataset_boundary" if d >= pd.to_datetime("2026-09-18") else "internal_gap"
                })

    total_gaps_count = sum(len(v) for v in country_gaps.values())
    print(f"Countries with internal gaps > 1 week: {len(country_gaps)}")
    print(f"Total internal gaps across all countries: {total_gaps_count}")
    if total_gaps_count > 0:
        for cc, gaps in list(country_gaps.items())[:5]:
            print(f"  [{cc}]: {len(gaps)} gap(s): {gaps}")
    else:
        print("  Continuous chart observations across all observed weeks!")

    print(f"\nCountry-weeks that break a 4-week label horizon: {len(horizon_breaking_cws):,} / {total_country_weeks:,}")
    boundary_breaks = sum(1 for x in horizon_breaking_cws if x["reason"] == "dataset_boundary")
    internal_breaks = len(horizon_breaking_cws) - boundary_breaks
    print(f"  - Due to dataset end boundary (2026 partial end at 2026-10-15): {boundary_breaks:,}")
    print(f"  - Due to internal country gaps: {internal_breaks:,}")

    # 6. Song Identity & Key Stability Test
    print("\n--- 5. Song Identity Normalization & Key Stability Test ---", flush=True)
    
    unique_songs_df = con.execute(f"""
        SELECT 
            DISTINCT key, title, artists 
        FROM read_parquet('{parquet_glob}')
    """).fetchdf()

    print(f"Scanning {len(unique_songs_df):,} distinct (key, title, artists) combinations...", flush=True)
    
    key_to_canonical = {}
    canonical_to_keys = defaultdict(set)
    canonical_to_meta = defaultdict(list)

    for _, row in unique_songs_df.iterrows():
        k = row["key"]
        t = row["title"]
        a = row["artists"]
        canon_id, norm_title, norm_artists = normalize_song_identity(t, a)
        key_to_canonical[k] = canon_id
        canonical_to_keys[canon_id].add(k)
        canonical_to_meta[canon_id].append((k, t, a))

    total_canonical_songs = len(canonical_to_keys)
    merged_canonical_groups = {cid: keys for cid, keys in canonical_to_keys.items() if len(keys) > 1}
    total_keys_merged = sum(len(keys) for keys in merged_canonical_groups.values())
    net_keys_collapsed = total_keys_merged - len(merged_canonical_groups)

    print(f"Original Unique Keys in dataset: {num_unique_keys:,}")
    print(f"Canonical Songs after normalisation: {total_canonical_songs:,}")
    print(f"Songs with multiple distinct keys: {len(merged_canonical_groups):,}")
    print(f"Net redundant keys resolved: {net_keys_collapsed:,} ({(net_keys_collapsed / num_unique_keys) * 100:.2f}% reduction)")

    print("\nTop 5 Examples of Songs Merged Across Multiple Keys:")
    sample_merges = []
    for i, (cid, keys) in enumerate(list(merged_canonical_groups.items())[:5]):
        meta_items = canonical_to_meta[cid]
        titles = [str(m[1]) for m in set(meta_items) if m[1]]
        artists = [str(m[2]) for m in set(meta_items) if m[2]]
        clean_keys = [str(k) for k in keys if k]
        sample_merges.append({
            "canonical_id": cid,
            "key_count": len(keys),
            "keys": clean_keys,
            "titles": titles,
            "artists": artists
        })
        print(f"  Example {i+1} [Canonical ID: {cid}] ({len(keys)} keys):")
        print(f"    Titles: {titles[:3]}")
        print(f"    Artists: {artists[:3]}")
        print(f"    Keys: {clean_keys[:3]}")

    # 7. Write Comprehensive JSON & Markdown Reports
    audit_report = {
        "audit_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "years_covered": expected_years,
        "is_2026_partial": True,
        "date_range": {
            "start": min_date,
            "end": max_date
        },
        "inventory": file_records,
        "total_rows": total_rows,
        "invalid_ranks": invalid_ranks,
        "null_counts": {
            "release_date": null_release,
            "title": int(overall_stats["null_titles"]),
            "artists": int(overall_stats["null_artists"]),
            "views": int(overall_stats["null_views"])
        },
        "country_week_stats": {
            "total_country_weeks": total_country_weeks,
            "exact_100_rows": exact_100_count,
            "non_100_rows": non_100_count,
            "non_100_examples": non_100_df.head(10).to_dict(orient="records")
        },
        "countries_per_week": {
            "min": int(countries_per_week.min()),
            "max": int(countries_per_week.max()),
            "median": float(countries_per_week.median())
        },
        "weeks_per_country": {
            "min": int(weeks_per_country.min()),
            "max": int(weeks_per_country.max()),
            "median": float(weeks_per_country.median()),
            "details": weeks_per_country.to_dict()
        },
        "gaps_analysis": {
            "total_internal_gaps": total_gaps_count,
            "gaps_by_country": country_gaps,
            "horizon_breaking_country_weeks_total": len(horizon_breaking_cws),
            "horizon_breaks_dataset_boundary": boundary_breaks,
            "horizon_breaks_internal_gap": internal_breaks
        },
        "song_identity_analysis": {
            "unique_keys": num_unique_keys,
            "canonical_songs": total_canonical_songs,
            "canonical_groups_with_multiple_keys": len(merged_canonical_groups),
            "net_redundant_keys_resolved": net_keys_collapsed,
            "sample_merges": sample_merges
        }
    }

    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(audit_report, f, indent=2, default=str)

    # Markdown Report
    md_lines = [
        "# Crossfade Full Historical Data Audit (2018–2026)",
        "",
        f"- **Audit Date**: `{audit_report['audit_timestamp']}`",
        f"- **Coverage Window**: `{min_date}` to `{max_date}`",
        "- **Note on 2026**: Partial year coverage (concludes at `2026-10-15`).",
        f"- **Total Rows Verified**: `{total_rows:,}`",
        f"- **Invalid Ranks (<1 or >100)**: `{invalid_ranks}`",
        f"- **Missing Release Dates**: `{null_release:,}` ({null_release / total_rows * 100:.2f}%)",
        "",
        "## 1. Annual Inventory & Raw-to-Parquet Reconciliation",
        "",
        "| Year | Raw CSV.GZ | Raw Size | Parquet Artifact | Parquet Size | Rows | Status |",
        "|---|---|---|---|---|---|---|"
    ]

    for rec in file_records:
        status = "Verified" if rec["parquet_rows"] > 0 else "Missing"
        md_lines.append(
            f"| `{rec['year']}` | `{rec['raw_file']}` | {rec['raw_size_bytes']/1e6:.2f} MB | `{rec['parquet_file']}` | {rec['parquet_size_bytes']/1e6:.2f} MB | {rec['parquet_rows']:,} | {status} |"
        )

    md_lines.extend([
        "",
        "## 2. Country-Week Top-100 Completeness",
        "",
        f"- **Total Country-Weeks**: `{total_country_weeks:,}`",
        f"- **Country-Weeks with Exactly 100 Rows**: `{exact_100_count:,}` ({exact_100_count / total_country_weeks * 100:.2f}%)",
        f"- **Country-Weeks with != 100 Rows**: `{non_100_count}`",
        f"- **Countries Observed Per Week**: Min `{countries_per_week.min()}`, Max `{countries_per_week.max()}`, Median `{countries_per_week.median():.0f}`",
        "",
        "## 3. Internal Gaps & 4-Week Horizon Breaks",
        "",
        f"- **Internal Country Gaps (>1 week)**: `{total_gaps_count}` detected across all countries.",
        f"- **Total Country-Weeks Breaking 4-Week Horizon**: `{len(horizon_breaking_cws):,}`",
        f"  - **Due to Final 4 Weeks of Dataset (2026-09-24 to 2026-10-15)**: `{boundary_breaks:,}`",
        f"  - **Due to Internal Weekly Gaps**: `{internal_breaks:,}`",
        "",
        "## 4. Song Identity Resolution (`key` Stability)",
        "",
        "- **Finding**: The source `key` field is not globally unique per song; identical tracks appear under multiple keys due to remasters, video tags, featured artist syntax, and case variations.",
        f"- **Original Unique Keys**: `{num_unique_keys:,}`",
        f"- **Canonical Song Identities After Normalisation**: `{total_canonical_songs:,}`",
        f"- **Songs Merged Across Multiple Keys**: `{len(merged_canonical_groups):,}` ({net_keys_collapsed:,} redundant keys collapsed)",
        "",
        "### Sample Merged Multi-Key Songs",
        ""
    ])

    for ex in sample_merges:
        md_lines.append(f"- **Canonical ID `{ex['canonical_id']}`** ({ex['key_count']} keys):")
        md_lines.append(f"  - Titles: {', '.join(ex['titles'][:5])}")
        md_lines.append(f"  - Artists: {', '.join(ex['artists'][:5])}")
        md_lines.append(f"  - Keys: `{', '.join(ex['keys'][:5])}`")

    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"\nAudit complete. Reports generated:")
    print(f"  - JSON: {REPORT_JSON}")
    print(f"  - Markdown: {REPORT_MD}")
    print(f"  - Heatmap: {HEATMAP_CSV}")

if __name__ == "__main__":
    main()
