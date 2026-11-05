# Data Notes and Provenance

## Source Dataset

- **Title**: Historical Music Chart Positions (YouTube Top 100 & Spotify Top 200)
- **Repository**: Zenodo
- **Record Identifier**: `zenodo.23644526`
- **DOI / URL**: [https://zenodo.org/records/23644526](https://zenodo.org/records/23644526)
- **License**: [Creative Commons Attribution 4.0 International (CC-BY 4.0)](https://creativecommons.org/licenses/by/4.0/)

## Citation

```bibtex
@dataset{zenodo_23644526_music_charts,
  title        = {Historical Music Chart Positions: YouTube Top 100 and Spotify Top 200 (2018–2026)},
  publisher    = {Zenodo},
  year         = {2026},
  url          = {https://zenodo.org/records/23644526}
}
```

## Dataset Composition & Verified Status

1. **Source Formats**:
   - Gzipped CSV files by platform and year (`youtube_top100_positions_YYYY.csv.gz`).
   - Downloaded using exact Zenodo API v2 record manifest links and verified with official MD5 checksums and gzip integrity checks.

2. **Processed Storage**:
   - Converted using DuckDB streaming into columnar year-partitioned Parquet files:
     ```
     data/processed/youtube/year=YYYY/youtube_top100_positions_YYYY.parquet
     ```

3. **Current Verified Coverage (YouTube Top 100)**:
   - **Years Processed**: 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026 (all 9 years verified).
   - **Total Observations**: 1,944,951 weekly chart rows across 44 countries and global charts.
   - **Continuous Weekly Observation Window**: May 3, 2018 through October 15, 2026 (442 consecutive weekly periods).
   - **Partial Year Note**: 2018 begins on May 3, 2018; 2026 concludes at October 15, 2026.
   - **Internal Weekly Gaps**: 0 gaps inside any country across the 442 weeks.

4. **Song identity normalisation** (`ml/pipelines/song_identity.py`, applied in `ml/pipelines/build_base.py`):
   - The source `key` (a YouTube video ID) has **121,256** unique values. The same song appears under several keys: remasters, live versions, official vs. lyric videos, and featured-artist spelling.
   - After normalisation there are **119,728** canonical songs. 1,349 songs were merged from 1,528 redundant keys.
   - Keys with no usable title keep their own identity, so empty metadata never merges unrelated songs.
   - Earlier drafts quoted 120,011 songs. That run crashed on missing artist names, and the fix (5 Nov 2026) changed the count.

5. **Country reference** (`data/reference/country_meta.csv`): region, sub-region and primary language for each chart, used for proximity features.
