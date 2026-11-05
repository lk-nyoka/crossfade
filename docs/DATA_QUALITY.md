# Crossfade Full Historical Data Audit (2018–2026)

- **Audit Date**: `2026-10-27T17:11:22.495163+00:00`
- **Coverage Window**: `2018-05-03 00:00:00` to `2026-10-15 00:00:00`
- **Note on 2026**: Partial year coverage (concludes at `2026-10-15`).
- **Total Rows Verified**: `1,944,951`
- **Invalid Ranks (<1 or >100)**: `0`
- **Missing Release Dates**: `11,714` (0.60%)

## 1. Annual Inventory & Raw-to-Parquet Reconciliation

| Year | Raw CSV.GZ | Raw Size | Parquet Artifact | Parquet Size | Rows | Status |
|---|---|---|---|---|---|---|
| `2018` | `youtube_top100_positions_2018.csv.gz` | 5.16 MB | `data\processed\youtube\year=2018\youtube_top100_positions_2018.parquet` | 2.70 MB | 154,010 | Verified |
| `2019` | `youtube_top100_positions_2019.csv.gz` | 7.98 MB | `data\processed\youtube\year=2019\youtube_top100_positions_2019.parquet` | 4.01 MB | 228,817 | Verified |
| `2020` | `youtube_top100_positions_2020.csv.gz` | 8.25 MB | `data\processed\youtube\year=2020\youtube_top100_positions_2020.parquet` | 4.14 MB | 233,217 | Verified |
| `2021` | `youtube_top100_positions_2021.csv.gz` | 8.17 MB | `data\processed\youtube\year=2021\youtube_top100_positions_2021.parquet` | 4.11 MB | 228,818 | Verified |
| `2022` | `youtube_top100_positions_2022.csv.gz` | 8.35 MB | `data\processed\youtube\year=2022\youtube_top100_positions_2022.parquet` | 4.14 MB | 228,826 | Verified |
| `2023` | `youtube_top100_positions_2023.csv.gz` | 8.52 MB | `data\processed\youtube\year=2023\youtube_top100_positions_2023.parquet` | 4.16 MB | 228,816 | Verified |
| `2024` | `youtube_top100_positions_2024.csv.gz` | 8.65 MB | `data\processed\youtube\year=2024\youtube_top100_positions_2024.parquet` | 4.25 MB | 228,813 | Verified |
| `2025` | `youtube_top100_positions_2025.csv.gz` | 8.47 MB | `data\processed\youtube\year=2025\youtube_top100_positions_2025.parquet` | 4.25 MB | 228,814 | Verified |
| `2026` | `youtube_top100_positions_2026.csv.gz` | 6.69 MB | `data\processed\youtube\year=2026\youtube_top100_positions_2026.parquet` | 3.49 MB | 184,820 | Verified |

## 2. Country-Week Top-100 Completeness

- **Total Country-Weeks**: `19,448`
- **Country-Weeks with Exactly 100 Rows**: `19,301` (99.24%)
- **Country-Weeks with != 100 Rows**: `147`
- **Countries Observed Per Week**: Min `44`, Max `44`, Median `44`

## 3. Internal Gaps & 4-Week Horizon Breaks

- **Internal Country Gaps (>1 week)**: `0` detected across all countries.
- **Total Country-Weeks Breaking 4-Week Horizon**: `176`
  - **Due to Final 4 Weeks of Dataset (2026-09-24 to 2026-10-15)**: `176`
  - **Due to Internal Weekly Gaps**: `0`

## 4. Song Identity Resolution (`key` Stability)

- **Finding.** The source `key` field isn't unique per song. The same track appears under several keys because of remasters, video tags, featured-artist syntax and case differences.
- **Original unique keys.** 121,256.
- **Canonical song identities after normalisation.** 119,728 (rebuilt 5 Nov 2026 after fixing a crash on missing artist names).
- **Songs merged across multiple keys.** 1,349 (1,528 redundant keys collapsed).
- **Examples.** "Bad Habits" and "Bad Habits (Live)" by Ed Sheeran; "BILLIE JEAN" and "Billie Jean" by Michael Jackson.
