# Problem definition & experimental design

## Decision question
*For this rising song, which markets is it most likely to enter in the next 4 weeks, and what evidence supports each pick?*

## Protocol (source of truth: `ml/config/config.yaml`)

| Parameter | Value | Why |
|---|---|---|
| Prediction unit | (snapshot week *t*, song *s*, target country *c*) | One question per market the song could still reach |
| Markets | 43 national YouTube Top-100 charts; `global` is a feature, never a target | The global chart is an aggregate, not a market you can target |
| Active song | Charting in ≥1 national chart at *t*, ≤ 26 weeks since national debut | Focus on rising songs, not catalogue |
| Eligible target | *c* never entered by *s* at any week ≤ *t* | "Next market" means a new market |
| Label | 1 if *s* first enters *c* in (*t*, *t* + 28 days] | Four weeks is a realistic campaign window |
| Snapshots | 1 Nov 2018 (26-week burn-in) – 17 Sep 2026 (last with a full 4-week outcome) + live week 15 Oct 2026 | No truncated labels |
| Coordinated launch | Debut in ≥ 3 national charts in the same week (9,852 songs, 8.3%) | Kept in the data, reported as its own stratum |

## Size of the problem
- **Songs.** 119,728 canonical songs (from 121,256 YouTube keys).
- **Candidate rows.** 30,241,019, with 69,610 positives, a **0.230% base rate**.
- **Song-weeks.** 747,119.

## Splits (no random splits; time only)

```
2018-11-01 ─ 2021-12-02  train ─┐ purge ≥4 wk ─► validate 2022
2018-11-01 ─ 2022-12-01  train ─┐ purge ≥4 wk ─► validate 2023
2018-11-01 ─ 2023-11-30  train ─┐ purge ≥4 wk ─► validate 2024
2018-11-01 ─ 2024-12-05  final train ─► TEST 2025-01-01 ─ 2026-09-17 (scored once)
```

## Leakage controls
1. **Point-in-time features.** Every feature is computed from data dated ≤ *t*.
2. **Corridor rates.** Country-to-country rates P(a→c) are re-estimated every quarter. They use only (song, market) entries whose 4-week outcome had fully resolved before the quarter began, smoothed toward the target market's prior (α = 20).
3. **Artist history.** The artist's record in *c* counts other songs that entered *c* on or before *t* (as-of join).
4. **Automated tests.** `tests/test_leakage.py` recomputes labels, eligibility and market counts from the raw charts and checks there are no training rows in the test period.

## Corrections vs. the first iteration
- **Base rate.** The first diagnostics reported an 8.58% positive rate and a 683% "share of positives". Both were bugs from counting entries that had already happened. The true rate is 0.230%.
- **Baseline benchmark.** The old benchmark scored a different candidate set than the one defined above: it had no age filter, used raw keys, and set the global chart as a target. It was replaced by a benchmark where every model is scored on the same rows.
- **Pasted results.** The old diagnostics script contained pasted (not computed) results. It's archived in `archive/legacy_scripts/`.
