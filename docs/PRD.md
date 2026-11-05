# Crossfade — Product Requirements Document

*Format follows the SkillFoundry method: requirements → stories with acceptance criteria → quality gates → Definition of Done.*

**Version** 2.1 · 5 Nov 2026 · DataNova 2026 Global Data-thon · Problem statement **DN-OP-01 Open Innovation** · Solo entry

---

## 1. Problem

A label or distributor working African repertoire can push a rising song in two or three countries, not forty-three. They pick those countries on gut feel, on where the big labels go, or not at all. The cost of a wrong pick is real: ad spend, playlist pitching, a radio push or a trip in a market where the song was never going to land, while the market it *was* about to break in goes unworked.

Songs do not spread at random. They travel along corridors (Nigeria → Tanzania, Uganda, Kenya; the UK → Ireland → the Nordics; Brazil → Portugal), and those corridors are visible in eight years of public chart data. Nobody packages that evidence for the people who need it most.

**Decision question.** *For this rising song, which markets is it most likely to enter in the next 4 weeks, how confident should I be, and what evidence supports each pick?*

## 2. Users

| User | Need | What they do with Crossfade |
|---|---|---|
| Distributor marketing / label services team (primary) | Decide each week which songs get local effort, and where | Works the weekly slate: pitches local DSP editors and playlists, briefs local PR |
| Label A&R / marketing | Prioritise a roster across markets | Flags songs about to travel and plans local-language content or features |
| Artist manager of a song that has just charted | Know where to go next | Picks 2–3 markets for the next month, including show booking |

Songs need to be charting somewhere before Crossfade can help, so its users are teams around artists who have already broken out somewhere, not unknown artists.

## 3. Goals and non-goals

**Goals**
- G1. Rank all not-yet-reached markets for a rising song by probability of a first chart entry within 4 weeks.
- G2. Beat transparent rule-based baselines on held-out data the model never saw, with honest confidence intervals.
- G3. Give calibrated probabilities (a "20%" should happen about 20% of the time) so expected value can be reasoned about.
- G4. Explain every pick with evidence a non-technical user can check (corridor rate, artist history, momentum).
- G5. Run on free tools only: open data, Python, a static web page.

**Non-goals**
- Predicting chart *position* or streams. Causal claims about what a campaign will do. Markets outside the 43 charts in the data.

## 4. Data

YouTube weekly Top-100 music charts, 43 national charts + a global chart, 3 May 2018 – 15 Oct 2026, 1.94M rows (Zenodo record 23644526, CC-BY 4.0). Verified by MD5 checksums, gzip integrity checks and a full audit (`docs/DATA_QUALITY.md`). No personal data.

## 5. User stories and acceptance criteria

**S1 — Canonical songs.** As an analyst I need one identity per song so the same track under several YouTube videos is not counted as several songs.
- AC1: Remaster / "(Live)" / "[Official Video]" / featured-artist spelling variants map to one ID (unit tests).
- AC2: Missing titles or artists never crash the pipeline and never merge unrelated songs (unit test).
- Result: 121,256 keys → 119,728 songs; 1,349 songs merged from 1,528 redundant keys.

**S2 — Leakage-safe prediction set.** As a judge I need to trust that no feature peeks at the future.
- AC1: Prediction unit = (week *t*, song, target country); target is a national chart the song has never entered up to *t*; the global chart is never a target.
- AC2: Label = first entry into that chart in (*t*, *t* + 28 days].
- AC3: Every feature uses data dated ≤ *t*. Corridor rates are re-estimated quarterly from outcomes fully observed before the quarter starts.
- AC4: Automated tests recompute labels and features from raw charts and must match (`tests/test_leakage.py`).
- Result: 30,241,019 candidate rows, 69,610 positives (0.230%), 747,119 song-weeks.

**S3 — Better than baselines, honestly measured.** As a judge I need evidence the model adds value over simple rules.
- AC1: Six baselines (random, market popularity, strongest corridor, combined corridors, cultural proximity, logistic blend) scored on the *same* rows as the model.
- AC2: Walk-forward validation (2022, 2023, 2024) with a ≥4-week purge; the 2025–26 test set is scored once.
- AC3: Report Hit@1/3, Recall@5/10, NDCG@10, MRR, PR-AUC, bootstrap CIs by week, emerging vs. coordinated-launch strata.
- Result: Crossfade wins on every headline metric in every fold and on the test set (see §7).

**S4 — Explainable picks.** As an artist I need to know *why* a country is recommended.
- AC1: Exact TreeSHAP contributions for every live prediction.
- AC2: Each reason shown is backed by the actual feature value (e.g. "Artist has charted here before (9 songs)"); a reason is suppressed if the value does not support it.

**S5 — Usable demo.** As a user I can open one link, find a song and see its next markets.
- AC1: Live radar for the latest chart week, searchable, with an "Charting in Africa" and "Not a global launch" filter.
- AC2: Backtest view showing predictions against what actually happened.
- AC3: Corridor explorer and a model page with benchmark, calibration and drivers.
- AC4: Single self-contained HTML file; free hosting (GitHub Pages / Netlify / Vercel); works on mobile; no console errors.

## 6. Solution

```
Zenodo CSVs ─► audit + Parquet ─► build_base (canonical songs, clean chart table)
            ─► build_dataset (candidates, labels, 29 point-in-time features)
            ─► train (6 baselines + LightGBM, walk-forward, held-out test, TreeSHAP)
            ─► export_demo ─► apps/web (static radar)
```

Features (29): song momentum (age, markets now, rank and view growth, markets gained), corridor strength from the song's markets to the target (max / noisy-OR / mean / from markets charting now / from markets just entered), regional and language proximity, target-market openness to imports, and the artist's track record in the target market.

## 7. Results (held-out test, Jan 2025 – Sep 2026)

| | Crossfade | Best baseline |
|---|---|---|
| Hit@1 (#1 pick entered) | **61.0%** | 57.9% |
| Recall@10 | **89.9%** | 87.6% |
| PR-AUC (which songs travel) | **0.138** | 0.039 (3.6× lower) |
| Recall@5, songs that debuted in Africa | **83.8%** | 77.8% |
| Weekly slate: top-10 song–market picks that entered within 4 wks | **41.2%** | 13.3% |
| Calibration error (ECE) | 0.019 pp | — |

## 8. Quality gates (must pass before submission)

Run `python scripts/quality_gates.py`. Details in `docs/QUALITY_GATES.md`.

| Gate | Check |
|---|---|
| T0 Correctness | Benchmark reproduces; model beats best baseline on Hit@1, Recall@10 and PR-AUC in test and every fold |
| T1 Banned patterns | No hard-coded machine paths; no pasted metric literals in pipeline code |
| T2 Compile | Every Python file compiles |
| T3 Tests | Unit + leakage tests pass |
| T4 Security | No secrets/tokens, no `eval`/`exec`, demo escapes all data before rendering |
| T5 Build | Demo builds, JSON valid; every headline number in README and the demo traces to `ml/reports/` |
| T6 Scope | Docs agree with each other; superseded artefacts are archived, not mixed with live code |

## 9. Risks and responsible use

- **Popularity bias** — established artists have more history; mitigated by reporting the emerging-only slice and showing artist history as an explicit, visible reason.
- **Self-fulfilling use** — if everyone follows the radar, corridors could harden. The tool shows probabilities and evidence, not commands.
- **Coverage** — 43 charts; most of Africa, the Middle East and South/South-East Asia are missing. Stated on the page.
- **Correlation, not causation** — said plainly in the UI and model card. The model predicts organic spread; measuring what promotion adds (uplift) needs a randomised pilot with a distributor.
- **Chart entry ≠ lasting audience** — a foreign chart entry is a visibility signal. A production version should add repeat-listening data.
- **Data access** — the demo uses an open CC-BY dataset; a commercial version needs licensed chart data (e.g. Chartmetric or Soundcharts APIs), as Next Big Sound's loss of data access in 2015–2021 shows.

## 10. Definition of Done

- [x] Pipeline reproduces from raw data with one command per step, on any machine (no absolute paths)
- [x] All docs quote the same numbers, generated from `ml/reports/`
- [x] Quality gates T0–T6 pass
- [x] Demo builds into one HTML file; checked on desktop and mobile, zero console errors
- [ ] Demo hosted on a public URL (artifact link shared publicly, or GitHub Pages)
- [x] Pitch deck (PPT) drafted; optional 2–3 min video
- [ ] Unstop: register + pay ₹100 by 9 Nov 20:30 SAST, upload by 10 Nov 15:05 SAST
- [ ] Devpost submission before 11 Nov 2026 00:00 SAST
