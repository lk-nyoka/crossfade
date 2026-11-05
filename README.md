# Crossfade

**Get ahead of where a rising song breaks next.** Crossfade is a next-market radar for African labels and distributors. It reads how songs have travelled between 43 national YouTube music charts since 2018 and ranks the countries each rising song is most likely to enter in the next four weeks. Every pick comes with a calibrated probability and checkable evidence, so local playlist pitching, PR and content go where momentum is already heading.

Built for **DataNova 2026 Global Data-thon** · Problem statement **DN-OP-01 Open Innovation** · Solo entry by Lindokuhle Nyoka.

## Why it matters

A label or distributor can work a handful of markets a week, not forty-three. Songs travel along corridors (Nigeria → Tanzania / Uganda / Kenya, UK → Ireland → the Nordics, Brazil → Portugal), and those corridors show up in public chart history. Crossfade turns that history into a weekly decision: *which songs, in which countries, deserve local effort this week?*

## Results (held-out test: Jan 2025 – Sep 2026, never seen during training or tuning)

| Metric | Crossfade | Best baseline | Random |
|---|---|---|---|
| **Weekly slate:** share of each week's top-10 song–market picks that entered within 4 weeks | **41.2%** | 13.3% | 0.2% |
| Hit@1: the #1 pick was entered within 4 weeks | **61.0%** | 57.9% | 7.9% |
| Recall@10: share of real next markets in the top 10 of 43 | **89.9%** | 87.6% | 28.8% |
| PR-AUC: spotting *which* songs travel (base rate 0.24%) | **0.138** | 0.039 | 0.002 |
| Recall@5 for songs that debuted in an African chart | **83.8%** | 77.8% | — |
| Expected calibration error | 0.019 pp | — | — |

- **Consistent across years.** Recall@5 vs. the best baseline in walk-forward validation: 2022 +2.9 pts, 2023 +3.0 pts, 2024 +0.7 pts.
- **Exact explanations.** TreeSHAP explanations for every prediction.
- **Not just global launches.** The weekly-slate result holds on emerging songs only (40.7% vs 18.6%) and when each song gets just one pick (34.9% vs 13.2%).
- **Full numbers.** Results are in `ml/reports/benchmark.csv` and `ml/reports/decision_value.csv`, and the method is in `docs/EVALUATION.md`.

## What's in the demo (`apps/web/dist/index.html`)

- **Live radar.** Rising songs on the latest chart week (15 Oct 2026), each with its 8 most likely next markets and the evidence behind each pick. You can filter to songs charting in Africa.
- **Backtest.** 60 songs from the held-out weeks, showing predictions next to what actually happened.
- **Corridors.** For any country, where its songs go next.
- **Model & method.** Benchmark tables, calibration, the main drivers and the limits.

## Quick start

```bash
pip install -r requirements.txt
python scripts/download_data.py            # Zenodo CSVs (MD5-verified)
python scripts/convert_to_parquet.py       # -> data/processed/youtube/year=*/
python -m ml.pipelines.build_base          # canonical songs + clean chart table
python -m ml.pipelines.build_dataset       # 30.2M candidates, labels, 29 point-in-time features (~2 min)
python -m ml.models.train                  # baselines + LightGBM, walk-forward, held-out test (~15 min, 7 GB RAM)
python -m ml.evaluation.slices             # Africa / region slices
python -m ml.evaluation.decision_value     # weekly top-K slate vs baselines
python -m ml.export_demo                   # JSON for the web demo
python apps/web/build.py                   # single-file demo -> apps/web/dist/index.html
python scripts/quality_gates.py            # gates T0-T6 must all pass
```

`python -m ml.models.finish` regenerates predictions, SHAP and the run summary from the saved model without retraining.

## Hosting the demo for free

The demo is one self-contained HTML file with no server. Push the repo to GitHub, then go to **Settings → Pages**, deploy from the `main` branch, and point it at `/apps/web/dist`. You can also drag `apps/web/dist/index.html` onto Netlify Drop.

## Project structure

```text
├── apps/web/            src/index.html (UI), data/crossfade.json (export), build.py, dist/index.html (deployable)
├── data/
│   ├── raw/             Zenodo CSV.GZ (git-ignored)
│   ├── processed/       year-partitioned Parquet (git-ignored)
│   ├── interim/         canonical songs, dataset, predictions (git-ignored, rebuilt by the pipeline)
│   └── reference/       country_meta.csv (region, sub-region, language)
├── docs/                PRD, problem definition, evaluation, model card, data notes & audit, quality gates
├── ml/
│   ├── config/          config.yaml (single source of truth for dates, horizon, splits)
│   ├── pipelines/       song_identity.py, build_base.py, build_dataset.py
│   ├── models/          train.py, finish.py
│   ├── evaluation/      metrics.py, slices.py
│   ├── artifacts/       crossfade_lgbm.txt (trained model)
│   └── reports/         benchmark.csv, calibration_test.csv, shap_importance.csv, slices_test.csv, run_summary.json
├── scripts/             data download/audit/conversion, quality_gates.py
├── tests/               song identity + leakage tests
└── archive/             superseded scripts and reports from the first iteration (kept for history, not used)
```

## Data

YouTube weekly Top-100 charts, 43 countries + global, 3 May 2018 – 15 Oct 2026, 1,944,951 rows. Zenodo record [23644526](https://zenodo.org/records/23644526), CC-BY 4.0. See `docs/DATA_NOTES.md` and `docs/DATA_QUALITY.md`.

## Limits

- **YouTube only.** Streaming, radio and short-video data aren't included.
- **Coverage.** The data has 43 charts, so most of Africa and Asia is missing.
- **Correlational.** It shows where similar songs went next, not what a campaign will cause.
- **Popularity bias.** Established artists carry more history, so emerging-only results are reported separately.
- **Visibility, not causation.** A chart entry is a visibility signal, not proof of a lasting audience, and the model predicts organic spread rather than what promotion would add. Measuring that uplift with a distributor partner is the next step.
