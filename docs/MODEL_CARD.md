# Model card — Crossfade next-market model

- **Type.** LightGBM gradient-boosted trees, binary objective, 156 trees (rounds = mean of the walk-forward early-stopping points). File: `ml/artifacts/crossfade_lgbm.txt`.
- **Output.** Probability that a song charting now enters a given national YouTube Top-100 chart for the first time within 4 weeks.
- **Intended use.** Helping labels, distributors and managers working African repertoire decide which rising songs get local effort, and in which markets, each week.
- **Not for.**
  - Predicting chart position, streams or revenue.
  - Judging artists' worth.
  - Causal claims about marketing impact.
  - Markets outside the 43 charts.

## Training data
- **Rows.** YouTube weekly charts from 1 Nov 2018 to 5 Dec 2024. That's 1,247,460 training rows: every positive (54,518) plus 5% of negatives, weighted ×20.
- **Features (29).** Song momentum, corridor strength, regional/language proximity, target-market openness and the artist's track record in the target. All are computed point-in-time.

## Performance (held-out 2025–26)
- **Ranking.** Hit@1 61.0%, Recall@10 89.9%, PR-AUC 0.138. Full tables are in `docs/EVALUATION.md`.
- **Calibration.** ECE is 0.019 percentage points. In the top decile it predicts 2.05% and 2.15% is observed.

## Explanations
Exact TreeSHAP (`pred_contrib`). The demo turns the top positive contributions into plain-language reasons. A reason is only shown when the underlying value supports it; for example, "artist has charted here before" requires at least one past song.

## Known limitations and bias
- **Single platform.** YouTube charts only.
- **Missing countries.** 43 countries, with most African, Middle Eastern and South/South-East Asian markets missing.
- **Popularity bias.** Artists with a long history get stronger evidence. The emerging-only slice is reported, and artist history is shown as an explicit reason rather than hidden.
- **Feedback loops.** If many users follow the same corridors, the tool can reinforce existing flows. It shows probabilities and evidence, not instructions.
- **Song identity.** Normalisation can occasionally merge different songs that share a title and artist, or miss renamed uploads.

## Data licence
Zenodo record 23644526, CC-BY 4.0. Attribution is in `docs/DATA_NOTES.md` and in the demo footer.
