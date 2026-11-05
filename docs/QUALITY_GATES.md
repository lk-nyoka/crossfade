# Quality gates

Run `python scripts/quality_gates.py`. The script exits non-zero if any gate fails. The gates follow the SkillFoundry "Anvil" pattern (T0–T6), adapted to a data-science project.

| Gate | What it checks | Why |
|---|---|---|
| T0 Correctness | Crossfade beats the best baseline on Hit@1, Recall@10 and PR-AUC in every validation fold, the full test and the emerging-only test | The model must earn its complexity |
| T1 Banned patterns | No absolute machine paths (`C:\Users\…`, `/Users/…`); no pasted metric literals in code | The first iteration had both |
| T2 Compile | Every live `.py` compiles | Basic hygiene |
| T3 Tests | `pytest tests/`: song identity (incl. missing values) and leakage tests | Labels/features must match raw data |
| T4 Security | No secrets or tokens, no `eval`/`exec`, demo escapes all text it renders | The demo renders third-party song titles |
| T5 Build | Demo JSON and single-file HTML exist; README headline numbers match `ml/reports/benchmark.csv` | Numbers can't drift from the source |
| T6 Scope | No stale numbers from the first iteration (8.58%, 683%, 85.04%, 120,011) left in `docs/` | One story, one set of numbers |
