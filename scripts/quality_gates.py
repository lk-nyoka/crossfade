"""Quality gates T0-T6 (SkillFoundry-style). Exit code 0 only if every gate passes.
Usage: python scripts/quality_gates.py"""
import json
import py_compile
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LIVE_CODE = [p for p in ROOT.rglob("*.py") if "archive" not in p.parts and ".venv" not in p.parts]
results = []


def gate(name, ok, detail=""):
    results.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))


# T0 correctness: model beats the best baseline on the headline metrics, in test and every fold
b = pd.read_csv(ROOT / "ml/reports/benchmark.csv")
fails = []
for split in ["fold_2022", "fold_2023", "fold_2024", "test_all", "test_emerging"]:
    s = b[b.split == split].set_index("model")
    for m in ["hit@1", "recall@10", "pr_auc"]:
        if s.loc["Crossfade_LGBM", m] <= s.drop("Crossfade_LGBM")[m].max():
            fails.append(f"{split}:{m}")
gate("T0 correctness", not fails, "model beats best baseline everywhere" if not fails else ", ".join(fails))

# T1 banned patterns
bad = []
for p in LIVE_CODE:
    t = p.read_text(encoding="utf-8", errors="ignore")
    if p.name != "quality_gates.py" and (re.search(r'[A-Za-z]:\\\\?Users\\\\?', t) or "/Users/" in t):
        bad.append(f"{p.relative_to(ROOT)}: absolute path")
    if p.name != "quality_gates.py" and re.search(r"['\"]Recall@\d+['\"]\s*:\s*['\"][\d.]+%", t):
        bad.append(f"{p.relative_to(ROOT)}: pasted metric literal")
gate("T1 banned patterns", not bad, "; ".join(bad))

# T2 compile
errs = []
for p in LIVE_CODE:
    try:
        py_compile.compile(str(p), doraise=True)
    except py_compile.PyCompileError as e:
        errs.append(str(e))
gate("T2 compile", not errs, f"{len(LIVE_CODE)} files" if not errs else errs[0])

# T3 tests
r = subprocess.run([sys.executable, "-m", "pytest", "-q", str(ROOT / "tests")], cwd=ROOT, capture_output=True, text=True)
gate("T3 tests", r.returncode == 0, r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-200:])

# T4 security
sec = []
for p in LIVE_CODE + list((ROOT / "apps/web/src").glob("*.html")):
    t = p.read_text(encoding="utf-8", errors="ignore")
    if re.search(r"(api[_-]?key|secret|token|password)\s*=\s*['\"][^'\"]{8,}", t, re.I):
        sec.append(f"{p.name}: possible secret")
    if p.suffix == ".py" and p.name != "quality_gates.py" and re.search(r"\b(eval|exec)\(", t):
        sec.append(f"{p.name}: eval/exec")
html = (ROOT / "apps/web/src/index.html").read_text(encoding="utf-8")
if "esc(s.title)" not in html or "esc(artists(" not in html:
    sec.append("demo renders song text without escaping")
gate("T4 security", not sec, "; ".join(sec))

# T5 build + numbers trace to reports
dist = ROOT / "apps/web/dist/index.html"
data = json.loads((ROOT / "apps/web/data/crossfade.json").read_text())
t = b[b.split == "test_all"].set_index("model")
ours = t.loc["Crossfade_LGBM"]
readme = (ROOT / "README.md").read_text(encoding="utf-8")
dvv = pd.read_csv(ROOT / "ml/reports/decision_value.csv")
slate = dvv[(dvv.roster == "all rising songs") & (dvv.k_per_week == 10) & (dvv.model == "Crossfade_LGBM")].precision.iloc[0]
need = [f"{ours['hit@1']*100:.1f}%", f"{ours['recall@10']*100:.1f}%", f"{ours['pr_auc']:.3f}", f"{slate*100:.1f}%"]
missing = [n for n in need if n not in readme]
ok5 = dist.exists() and len(data["radar"]) > 0 and not missing
gate("T5 build", ok5, f"dist {dist.stat().st_size/1e6:.2f} MB, {len(data['radar'])} radar songs" if ok5 else f"missing in README: {missing}")

# T6 scope / consistency
docs = {p.name: p.read_text(encoding="utf-8") for p in (ROOT / "docs").glob("*.md")}
STALE = re.compile(r"8\.58%|683%|85\.04%|120,011")
MARKERS = ("bug", "earlier", "first iteration", "stale", "corrections")  # lines that explain the old numbers
stale = sorted({n for n, txt in docs.items() for line in txt.splitlines()
                if STALE.search(line) and not any(m in line.lower() for m in MARKERS)})
gate("T6 scope", not stale, "docs consistent" if not stale else f"stale numbers in {stale}")

print(f"\n{sum(ok for _, ok, _ in results)}/{len(results)} gates passed")
sys.exit(0 if all(ok for _, ok, _ in results) else 1)
