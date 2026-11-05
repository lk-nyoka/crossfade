"""Central paths. Everything is relative to the repo root, so the project runs on any machine."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PARQUET_GLOB = (DATA / "processed" / "youtube" / "year=*" / "*.parquet").as_posix()
COUNTRY_META = DATA / "reference" / "country_meta.csv"
INTERIM = DATA / "interim"
CONFIG = ROOT / "ml" / "config" / "config.yaml"
REPORTS = ROOT / "ml" / "reports"
MODELS = ROOT / "ml" / "artifacts"
WEB_DATA = ROOT / "apps" / "web" / "data"
for p in (INTERIM, REPORTS, MODELS):
    p.mkdir(parents=True, exist_ok=True)
