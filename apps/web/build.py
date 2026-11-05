"""Inline the exported JSON into a single self-contained HTML file (apps/web/dist/index.html).
That one file can be hosted free on GitHub Pages, Netlify or Vercel, or opened directly."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
src = (HERE / "src" / "index.html").read_text(encoding="utf-8")
data = (HERE / "data" / "crossfade.json").read_text(encoding="utf-8").replace("</", "<\\/")
out = HERE / "dist" / "index.html"
out.parent.mkdir(exist_ok=True)
out.write_text(src.replace("__DATA__", data, 1), encoding="utf-8")
print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB)")
