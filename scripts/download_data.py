"""Download the YouTube Top-100 yearly files (2018-2026) from Zenodo record 23644526 and verify MD5 checksums.
Usage: python scripts/download_data.py"""
import hashlib
import os

import requests

RECORD = "https://zenodo.org/api/records/23644526/files"
YEARS = range(2018, 2027)
TARGET_FILES = [f"youtube_top100_positions_{y}.csv.gz" for y in YEARS]
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "raw")


def manifest():
    """Return {filename: md5} from the Zenodo files API."""
    r = requests.get(RECORD, timeout=60)
    r.raise_for_status()
    entries = r.json().get("entries", [])
    return {e["key"]: e.get("checksum", "").replace("md5:", "") for e in entries}


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download():
    os.makedirs(OUT_DIR, exist_ok=True)
    sums = manifest()
    for fname in TARGET_FILES:
        if fname not in sums:
            raise SystemExit(f"{fname} is not in the Zenodo record")
        out_path = os.path.join(OUT_DIR, fname)
        if os.path.exists(out_path) and md5(out_path) == sums[fname]:
            print(f"ok (cached)  {fname}")
            continue
        print(f"downloading  {fname}")
        with requests.get(f"{RECORD}/{fname}/content", stream=True, timeout=300) as r:
            r.raise_for_status()
            with open(out_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
        got = md5(out_path)
        if got != sums[fname]:
            raise SystemExit(f"MD5 mismatch for {fname}: {got} != {sums[fname]}")
        print(f"ok (md5)     {fname}")


if __name__ == "__main__":
    download()
