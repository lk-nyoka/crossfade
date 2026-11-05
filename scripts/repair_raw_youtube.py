import os
import sys
import gzip
import hashlib
import json
import requests
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

ZENODO_RECORD_URL = "https://zenodo.org/api/records/23644526"

def compute_md5(filepath: Path) -> str:
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

def test_gzip_validity(filepath: Path) -> bool:
    try:
        with gzip.open(filepath, "rb") as gz:
            for _ in iter(lambda: gz.read(1024 * 1024), b""):
                pass
        return True
    except Exception as e:
        print(f"Gzip integrity check failed for {filepath}: {e}")
        return False

def print_raw_directory_ls():
    print("\n--- Current data/raw listing ---")
    files = sorted(RAW_DIR.glob("*"))
    for f in files:
        size_bytes = f.stat().st_size
        size_mb = size_bytes / (1024 * 1024)
        print(f"{f.name:<40} {size_bytes:>10} bytes ({size_mb:>6.2f} MB)")
    print("--------------------------------\n")

def main():
    print(f"Fetching Zenodo record metadata from {ZENODO_RECORD_URL}...")
    resp = requests.get(ZENODO_RECORD_URL, timeout=30)
    resp.raise_for_status()
    record = resp.json()
    files_manifest = {f["key"]: f for f in record.get("files", [])}

    print(f"Retrieved manifest with {len(files_manifest)} files.\n")

    # 1. Print all Zenodo files
    print(f"{'Filename':<40} {'Size (MB)':<12} {'Checksum'}")
    print("-" * 80)
    for key, f in sorted(files_manifest.items()):
        size_mb = f['size'] / (1024 * 1024)
        print(f"{key:<40} {size_mb:<12.2f} {f['checksum']}")
    print("-" * 80)

    # 2. Verify existing 2020, 2022, 2026
    existing_years = ["2020", "2022", "2026"]
    print("\nVerifying existing files (2020, 2022, 2026)...")
    for yr in existing_years:
        fname = f"youtube_top100_positions_{yr}.csv.gz"
        fpath = RAW_DIR / fname
        if not fpath.exists():
            print(f"ERROR: Expected existing file missing: {fname}")
            continue
        expected_md5 = files_manifest[fname]["checksum"].replace("md5:", "")
        expected_size = files_manifest[fname]["size"]
        actual_md5 = compute_md5(fpath)
        actual_size = fpath.stat().st_size
        gz_ok = test_gzip_validity(fpath)
        
        md5_match = (actual_md5 == expected_md5)
        size_match = (actual_size == expected_size)
        print(f"[{fname}] Size Match: {size_match} ({actual_size}/{expected_size}) | MD5 Match: {md5_match} ({actual_md5}) | Gzip Valid: {gz_ok}")
        if not (md5_match and size_match and gz_ok):
            print(f"CORRUPT FILE DETECTED: {fname}. Will re-download.")
            fpath.unlink()

    # 3. Download missing files: 2021, 2023, 2024, 2025 (priority), then 2018, 2019
    download_queue = ["2021", "2023", "2024", "2025", "2018", "2019"]
    
    for yr in download_queue:
        fname = f"youtube_top100_positions_{yr}.csv.gz"
        fpath = RAW_DIR / fname
        part_path = RAW_DIR / f"{fname}.part"

        if fpath.exists():
            print(f"File {fname} already exists. Verifying...")
            expected_md5 = files_manifest[fname]["checksum"].replace("md5:", "")
            if compute_md5(fpath) == expected_md5 and test_gzip_validity(fpath):
                print(f"File {fname} verified. Skipping download.")
                continue
            else:
                print(f"Existing {fname} failed verification. Removing and re-downloading.")
                fpath.unlink()

        f_meta = files_manifest.get(fname)
        if not f_meta:
            print(f"ERROR: {fname} not found in Zenodo manifest!")
            continue

        download_url = f_meta["links"].get("content") or f_meta["links"].get("self")
        expected_md5 = f_meta["checksum"].replace("md5:", "")
        expected_size = f_meta["size"]

        print(f"\nDownloading {fname} ({expected_size / (1024*1024):.2f} MB) from {download_url}...")
        success = False
        for attempt in range(1, 4):
            try:
                if part_path.exists():
                    part_path.unlink()

                with requests.get(download_url, stream=True, timeout=60) as r:
                    r.raise_for_status()
                    with open(part_path, "wb") as pf:
                        for chunk in r.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                pf.write(chunk)

                # Validate downloaded .part file
                part_size = part_path.stat().st_size
                part_md5 = compute_md5(part_path)
                part_gz_ok = test_gzip_validity(part_path)

                if part_size == expected_size and part_md5 == expected_md5 and part_gz_ok:
                    # Atomic rename
                    part_path.replace(fpath)
                    print(f"SUCCESS: {fname} downloaded and verified (MD5: {part_md5}).")
                    print_raw_directory_ls()
                    success = True
                    break
                else:
                    print(f"Attempt {attempt} verification failed: Size={part_size}/{expected_size}, MD5={part_md5} (exp {expected_md5}), Gzip={part_gz_ok}")
            except Exception as e:
                print(f"Attempt {attempt} error downloading {fname}: {e}")

        if not success:
            print(f"FATAL: Failed to download and verify {fname} after 3 attempts.")
            sys.exit(1)

    print("\nAll target YouTube Top-100 files downloaded and verified!")
    print_raw_directory_ls()

if __name__ == "__main__":
    main()
