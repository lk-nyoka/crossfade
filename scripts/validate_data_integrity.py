import os, json, hashlib, requests, sys

# Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')
REPORT_JSON = os.path.join(BASE_DIR, 'ml', 'config', 'data_coverage.json')
REPORT_MD = os.path.join(BASE_DIR, 'docs', 'DATA_QUALITY.md')
MANIFEST_URL = 'https://zenodo.org/api/records/23644526/files'

def md5_file(path, chunk_size=8192):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        while True:
            data = f.read(chunk_size)
            if not data:
                break
            h.update(data)
    return h.hexdigest()

def fetch_manifest():
    resp = requests.get(MANIFEST_URL)
    resp.raise_for_status()
    data = resp.json()
    entries = data.get('entries', [])
    manifest = {}
    for e in entries:
        key = e['key']
        checksum = e.get('checksum', '')
        # checksum format: md5:abcdef...
        if checksum.startswith('md5:'):
            checksum = checksum.split(':', 1)[1]
        manifest[key] = {
            'size': e.get('size'),
            'checksum': checksum,
            'mimetype': e.get('mimetype')
        }
    return manifest

def validate_files():
    manifest = fetch_manifest()
    results = []
    for fname in os.listdir(RAW_DIR):
        if not fname.endswith('.csv.gz'):
            continue
        fpath = os.path.join(RAW_DIR, fname)
        size = os.path.getsize(fpath)
        md5 = md5_file(fpath)
        entry = manifest.get(fname, {})
        expected_md5 = entry.get('checksum')
        expected_size = entry.get('size')
        ok = (expected_md5 == md5) and (expected_size == size)
        results.append({
            'file': fname,
            'size_bytes': size,
            'md5': md5,
            'expected_md5': expected_md5,
            'expected_size': expected_size,
            'match': ok,
            'exists_in_manifest': fname in manifest
        })
    return results

def write_reports(results):
    # JSON
    os.makedirs(os.path.dirname(REPORT_JSON), exist_ok=True)
    with open(REPORT_JSON, 'w') as jf:
        json.dump(results, jf, indent=2)
    # Markdown
    lines = ['# Data Integrity Report', '']
    total = len(results)
    matched = sum(1 for r in results if r['match'])
    lines.append(f'Total files checked: **{total}**')
    lines.append(f'Files matching manifest (size+md5): **{matched}**')
    lines.append('')
    for r in results:
        status = 'OK' if r['match'] else 'MISMATCH'
        lines.append(f"- {r['file']}: {status}")
        if not r['match']:
            lines.append(f"  - Expected size: {r['expected_size']}, actual: {r['size_bytes']}")
            lines.append(f"  - Expected md5: {r['expected_md5']}, actual: {r['md5']}")
    with open(REPORT_MD, 'w') as mf:
        mf.write('\n'.join(lines))

if __name__ == '__main__':
    res = validate_files()
    write_reports(res)
    print('Integrity validation completed. Reports written.')
