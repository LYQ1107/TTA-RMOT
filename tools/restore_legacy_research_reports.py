"""Restore byte-identical archived narrative reports without overwriting edits."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT/'docs/legacy_archives/research_reports_20261127/manifest.json'


def restore(output_root):
    output_root = output_root.resolve()
    manifest = json.loads(MANIFEST.read_text())
    rows = []
    plans = []
    for entry in manifest['reports']:
        relative = PurePosixPath(entry['original_path'])
        assert not relative.is_absolute() and '..' not in relative.parts
        assert relative.parts[0] == 'docs' and relative.suffix in ('.txt', '.md')
        ref = entry['original_published_ref']
        assert re.fullmatch('[0-9a-f]{40}', ref)
        data = subprocess.check_output(['git','show',ref+':'+str(relative)], cwd=ROOT)
        assert len(data) == entry['original_bytes']
        assert hashlib.sha256(data).hexdigest() == entry['original_sha256']
        target = (output_root/entry['original_path']).resolve()
        assert target.is_relative_to(output_root)
        if target.exists():
            if target.read_bytes() != data:
                raise RuntimeError('Existing report has different bytes; left unchanged: '+str(target))
        else:
            plans.append((target,data))
        rows.append(str(target))
    # Verify every source and collision before creating any destination.
    for target,data in plans:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            stream.write(data)
        assert target.read_bytes() == data
    return dict(status='PASS_BYTE_IDENTICAL_RESEARCH_REPORT_RESTORATION', reports=len(rows), paths=rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-root', type=Path, default=ROOT)
    args = parser.parse_args()
    print(json.dumps(restore(args.output_root), ensure_ascii=False))
