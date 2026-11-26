"""Restore archived historical PNGs byte-exactly from their published Git ref.

An explicit destination is required. No network, NAS writes, or overwrites of
different files are performed. The retained PDFs and numerical data remain in
the lightweight checkout; this helper restores optional raster exports only.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT/'docs/legacy_archives/legacy_raster_figures_20261124.json'


def restore(manifest, destination):
    source = manifest['immutable_published_restore_commit']
    if not re.fullmatch(r'[0-9a-f]{40}', source):
        raise ValueError('Expected an immutable full Git commit')
    destination = Path(destination).expanduser().resolve()
    nas = (ROOT.parent/'nas').resolve()
    if destination == nas or nas in destination.parents:
        raise ValueError('Choose a local destination outside the read-only NAS')
    plans = []; existing = 0
    for row in manifest['files']:
        relative = PurePosixPath(row['path'])
        if relative.is_absolute() or '..' in relative.parts or relative.parts[0] != 'docs' or relative.suffix != '.png':
            raise ValueError('Unexpected archived raster path')
        target = (destination/relative).resolve()
        if not target.is_relative_to(destination):
            raise ValueError('Destination path escapes through a symlink')
        raw = subprocess.check_output(['git', 'show', source+':'+str(relative)], cwd=ROOT)
        if (len(raw) != row['bytes'] or hashlib.sha256(raw).hexdigest() != row['sha256']
                or not raw.startswith(b'\x89PNG\r\n\x1a\n')):
            raise ValueError('Source bytes differ: '+str(relative))
        if target.exists():
            if not target.is_file() or target.read_bytes() != raw:
                raise FileExistsError('Refusing to overwrite: '+str(target))
            existing += 1
        else:
            plans.append((target, raw))
    # Complete all source/collision checks before creating any destination file.
    for target, raw in plans:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as output:
            output.write(raw)
    return dict(status='PASS_BYTE_EXACT_HISTORICAL_RASTER_RESTORE',
        source_commit=source, files=len(manifest['files']), restored=len(plans),
        already_identical=existing, bytes=sum(r['bytes'] for r in manifest['files']),
        destination=str(destination), different_existing_files_overwritten=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(restore(json.loads(args.manifest.read_text()), args.destination)))
