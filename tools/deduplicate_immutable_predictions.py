"""Share identical sealed research outputs without deleting any result path.

Only MOT prediction files named in a prediction seal and their local mask/score
arrays are eligible. NAS, checkpoints, raw QA and unsealed cold jobs are left
alone. Every replacement is verified byte-identical and performed atomically.
The experiment contract already forbids rewriting sealed predictions.
"""
import argparse
import errno
import hashlib
import json
import os
from pathlib import Path
import time


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def run(root, apply):
    begin = time.perf_counter();seen = {};verified = set();visited = set()
    counts = dict(sealed_methods=0, files=0, replacements=0,
                  already_shared=0, physical_bytes_released=0, equivalent_bytes_shared=0,
                  hardlink_limit_reference_rotations=0)
    def consider(path, expected=None):
        path = Path(path)
        if path in visited or path.is_symlink():
            return
        path.relative_to(root)
        visited.add(path);stat = path.stat();counts['files'] += 1
        digest = expected if expected is not None else sha(path)
        key = (stat.st_dev, stat.st_size, digest)
        if key not in seen:
            seen[key] = path
            return
        reference = seen[key];source_stat = reference.stat()
        if (source_stat.st_dev, source_stat.st_ino) == (stat.st_dev, stat.st_ino):
            counts['already_shared'] += 1
            return
        source_inode = (source_stat.st_dev, source_stat.st_ino)
        if source_inode not in verified:
            assert sha(reference) == digest, reference
            verified.add(source_inode)
        if expected is not None:
            assert sha(path) == digest, path
        if apply:
            temporary = path.with_name(path.name + '.dedup-' + str(os.getpid()))
            try:
                os.link(reference, temporary)
            except OSError as error:
                if error.errno != errno.EMLINK:
                    raise
                # Empty MOT outputs can exceed the filesystem's per-inode
                # link limit. This already verified identical file starts a
                # new reference group; no output content needs to change.
                seen[key] = path
                counts['hardlink_limit_reference_rotations'] += 1
                return
            os.replace(temporary, path)
            if stat.st_nlink == 1:
                counts['physical_bytes_released'] += stat.st_blocks * 512
        counts['equivalent_bytes_shared'] += stat.st_size
        counts['replacements'] += 1

    # Completed legacy studies have family/pair/method depth. Avoid walking
    # hundreds of thousands of MOT directories and current deeper cold jobs.
    for seal_path in sorted(root.glob('*/*/*/predictions_seal.json')):
        seal = json.loads(seal_path.read_text());counts['sealed_methods'] += 1
        for entry in seal['episodes']:
            consider(entry['prediction_path'], entry['prediction_sha256'])
        for path in sorted((seal_path.parent / 'videos').glob('*/*.npy')):
            if path.name in ('mask.npy', 'scores.npy'):
                consider(path)
        if counts['sealed_methods'] % 50 == 0:
            print(json.dumps(dict(event='IMMUTABLE_DEDUP_PROGRESS', **counts)), flush=True)
    result = dict(status='BYTE_IDENTICAL_HARDLINKS_APPLIED' if apply else 'DRY_RUN', **counts,
                  root=str(root), wall_seconds=time.perf_counter()-begin,
                  prediction_paths_preserved=True, sealed_hashes_unchanged=True,
                  NAS_and_checkpoints_and_raw_QA_untouched=True)
    (root / 'immutable_dedup_receipt.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    run(args.root.absolute(), args.apply)
