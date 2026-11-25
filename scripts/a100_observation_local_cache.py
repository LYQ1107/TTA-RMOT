"""Copy only the fixed historical feature bytes away from long NAS mmaps.

No image dataset, checkpoint, teacher request or GT is copied. A buffered
source read is checked against a separate local SHA read before the local
file becomes immutable. NAS originals remain the authoritative assets.
"""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json
from flex_uda.a100.mechanism import PAIRS, HISTORY, metadata, generation_guard
from scripts.a100_six_direction import OUT as SIX, PUBLIC

CACHE = SIX/'historical_observation_feature_cache_20261123'
STATE = CACHE/'progress.json'
FILES = ('metadata.json', 'observations.npy', 'index.npy', 'pcd.npy')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        while chunk := f.read(8*1024*1024): h.update(chunk)
    return h.hexdigest()


def copy_one(relative):
    source, destination = HISTORY/relative, CACHE/relative
    receipt_path = destination.with_name(destination.name+'.copy_receipt.json')
    before = source.stat(); destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and receipt_path.is_file():
        old = json.loads(receipt_path.read_text())
        if (destination.stat().st_size == old['bytes'] == before.st_size and
                old['source_mtime_ns'] == before.st_mtime_ns and digest(destination) == old['sha256']):
            return dict(old, reused_verified_local_copy=True)
    temp = destination.with_name(destination.name+'.copying'); h = hashlib.sha256(); copied = 0
    with source.open('rb') as incoming, temp.open('wb') as outgoing:
        while chunk := incoming.read(8*1024*1024):
            outgoing.write(chunk); h.update(chunk); copied += len(chunk)
        outgoing.flush(); os.fsync(outgoing.fileno())
    after = source.stat()
    assert copied == before.st_size == after.st_size and before.st_mtime_ns == after.st_mtime_ns, relative
    assert digest(temp) == h.hexdigest(), relative
    temp.chmod(0o444); temp.replace(destination)
    row = dict(relative_path=str(relative), bytes=copied, sha256=h.hexdigest(),
        source_mtime_ns=before.st_mtime_ns, source_read_and_independent_local_read_agree=True)
    atomic_json(receipt_path, row); return row


def main():
    guard = generation_guard(); started = time.perf_counter(); relative = []
    for _, (model, target) in PAIRS.items():
        for video in sorted(metadata(target)['videos']):
            relative.extend(Path('features')/target/model/'videos'/video/name for name in FILES)
    assert len(relative) == len(set(relative)) == 592
    expected = sum((HISTORY/p).stat().st_size for p in relative)
    CACHE.mkdir(parents=True, exist_ok=True)
    stat = os.statvfs(CACHE); assert stat.f_bavail*stat.f_frsize > expected*1.1
    records = []; last = 0.
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(copy_one, p):p for p in relative}
        for future in concurrent.futures.as_completed(futures):
            records.append(future.result()); elapsed = time.perf_counter()-started
            if len(records)%16 == 0 or elapsed-last > 15 or len(records) == 592:
                value = dict(status='COPYING_EXACT_HISTORICAL_FEATURE_BYTES_NO_GT', pid=os.getpid(),
                    completed=len(records), total=592, verified_bytes=sum(r['bytes'] for r in records),
                    expected_bytes=expected, wall_seconds=elapsed, manual_GT_open_attempts=len(guard))
                atomic_json(STATE, value); print(json.dumps(value), flush=True); last = elapsed
    assert not guard and len(records) == 592 and sum(r['bytes'] for r in records) == expected
    records.sort(key=lambda r:r['relative_path'])
    aggregate = hashlib.sha256(json.dumps([(r['relative_path'], r['bytes'], r['sha256']) for r in records],
        separators=(',', ':')).encode()).hexdigest()
    atomic_json(CACHE/'manifest.json', dict(files=records, aggregate_sha256=aggregate, not_to_Git=True))
    receipt = dict(status='PASS_BYTE_IDENTICAL_HISTORICAL_OBSERVATION_LOCAL_CACHE',
        files=592, setting_videos=148, bytes=expected, aggregate_sha256=aggregate,
        source_read_hash_equals_separate_local_hash_for_every_file=True,
        source_size_and_mtime_stable_during_each_copy=True, local_cached_data_files_read_only=True,
        NAS_sources_not_modified=True, no_images_weights_teacher_or_GT_copied=True,
        original_feature_version_not_recomputed=True, numerical_learning_recipes_unchanged=True,
        preparation_not_single_method_cold_cost=True, manual_GT_open_attempts=0,
        wall_seconds=time.perf_counter()-started, goal_status='ACTIVE')
    atomic_json(CACHE/'completion.json', receipt)
    atomic_json(PUBLIC/'historical_observation_local_cache_receipt.json', receipt)
    atomic_json(STATE, dict(receipt, pid=os.getpid(), completed=592, total=592))
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__': main()
