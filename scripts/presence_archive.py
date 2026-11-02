"""Archive this task's small experiment logs; never include data/weights.

Run once after the corresponding coordinators have returned. The large
features, labels, predictions and checkpoints remain in their recorded paths.
"""
import json
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flex_uda.presence.runtime import RUN, PUBLIC
from flex_tta.runtime.pf_receipts import atomic_json, file_sha


def log_files():
    files = set()
    for name in ('launchers', 'priority_export', 'rescue_launchers', 'head_pipeline',
                 'source_lr_trial', 'source_lr_eval', 'source_label_fix_v2', 'source_label_fix_v2_eval', 'finish_A', 'pipeline'):
        base = RUN / name
        if base.exists():
            files.update(p for p in base.rglob('*') if p.is_file() and p.suffix in ('.txt', '.log', '.json', '.jsonl'))
    for name in ('heads', 'heads_source_lr1e5', 'heads_source_labels_v2'):
        files.update(p for p in (RUN / name).rglob('*') if p.is_file() and p.name in ('curve.jsonl', 'training_receipt.json'))
    files.update((RUN / 'resources').glob('*.json'))
    files.update((RUN / 'features').rglob('export_receipt.json'))
    files.update((RUN / 'features').rglob('smoke_receipt.json'))
    for name in ('stage_A', 'stage_A_source_lr1e5', 'stage_A_label_v2', 'stage_A_source_lr1e5_label_v2'):
        base = RUN / name
        files.update(p for p in base.rglob('*') if p.is_file() and
                     (p.name in ('decode_receipt.json', 'official_completion.json', 'offline_summary.json',
                                 'pedestrian_detailed.csv', 'pedestrian_summary.txt') or
                      p.suffix == '.log' or (p.parent.name == 'trackeval_attempts' and p.suffix == '.log')))
    return sorted(files)


def main():
    paths = log_files()
    records = [dict(path=str(p), archive_path=str(p.relative_to(RUN)),
                    bytes=p.stat().st_size, sha256=file_sha(p)) for p in paths]
    destination = PUBLIC / 'experiment_logs.tar.gz'
    temporary = destination.with_suffix('.partial')
    with tarfile.open(temporary, 'w:gz', compresslevel=6) as archive:
        for path, record in zip(paths, records):
            archive.add(path, arcname=record['archive_path'], recursive=False)
    temporary.replace(destination)
    # A changed log means its owner has not finished; do not call that a final
    # archive. This is one consistency check, not a background polling loop.
    for path, record in zip(paths, records):
        if file_sha(path) != record['sha256']:
            raise RuntimeError('Log changed during final packaging: '+str(path))
    result = dict(archive_path=str(destination), bytes=destination.stat().st_size,
        sha256=file_sha(destination), files=records,
        large_files_excluded='features/GT labels/images/predictions/checkpoints; paths and SHA in stage-A manifests',
        logs_include_failed_attempts=True, all_paths_scoped_to_this_task=str(RUN))
    atomic_json(PUBLIC / 'experiment_logs.json', result)
    print(json.dumps({k:v for k,v in result.items() if k != 'files'}, indent=2), flush=True)


if __name__ == '__main__': main()
