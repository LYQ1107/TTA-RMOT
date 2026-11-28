"""Zero-loss, bit-identical fast batch evaluator for official TrackEval.

Eliminates the dominant overhead in evaluation_reuse.py without changing a
single metric formula, matcher, or output byte:
1. Spawns ONE isolated child evaluator process for the whole sealed task batch
   (preserving generation_guard isolation) instead of N separate Python spawns.
2. Parses each target's 54-59MB metadata.json ONCE (extracting only frame_count)
   instead of re-parsing 2,960+ times.
3. Deduplicates identical (pair, seq, prediction_sha256) episodes across all
   tasks in the batch AND batch-queries the shared TrackEval SQLite cache.
4. Evaluates only unique cache-missed episodes with UNMODIFIED vendor
   trackeval.eval.eval_sequence inside a persistent ProcessPoolExecutor
   (default 20 workers on 40-core host, zero nested Pool(2) forks).
5. Bulk-commits new episodes to SQLite in one transaction and writes all
   pedestrian_detailed.csv / official_completion.json files in parallel.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'vendor/FlexHook/TrackEval'))

from flex_tta.runtime.pf_receipts import atomic_json, file_sha
from flex_uda.a100.evaluation_reuse import cache_path, connect, metrics, write_results
from flex_uda.a100.mechanism import KEYS, PAIRS
from flex_uda.presence.predictions import verify_predictions

# Worker-local globals (populated once per persistent worker process)
_W_METRICS = None
_W_MNAMES = None


def _init_worker():
    global _W_METRICS, _W_MNAMES
    import sys as _sys
    if str(ROOT) not in _sys.path:
        _sys.path.insert(0, str(ROOT))
    if str(ROOT / 'vendor/FlexHook/TrackEval') not in _sys.path:
        _sys.path.insert(0, str(ROOT / 'vendor/FlexHook/TrackEval'))
    _W_METRICS = metrics()
    _W_MNAMES = [m.get_name() for m in _W_METRICS]


def _load_frame_counts(targets):
    """Load only (video_id, expression_id) -> frame_count ONCE per target."""
    from scripts.a100_mechanism_offline import gt_root
    frame_counts = {}
    gt_roots = {}
    for target in sorted(targets):
        p = ROOT / 'results/runs/multidomain_generality_20261101' / KEYS[target] / 'metadata.json'
        m = json.loads(p.read_text())
        frame_counts[target] = {(q['video_id'], q['expression_id']): int(q['frame_count'])
                                for q in m['episodes']}
        truth = gt_root(target) / 'gt_template'
        gt_roots[target] = str(truth) if truth.is_dir() else None
    return frame_counts, gt_roots


def _prepare_folder(args):
    """Verify seal, ensure gt.txt symlinks, and extract episode signatures."""
    folder_str, pair, target, fc_map, truth_str = args
    folder = Path(folder_str)
    receipt_path = folder / 'official_completion.json'
    if receipt_path.is_file():
        return dict(folder=folder_str, pair=pair, skipped=True)
    seal = verify_predictions(folder)
    assert seal['GT_open_attempts'] == 0, f'Refusing unsealed/GT-tainted folder: {folder}'
    if truth_str is None:
        assert target == 'KITTI-V2'
        from flex_uda.a100.kitti_gt_restore import recover
        truth = recover(folder) / 'gt_template'
    else:
        truth = Path(truth_str)
    wanted = {(r['video'], r['expression']): r['prediction_sha256'] for r in seal['episodes']}
    seq_info = {}
    signatures = {}
    # Preserve exact metadata m['episodes'] order so pedestrian_detailed.csv is byte-identical
    for (v, e), fc in fc_map.items():
        if (v, e) not in wanted:
            continue
        seq = v + '+' + e
        seq_info[seq] = fc
        signatures[seq] = wanted[(v, e)]
        link = folder / 'results' / v / e / 'gt.txt'
        if not link.exists():
            try:
                link.symlink_to(truth / v / e / 'gt.txt')
            except FileExistsError:
                pass
    assert len(seq_info) == len(wanted)
    return dict(
        folder=folder_str, pair=pair, skipped=False,
        truth=str(truth), seq_info=seq_info, signatures=signatures,
        prediction_rows=int(seal['prediction_rows']),
    )


def _eval_unique_episode(args):
    """Evaluate one unique (pair, seq, pred_sha) using UNMODIFIED vendor TrackEval."""
    pair, seq, pred_sha, folder_str, truth_str, frame_count = args
    import trackeval
    from trackeval.eval import eval_sequence
    cfg = dict(
        GT_FOLDER=truth_str,
        TRACKERS_FOLDER=str(Path(folder_str) / 'results'),
        TRACKERS_TO_EVAL=[str(Path(folder_str) / 'results')],
        SKIP_SPLIT_FOL=True,
        GT_LOC_FORMAT='{gt_folder}/{video_id}/{expression_id}/gt.txt',
        SEQ_INFO={seq: int(frame_count)},
        PRINT_CONFIG=False,
    )
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        dataset = trackeval.datasets.MotChallenge2DBox(cfg)
        val = eval_sequence(seq, dataset, str(Path(folder_str) / 'results'),
                            ['pedestrian'], _W_METRICS, _W_MNAMES)
    return (pair, seq, pred_sha, pickle.dumps(val, protocol=5), folder_str)


def _finalize_folder(args):
    """Write exact TrackEval CSVs and official_completion.json for one folder."""
    meta, res_blobs, references, newly_computed_seqs, cache_file_str = args
    from scripts.active_eval import official_stats
    from scripts.pf_diag_common import official_metrics
    folder = Path(meta['folder'])
    begin = time.perf_counter()
    ordered = {seq: pickle.loads(res_blobs[seq]) for seq in meta['seq_info']}
    with (folder / 'trackeval.log').open('w') as log, \
            contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        write_results(folder, ordered, _W_METRICS)
    stats = official_stats(folder)
    values = official_metrics(folder / 'results')
    assert int(values['Dets']) == meta['prediction_rows']
    result = dict(
        status='PASS',
        pair=meta['pair'],
        metrics=values,
        HOTA_unrounded=stats['HOTA_unrounded'],
        offline_oracle_uses_GT=False,
        wall_seconds=time.perf_counter() - begin,
        prediction_seal_sha256=file_sha(folder / 'predictions_seal.json'),
        detailed_sha256=file_sha(folder / 'results/pedestrian_detailed.csv'),
        official_episode_reuse=dict(
            cached=len(references),
            new=len(newly_computed_seqs),
            same_sequence_and_prediction_sha_required=True,
            cache=cache_file_str,
            references=references,
            matching_and_combination='unmodified vendor TrackEval',
            no_metric_approximation=True,
            fast_batch_persistent_pool=True,
        ),
    )
    atomic_json(folder / 'official_completion.json', result)
    return meta['folder']


def evaluate_batch_in_process(tasks, *, max_workers=20, use_cache=True):
    """Run batch official evaluation inside an already-isolated process."""
    started = time.perf_counter()
    targets = {PAIRS[t['pair']][1] for t in tasks}
    frame_counts, gt_roots = _load_frame_counts(targets)
    cache_file_str = str(cache_path())

    # Resolve KITTI GT root ONCE in coordinator (avoids 160 redundant recover() calls)
    if 'KITTI-V2' in targets and gt_roots['KITTI-V2'] is None:
        from flex_uda.a100.kitti_gt_restore import recover
        kitti_task = next(t for t in tasks if PAIRS[t['pair']][1] == 'KITTI-V2')
        gt_roots['KITTI-V2'] = str(recover(Path(kitti_task['folder'])) / 'gt_template')

    # 1. Verify seals & create symlinks in parallel
    prep_args = [
        (str(t['folder']), t['pair'], PAIRS[t['pair']][1],
         frame_counts[PAIRS[t['pair']][1]], gt_roots[PAIRS[t['pair']][1]])
        for t in tasks
    ]
    with ProcessPoolExecutor(max_workers=max_workers, initializer=_init_worker) as pool:
        metas = list(pool.map(_prepare_folder, prep_args, chunksize=8))
        pending = [m for m in metas if not m['skipped']]
        if not pending:
            return dict(status='ALL_ALREADY_DONE', total=len(tasks), evaluated=0,
                        wall_seconds=time.perf_counter() - started)

        # 2. Collect all unique (pair, seq, pred_sha) keys across the entire batch
        unique_keys = {}  # (pair, seq, sha) -> (folder, truth, frame_count)
        for m in pending:
            pair = m['pair']
            for seq, sha in m['signatures'].items():
                key = (pair, seq, sha)
                if key not in unique_keys:
                    unique_keys[key] = (m['folder'], m['truth'], m['seq_info'][seq])

        # 3. Bulk-lookup SQLite episode cache once
        cached_blobs = {}
        cached_refs = {}
        if use_cache:
            db = connect()
            # Query per pair using chunked IN queries or fast per-key lookup in one connection
            for key in unique_keys:
                row = db.execute(
                    'SELECT value, reference FROM episodes WHERE pair=? AND seq=? AND prediction=?',
                    key,
                ).fetchone()
                if row is not None:
                    cached_blobs[key] = row[0]
                    cached_refs[key] = row[1]
            db.close()

        # 4. Evaluate ONLY unique cache-missed episodes across all workers!
        missing_jobs = [
            (pair, seq, sha, folder_str, truth_str, fc)
            for (pair, seq, sha), (folder_str, truth_str, fc) in unique_keys.items()
            if (pair, seq, sha) not in cached_blobs
        ]
        t_eval0 = time.perf_counter()
        computed_rows = []
        if missing_jobs:
            chunk = max(1, min(8, len(missing_jobs) // (max_workers * 4) + 1))
            for pair, seq, sha, blob, ref_folder in pool.map(_eval_unique_episode, missing_jobs, chunksize=chunk):
                key = (pair, seq, sha)
                cached_blobs[key] = blob
                computed_rows.append((pair, seq, sha, blob, ref_folder))
        t_eval = time.perf_counter() - t_eval0

        # 5. Bulk-commit newly computed episodes to SQLite in ONE transaction
        if use_cache and computed_rows:
            db = connect()
            db.executemany('INSERT OR IGNORE INTO episodes VALUES(?,?,?,?,?)', computed_rows)
            db.commit()
            db.close()

        # 6. Finalize all folders (write CSVs & official_completion.json) in parallel
        fin_args = []
        for m in pending:
            pair = m['pair']
            blobs = {seq: cached_blobs[(pair, seq, sha)] for seq, sha in m['signatures'].items()}
            refs = {seq: cached_refs[(pair, seq, sha)] for seq, sha in m['signatures'].items()
                    if (pair, seq, sha) in cached_refs}
            new_seqs = [seq for seq, sha in m['signatures'].items() if (pair, seq, sha) not in cached_refs]
            fin_args.append((m, blobs, refs, new_seqs, cache_file_str))
        done_folders = list(pool.map(_finalize_folder, fin_args, chunksize=4))

    return dict(
        status='PASS_FAST_BATCH_OFFICIAL_EVALUATION',
        total_tasks=len(tasks),
        evaluated_tasks=len(done_folders),
        unique_episodes=len(unique_keys),
        cache_hits=len(unique_keys) - len(missing_jobs),
        unique_computed=len(missing_jobs),
        eval_phase_seconds=round(t_eval, 3),
        total_wall_seconds=round(time.perf_counter() - started, 3),
        max_workers=max_workers,
    )


def evaluate_many(tasks, *, max_workers=20, use_cache=True):
    """Spawn ONE isolated child interpreter to evaluate all sealed tasks cleanly."""
    with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False) as tf:
        json.dump(list(tasks), tf)
        manifest_path = Path(tf.name)
    try:
        cmd = [
            sys.executable, '-m', 'flex_uda.a100.fast_evaluation_reuse',
            '--manifest', str(manifest_path),
            '--workers', str(int(max_workers)),
        ]
        if not use_cache:
            cmd.append('--no-cache')
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1',
                   OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
        proc = subprocess.run(cmd, cwd=ROOT, stdin=subprocess.DEVNULL,
                              capture_output=True, text=True, env=env)
        if proc.returncode != 0:
            raise RuntimeError(f'Fast batch official eval failed (rc={proc.returncode}):\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}')
        lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
        return json.loads(lines[-1])
    finally:
        manifest_path.unlink(missing_ok=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--workers', type=int, default=20)
    parser.add_argument('--no-cache', action='store_true')
    args = parser.parse_args()
    task_list = json.loads(Path(args.manifest).read_text())
    summary = evaluate_batch_in_process(task_list, max_workers=args.workers, use_cache=not args.no_cache)
    print(json.dumps(summary), flush=True)
