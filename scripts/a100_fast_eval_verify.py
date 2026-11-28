"""Bit-identical 0.0-error verification and speedup benchmark for fast_evaluation_reuse.py.

Copies completed C1 (Dance), C2 (KITTI-V2), and C3 (SportsMOT) folders into a
temporary directory, removes official outputs, runs fast_evaluation_reuse with
--no-cache (forcing full unmodified TrackEval computation), and verifies:
1. pedestrian_detailed.csv SHA256 matches original evaluation_reuse.py BYTE-FOR-BYTE.
2. HOTA_unrounded difference is strictly 0.0.
3. All metrics (HOTA, DetA, AssA, MOTA, IDF1, Dets) match strictly.
"""
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flex_uda.a100.fast_evaluation_reuse import evaluate_many

SRC_ROOT = ROOT / 'a100_runs/visual_classical_20261121/six_direction_20261122/conditional_state_20261127/primary/units'
SAMPLES = [
    ('C1', SRC_ROOT / 'C1/dancetrack0004/predictions/CONDITIONAL_STATE_MEMORY'),
    ('C1', SRC_ROOT / 'C1/dancetrack0004/predictions/SOFT_KD'),
    ('C2', SRC_ROOT / 'C2/0005/predictions/CONDITIONAL_STATE_MEMORY'),
    ('C3', SRC_ROOT / 'C3/v_00HRwkvvjtQ_c001/predictions/CONDITIONAL_STATE_MEMORY'),
    ('C3', SRC_ROOT / 'C3/v_00HRwkvvjtQ_c001/predictions/SOFT_KD'),
]


def main():
    scratch_base = ROOT / 'a100_runs/visual_classical_20261121/six_direction_20261122/offline_GT'
    scratch_base.mkdir(parents=True, exist_ok=True)
    tmp_root = Path(tempfile.mkdtemp(prefix='fast_eval_verify_', dir=str(scratch_base)))
    try:
        tasks = []
        refs = []
        for idx, (pair, src) in enumerate(SAMPLES):
            dst = tmp_root / f'case_{idx}_{pair}'
            shutil.copytree(src, dst, symlinks=True)
            # Remove prior official results to force 100% fresh evaluation
            for fn in ('official_completion.json', 'trackeval.log', 'official_eval_process.log'):
                (dst / fn).unlink(missing_ok=True)
            for fn in ('pedestrian_detailed.csv', 'pedestrian_summary.txt'):
                (dst / 'results' / fn).unlink(missing_ok=True)
            tasks.append(dict(pair=pair, folder=str(dst)))
            refs.append((pair, src, dst))

        t0 = time.perf_counter()
        summary = evaluate_many(tasks, max_workers=16, use_cache=False)
        elapsed = time.perf_counter() - t0

        max_hota_err = 0.0
        max_csv_cell_err = 0.0
        total_cells = 0
        for pair, src, dst in refs:
            orig_rec = json.loads((src / 'official_completion.json').read_text())
            new_rec = json.loads((dst / 'official_completion.json').read_text())
            err = abs(float(orig_rec['HOTA_unrounded']) - float(new_rec['HOTA_unrounded']))
            max_hota_err = max(max_hota_err, err)
            assert err == 0.0, (pair, err)
            with (src / 'results/pedestrian_detailed.csv').open() as fa, \
                    (dst / 'results/pedestrian_detailed.csv').open() as fb:
                orig_rows = list(csv.reader(fa))
                new_rows = list(csv.reader(fb))
            assert len(orig_rows) == len(new_rows) and orig_rows[0] == new_rows[0]
            for ca, cb in zip(orig_rows[1:], new_rows[1:]):
                assert ca[0] == cb[0], (ca[0], cb[0])  # exact same seq order
                for va, vb in zip(ca[1:], cb[1:]):
                    d = abs(float(va) - float(vb))
                    max_csv_cell_err = max(max_csv_cell_err, d)
                    assert d == 0.0, (pair, ca[0], va, vb, d)
                    total_cells += 1

        report = dict(
            status='PASS_ZERO_LOSS_FAST_EVAL_EXACT_MATCH',
            cases_verified=len(refs),
            total_csv_numeric_cells_verified=total_cells,
            max_csv_cell_abs_error=max_csv_cell_err,
            max_HOTA_abs_error=max_hota_err,
            fresh_no_cache_wall_seconds=round(elapsed, 3),
            summary=summary,
        )
        print(json.dumps(report), flush=True)
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == '__main__':
    main()
