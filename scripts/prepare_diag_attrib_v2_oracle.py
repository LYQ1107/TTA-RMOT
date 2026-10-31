#!/usr/bin/env python3
"""Explicit offline ORACLE/B0 diagnostic. NEVER imported by GT-free stages."""
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.diag_attrib_v2_common import *
from scripts.analyze_pf_full_offline import load_eval_gt


def generate():
    require_prereg()
    before = json.loads((RUN / 'GT_free_predictions.json').read_text())
    if before['status'] != 'ALL_GT_FREE_PREDICTIONS_SEALED' or before['GT_free_target_open_attempts']:
        raise ValueError('finish and seal GT-free predictions before offline ORACLE')
    opened = []
    def audit(event, args):
        if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
            path = Path(os.fsdecode(args[0])).absolute().resolve()
            if path.is_relative_to(DATA) and path.name == 'gt.txt':
                opened.append(str(path))
    sys.addaudithook(audit)
    start, cpu = time.perf_counter(), time.process_time()
    metadata, scores, groups, historical = load_bank()
    lines = candidate_lines(metadata)
    g = metadata['query_frame_groups']
    k_gt = np.zeros(g, dtype=np.int32)
    total = 0
    for q in metadata['episodes']:
        truth = load_eval_gt(DATA, q['video_id'], q['expression_id'])
        if not set(truth).issubset(range(q['frame_indices']['start'], q['frame_indices']['stop'])):
            raise ValueError('GT timeline differs from fixed runtime video timeline')
        total += sum(map(len, truth.values()))
        for t, a, b, group in q['groups']:
            k_gt[group] = len(truth.get(t, []))
    if total != 212541:
        raise ValueError('offline GT count does not match fixed official protocol')
    bank_counts = np.bincount(groups, minlength=g)
    overflow = k_gt > bank_counts
    methods = []
    for rank in NAMES:
        ranks = rank_within_groups(scores[rank], groups, g)
        name = matrix_name(rank, 'ORACLE_GT')
        mask = ranks < k_gt[groups]
        write_selection(name, mask, metadata, lines, protocol='ORACLE_USING_TARGET_GT_FOR_COUNTS_DIAGNOSTIC_ONLY')
        methods.append(name)
    budgets = [(f'B0_SCALE_{str(scale).replace(".", "p")}', round_budget(scale, 212541))
               for scale in (.25, .5, .75, 1., 1.25, 1.5, 2., 3.)]
    budgets += [('B0_NATIVE_FROZEN_COUNT', 99704), ('B0_NATIVE_OFFICIAL_COMPLETE_COUNT', 254910), ('B0_NATIVE_D1_COUNT', 345302)]
    boundaries = []
    for name, count in budgets:
        mask, detail = top_budget(scores['FROZEN'], count)
        write_selection(name, mask, metadata, lines, protocol='B0_GT_INFORMED_SCALE_CURVE_NOT_A_METHOD')
        methods.append(name)
        boundaries.append(dict(method=name, registered_budget=count, **detail))
    atomic_json(RUN / 'ORACLE_predictions.json', dict(status='OFFLINE_ORACLE_DIAGNOSTICS_SEALED',
        role='EXPLICIT_TARGET_GT_ORACLE_NOT_A_GT_FREE_METHOD', methods=methods, target_GT_open_count=len(opened),
        target_GT_paths=opened, official_GT_detections=total, ORACLE_overflow_frames=int(overflow.sum()),
        ORACLE_truncated_GT_slots=int(np.maximum(k_gt-bank_counts,0).sum()),
        ORACLE_zero_candidate_overflow_frames=int((overflow & (bank_counts == 0)).sum()),
        B0_boundaries=boundaries, GT_feedback_to_methods=False, optimizer_steps=0, GPU_calls=0,
        CPU_wall_seconds=time.perf_counter()-start, CPU_process_seconds=time.process_time()-cpu,
        prediction_seals={m:file_sha(EVAL / m / 'predictions_seal.json') for m in methods}))
    print(json.dumps(dict(event='ATTRIB_V2_ORACLE_SEALED', methods=len(methods), GT_detections=total,
                         ORACLE_overflow_frames=int(overflow.sum()))), flush=True)


if __name__ == '__main__':
    generate()
