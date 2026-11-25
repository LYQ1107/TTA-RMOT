"""Exact teacher-only information loss from a constant query/TrackID score.

For Bernoulli cross entropy, the optimal constant is the mean teacher target.
Its excess loss over unconstrained per-observation targets is a Jensen gap.
This describes discarded teacher variation, not target accuracy or HOTA.
"""
from collections import defaultdict
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scipy.special import xlogy
from flex_tta.runtime.pf_receipts import atomic_json, file_sha
from flex_uda.a100.mechanism import PAIRS, generation_guard, metadata
from scripts.a100_observation_design_collect import OUT as COLLECTED, TEACHER_SHA
from scripts.a100_observation_design_prepare import POLICIES
from scripts.a100_six_direction import PUBLIC

REGISTRATION = PUBLIC/'teacher_coarsening_audit_registration_v1.json'


def entropy(probability):
    p = np.asarray(probability, dtype=float)
    return -xlogy(p, p)-xlogy(1.-p, 1.-p)


def audit_rows(rows):
    cells = defaultdict(list); frame_cells = defaultdict(list); unique = set()
    crop_cells = defaultdict(list)
    for row in rows:
        if row['request_id'] in unique:
            continue
        unique.add(row['request_id'])
        assert row['frozen_weight_sha256'] == TEACHER_SHA
        cell = row['local_query'], row['local_track']; frame = int(row['frame_id'])
        p = float(row['yes_probability']); assert 0 <= p <= 1
        cells[cell].append((frame, p)); frame_cells[cell+(frame,)].append(p)
        if row['effective_view'] == 'CROP':
            crop_cells[cell].append((frame, p))
    n = len(unique)
    raw_entropy = sum(float(entropy(p)) for values in cells.values() for _, p in values)
    cell_entropy = sum(len(values)*float(entropy(np.mean([p for _, p in values]))) for values in cells.values())
    frame_entropy = sum(len(values)*float(entropy(np.mean(values))) for values in frame_cells.values())
    view_gap = frame_entropy-raw_entropy; anchor_gap = cell_entropy-frame_entropy
    assert view_gap >= -1e-9 and anchor_gap >= -1e-9
    crop_n = sum(map(len, crop_cells.values())); crop_raw = crop_cell = 0.
    for values in crop_cells.values():
        probabilities = [p for _, p in values]
        crop_raw += float(entropy(probabilities).sum())
        crop_cell += len(values)*float(entropy(np.mean(probabilities)))
    clipped_excess = 0.
    for values in cells.values():
        p = float(np.mean([p for _, p in values])); c = np.clip(p, .02, .98)
        ce = -p*np.log(c)-(1.-p)*np.log1p(-c)
        clipped_excess += len(values)*(ce-float(entropy(p)))
    return dict(unique_requests=n, query_track_cells=len(cells), query_track_frame_cells=len(frame_cells),
        cells_with_multiple_requested_anchors=sum(len({f for f, _ in v}) >= 2 for v in cells.values()),
        sum_raw_entropy=raw_entropy, sum_optimal_constant_CE=cell_entropy,
        sum_excess_constant_CE=cell_entropy-raw_entropy,
        sum_within_requested_anchor_view_gap=view_gap, sum_between_requested_anchor_gap=anchor_gap,
        sum_additional_excess_from_fixed_02_98_clipping=clipped_excess,
        crop_unique_requests=crop_n, crop_query_track_cells=len(crop_cells),
        crop_cells_with_multiple_anchors=sum(len({f for f, _ in v}) >= 2 for v in crop_cells.values()),
        sum_crop_only_excess_constant_CE=crop_cell-crop_raw)


def main():
    plan = json.loads(REGISTRATION.read_text())
    for name, sha in plan['source_sha256'].items():
        assert file_sha(ROOT/name) == sha, name
    guard = generation_guard(); start = time.perf_counter(); summaries = {}
    # Analytic witness: constant 0.5 cannot fit both 0.1 and 0.9, while two
    # identical observations have zero coarsening loss. No target data here.
    assert abs(float(entropy(.5)-entropy([.1, .9]).mean())-.3680642071684971) < 1e-12
    assert float(entropy(.3)-entropy([.3, .3]).mean()) == 0
    for pair, (_, target) in PAIRS.items():
        summaries[pair] = {}
        for policy in POLICIES:
            total = defaultdict(float)
            for video in sorted(metadata(target)['videos']):
                document = json.loads((COLLECTED/'policy_answers'/pair/video/(policy+'_answers.json')).read_text())
                assert document['only_this_policy_requested_responses'] and document['GT_open_attempts'] == 0
                values = audit_rows(document['answers'])
                for name, value in values.items():
                    total[name] += value
            n = total['unique_requests']; ncrop = total['crop_unique_requests']
            result = dict(total)
            for name in list(total):
                if name.startswith('sum_'):
                    result[name.removeprefix('sum_')+'_per_request'] = total[name]/(ncrop if 'crop_only' in name else n)
            summaries[pair][policy] = result
    assert not guard
    report = dict(status='PASS_TEACHER_ONLY_COARSENING_LOSS_DECOMPOSITION', settings=summaries,
        identity='Minimum constant Bernoulli CE minus mean individual entropy = H(mean p)-mean H(p). Anchor-first decomposition sums exactly to this gap.',
        weighting='Uniform over distinct registered request_id within each query/TrackID; not the later group-confidence training weight.',
        conditioning='Requested anchor frame is a grouping index; CLIP spans multiple frames. CROP-only gap excludes cross-view mixing but is not a true temporal semantic label.',
        interpretation='Loss of raw teacher variation under a constant score. This variation can be useful state information, teacher error, or both; positive gap does not prove adaptation benefit.',
        between_policy_boundary='Policies select different observations; smaller gap is not automatically a better policy. A policy asking only one anchor has no between-anchor variation to measure.',
        no_GT_or_quality_predictions_read=True, no_new_teacher_calls_or_fits=True,
        source_policy_answers_not_modified=True, wall_seconds=time.perf_counter()-start,
        goal_status='ACTIVE', research_goal_complete=False)
    atomic_json(PUBLIC/'teacher_coarsening_audit_receipt.json', report)
    print(json.dumps(dict(status=report['status'], setting_policies=24,
        DESIGN={p:dict(raw_gap=d['DESIGN']['excess_constant_CE_per_request'],
            within_anchor_view_gap=d['DESIGN']['within_requested_anchor_view_gap_per_request'],
            between_anchor_gap=d['DESIGN']['between_requested_anchor_gap_per_request'],
            crop_only_gap=d['DESIGN']['crop_only_excess_constant_CE_per_request']) for p, d in summaries.items()})))


if __name__ == '__main__':
    main()
