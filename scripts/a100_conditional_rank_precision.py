"""Check whether stored probabilities tie at a CountMatched cutoff.

Uses completed raw outputs, without GT or new rankings/predictions. A
float32 sigmoid is monotone but can lose strict logit ordering at 0/1.
"""
from collections import Counter
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from flex_tta.runtime.pf_receipts import atomic_json, file_sha
from flex_uda.a100.mechanism import PAIRS, generation_guard, metadata
from scripts.a100_conditional_six_setting import OUT as QUALITY, PRIMARY
from scripts.a100_six_direction import OUT as SIX, PUBLIC

PLAN = PUBLIC/'conditional_rank_precision_plan.json'
OUT = SIX/'conditional_rank_precision_20261123'
COMPARATORS = tuple(stem+'_'+suffix for stem in ('FRAME_SOFT', 'FRAME_HARD', 'FLAT_SOFT', 'FLAT_HARD')
    for suffix in ('FINAL', 'MEAN'))
NAMES = (PRIMARY, *COMPARATORS)
PROJECTIONS = ((PRIMARY, PRIMARY),)+tuple((rank, donor) for baseline in COMPARATORS
    for rank, donor in ((baseline, PRIMARY), (PRIMARY, baseline), (baseline, baseline)))


def inspect(sorted_descending, count):
    if count == 0 or count == len(sorted_descending):
        return dict(nontrivial_cutoff=0, exact_cutoff_tie=0)
    value = sorted_descending[count-1]
    tied = bool(value == sorted_descending[count])
    return dict(nontrivial_cutoff=1, exact_cutoff_tie=int(tied),
        cutoff_tie_at_one=int(tied and value == 1),
        cutoff_tie_at_zero=int(tied and value == 0),
        cutoff_tie_strictly_inside_unit_interval=int(tied and 0 < value < 1))


def main():
    plan = json.loads(PLAN.read_text())
    assert file_sha(Path(__file__)) == plan['source_sha256']
    guard = generation_guard(); started = time.perf_counter()
    collapsed = (1/(1+np.exp(-np.asarray([20., 30.])))).astype('float32')
    assert np.array_equal(collapsed, [1., 1.])
    assert inspect(np.asarray([1., 1., .2, .1]), 1)['exact_cutoff_tie'] == 1
    assert inspect(np.asarray([1., 1., .2, .1]), 2)['exact_cutoff_tie'] == 0
    conditions = {}
    for pair, (_, target) in PAIRS.items():
        root = QUALITY/pair/'seed0'
        complete = json.loads((root/'completion.json').read_text())
        assert complete['GT_open_attempts'] == 0
        md = metadata(target); totals = {p:Counter() for p in PROJECTIONS}
        saturation = {n:Counter() for n in NAMES}; video_records = []
        for video, bank in sorted(md['videos'].items()):
            matrices = {n:np.load(root/'videos'/video/(n+'_scores.npy'), mmap_mode='r') for n in NAMES}
            assert all(v.dtype == np.float32 for v in matrices.values())
            obs = np.asarray(bank['observations'], int)
            local = {p:Counter() for p in PROJECTIONS}
            nq = len([q for q in md['episodes'] if q['video_id'] == video])
            for frame in np.unique(obs[:, 0]):
                ids = np.flatnonzero(obs[:, 0] == frame)
                for qi in range(nq):
                    values = {n:np.asarray(a[qi, ids]) for n, a in matrices.items()}
                    counts = {n:int((v > .5).sum()) for n, v in values.items()}
                    sorted_values = {n:np.sort(v)[::-1] for n, v in values.items()}
                    for n, value in values.items():
                        saturation[n].update(candidate_scores=len(value),
                            scores_exactly_zero=int((value == 0).sum()),
                            scores_exactly_one=int((value == 1).sum()))
                    for rank, donor in PROJECTIONS:
                        local[rank, donor].update(query_frame_units=1)
                        local[rank, donor].update(inspect(sorted_values[rank], counts[donor]))
            for key, value in local.items():
                totals[key].update(value)
            video_records.append(dict(video=video, projections={r+'__AT_'+d+'_COUNT':dict(v)
                for (r, d), v in local.items()}))
        projections = {}
        for (rank, donor), value in totals.items():
            if rank == donor:
                assert value['exact_cutoff_tie'] == 0, (pair, rank, value)
            projections[rank+'__AT_'+donor+'_COUNT'] = dict(value,
                exact_tie_fraction_of_nontrivial_cutoffs=value['exact_cutoff_tie']/max(value['nontrivial_cutoff'], 1))
        conditions[pair] = dict(seed=0, saturation={n:dict(v) for n, v in saturation.items()},
            projections=projections)
        atomic_json(OUT/pair/'per_video.json', dict(videos=video_records,
            no_GT_or_new_predictions=True, original_rankings_unchanged=True))
    assert not guard
    report = dict(status='PASS_SEALED_PROBABILITY_RANK_PRECISION_AUDIT_NO_GT',
        conditions=conditions, selection=plan['selection'],
        synthetic_distinct_logits_collapsed_to_equal_float32_probabilities=collapsed.tolist(),
        self_count_cutoffs_never_split_their_own_threshold_score_group=True,
        interpretation='A cross-count cutoff tie is ranking ambiguity in saved probability precision. It does not prove semantic error, change raw outputs, or recover the lost logit order.',
        no_logit_recovery_or_jitter_or_teacher_GT_tie_breaking=True,
        no_new_GT_training_QA_backbone_or_RMOT_outputs=True,
        no_changes_to_active_numerical_recipes=True,
        manual_GT_open_attempts=0, wall_seconds=time.perf_counter()-started,
        source_sha256=plan['source_sha256'], goal_status='ACTIVE')
    atomic_json(OUT/'completion.json', report)
    atomic_json(PUBLIC/'conditional_rank_precision.json', report)
    print(json.dumps(report))


if __name__ == '__main__':
    main()
