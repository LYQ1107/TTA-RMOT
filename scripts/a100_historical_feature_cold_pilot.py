"""Small real-image replay of the legacy exporter, without GT or head fits.

Recreate batch64/near-zero-query-isolation PCD before planning matched cold
costs. Feature agreement on this slice is not agreement over six full settings.
"""
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from flex_tta.runtime.pf_receipts import atomic_json, file_sha
from flex_uda.a100.mechanism import metadata, generation_guard, PAIRS, HISTORY
from scripts.a100_six_direction import OUT as SIX, PUBLIC

OUT = SIX/'historical_feature_cold_pilot_20261123'
STATE = OUT/'supervisor/state.json'
PLAN = PUBLIC/'historical_feature_cold_pilot_plan.json'
SUCCESS = 'PASS_HISTORICAL_FEATURE_COLD_PILOT_RESEARCH_GOAL_ACTIVE'


def main():
    plan = json.loads(PLAN.read_text())
    for name, sha in plan['source_sha256'].items():
        assert file_sha(ROOT/name) == sha, name
    guard = generation_guard()
    started = time.perf_counter()
    atomic_json(STATE, dict(status='LOADING_FROZEN_STUDENT_FOR_LEGACY_FEATURE_SLICE',
        pid=os.getpid(), completed=0, goal_status='ACTIVE'))
    import torch
    from flex_uda.a100.runtime import load_student
    from scripts.presence_export_target import export_video
    pair, video = plan['pair'], plan['video']
    source, target = PAIRS[pair]
    md = metadata(target)
    device = torch.device('cuda:0')
    student, tokenizer, transform, receipt = load_student(OUT/'student_log', device,
        model_name=source, target=target)
    versions = {n:p._version for n, p in student.named_parameters()}
    torch.cuda.synchronize()
    load_seconds = time.perf_counter()-started
    atomic_json(STATE, dict(status='REAL_LEGACY_BATCH64_THREE_FRAME_EXPORT_NO_GT',
        pid=os.getpid(), completed=0, goal_status='ACTIVE'))
    begin = time.perf_counter()
    # reference=None disables a source-margin parity assertion. The independent
    # comparison below instead checks actual saved PCD, which is our question.
    export_video(student, tokenizer, transform, device, md, video, target, source,
        OUT/'export', None, True)
    torch.cuda.synchronize()
    forward_export_seconds = time.perf_counter()-begin
    assert all(p._version == versions[n] for n, p in student.named_parameters())
    folder = OUT/'export/smoke'/video
    index = np.load(folder/'index.npy', mmap_mode='r')
    fresh = np.load(folder/'pcd.npy', mmap_mode='r')
    legacy = HISTORY/'features'/target/source/'videos'/video
    old_meta = json.loads((legacy/'metadata.json').read_text())
    old_obs = np.load(legacy/'observations.npy', mmap_mode='r')
    old_pcd = np.load(legacy/'pcd.npy', mmap_mode='r')
    old_index = np.load(legacy/'index.npy', mmap_mode='r')
    old_q = {q['expression_id']:(i, q) for i, q in enumerate(old_meta['queries'])}
    old_rows = {tuple(map(int, row)):i for i, row in enumerate(old_obs)}
    queries = [q for q in md['episodes'] if q['video_id'] == video]
    assert fresh.dtype == old_pcd.dtype == np.dtype('float16')
    maximum, absolute, changed, values, changed_rows, rows = 0., 0., 0, 0, 0, 0
    for qi, query in enumerate(queries):
        ids = np.flatnonzero(index[:, 0] == qi)
        oq, old_query = old_q[query['expression_id']]
        assert query['query_text'] == old_query['query_text']
        source_rows = oq*len(old_obs)+np.asarray([old_rows[tuple(map(int, r))] for r in index[ids, 1:]])
        assert np.array_equal(old_index[source_rows, 0], np.full(len(ids), oq))
        assert np.array_equal(old_index[source_rows, 1:], index[ids, 1:])
        difference = np.abs(np.asarray(fresh[ids], dtype='float32')-np.asarray(old_pcd[source_rows], dtype='float32'))
        maximum = max(maximum, float(difference.max()))
        absolute += float(difference.sum(dtype='float64'))
        changed += int((difference != 0).sum())
        values += difference.size
        changed_rows += int((difference != 0).any(1).sum())
        rows += len(ids)
    assert values == fresh.size and rows == len(fresh) and not guard
    report = dict(status=SUCCESS, pair=pair, video=video,
        frames=np.unique(index[:, 1]).tolist(), query_observation_rows=rows,
        compared_feature_elements=values, exact_equal=changed == 0,
        changed_float16_elements=changed, changed_query_observation_rows=changed_rows,
        maximum_absolute_difference=maximum, mean_absolute_difference=absolute/values,
        legacy_query_batch=64, legacy_near_zero_query_isolation=True,
        exporter_source_margin_parity_not_audited=True, checkpoint=receipt,
        student_load_and_integrity_wall_seconds=load_seconds,
        three_frame_forward_and_export_wall_seconds=forward_export_seconds,
        total_pilot_wall_seconds=time.perf_counter()-started,
        physical_GPU=os.environ.get('CUDA_VISIBLE_DEVICES'),
        frozen_student_unchanged=True, GT_open_attempts=0,
        new_teacher_calls_or_adaptation_fits=0, no_HOTA_or_method_outputs_read=True,
        no_tolerance_based_feature_substitution=True,
        one_slice_agreement_not_full_cohort_or_paired_cold_deployment=True,
        all_real_images_from_readonly_NAS=True, raw_features_remain_local_not_Git=True,
        goal_status='ACTIVE')
    atomic_json(OUT/'analysis.json', report)
    (PUBLIC/'historical_feature_cold_pilot_analysis.json').write_text(json.dumps(report, separators=(',', ':'))+'\n')
    atomic_json(STATE, dict(status=SUCCESS, pid=os.getpid(), completed=1, goal_status='ACTIVE'))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        atomic_json(STATE, dict(status='FAILED_HISTORICAL_FEATURE_COLD_PILOT_REQUIRES_ROOT_CAUSE',
            pid=os.getpid(), error=repr(error), goal_status='ACTIVE'))
        raise
