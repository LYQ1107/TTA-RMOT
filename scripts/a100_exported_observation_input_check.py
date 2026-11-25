"""Bind an already measured real PCD slice and newly computed image RGB.

This is a CPU integration check, not another teacher/feature GPU pilot, head
fit, RMOT quality study or cold deployment. Partial input remains explicit.
"""
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from flex_tta.runtime.pf_receipts import atomic_json, file_sha
from flex_uda.a100.mechanism import generation_guard
from flex_uda.active.exported_observation_inputs import collect_export_rgb, ExportedJointObservationVideo
from flex_uda.active.native_binary_inputs import NativeVisual
from scripts.a100_observation_teacher_pilot import key as request_key
from scripts.a100_six_direction import OUT as SIX, PUBLIC

OUT = SIX/'exported_observation_input_check_20261124'
FEATURES = SIX/'historical_feature_cold_pilot_20261123/export/smoke/0005'


def main():
    begin = time.perf_counter(); guard = generation_guard()
    opened_arrays = Counter(); disallowed = []
    allowed = (FEATURES.resolve(), OUT.resolve())

    def explicit_arrays(event, args):
        if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0]))
        if path.suffix not in ('.npy', '.npz', '.pt', '.pth', '.sqlite'):
            return
        resolved = path.resolve()
        if not any(resolved.is_relative_to(root) for root in allowed):
            disallowed.append(str(path))
            raise PermissionError('Integration check cannot read prior arrays or fitted heads: '+str(path))
        opened_arrays[str(path)] += 1
    sys.addaudithook(explicit_arrays)
    assert not (OUT/'completion.json').exists(), 'Completed check is not a new execution.'
    OUT.mkdir(parents=True, exist_ok=True)
    rgb_receipt = collect_export_rgb('C2', '0005', FEATURES, OUT/'real_RGB', allow_partial=True)
    value = ExportedJointObservationVideo('C2', '0005', FEATURES, OUT/'real_RGB', allow_partial=True)
    assert value.dimension == 1101 and not value.input_receipt['complete_fixed_candidate_coverage']
    actions = {}
    for qi, query in enumerate(value.queries):
        for ti, track in enumerate(value.tracks):
            for row in value.sample_rows[ti, value.present[ti]]:
                case = dict(target=value.target, video_id=value.video,
                    query_id=query['expression_id'], query_text=query['query_text'],
                    track_id=int(track), frame_id=int(value.obs[row, 0]), view='CROP')
                uid = hashlib.sha256(request_key(case).encode()).hexdigest()
                actions[uid] = dict(uid=uid, local_query=qi, local_track=ti, case=case)
    visual = NativeVisual(value, actions)
    order = sorted(actions); point = visual.point_arrays(order); paired = visual.arrays(order)
    nq, n = len(value.queries), len(value.obs)
    pcd = np.asarray(value.pcd).reshape(nq, n, 768)
    checked = 0
    for qi in range(nq):
        dense = value.dense_rows(np.full(n, qi), np.arange(n))
        assert dense.shape == (n, 1101) and np.isfinite(dense).all()
        assert np.array_equal(dense[:, :768], pcd[value.query_rows[qi]].astype('float32'))
        for ti, si in np.argwhere(value.present):
            assert np.array_equal(value.raw_bags[qi, ti, si], dense[value.sample_rows[ti, si]])
        checked += n*768
    assert checked == 907776 and len(order) == 1182
    assert np.isfinite(point['point_current']).all() and np.isfinite(paired['first']).all()
    try:
        ExportedJointObservationVideo('C2', '0005', FEATURES, OUT/'real_RGB')
    except AssertionError as error:
        assert str(error) == 'Full execution requires every fixed candidate observation.'
    else:
        raise AssertionError('A partial export was admitted as a full candidate execution.')
    assert not guard and not disallowed
    report = dict(status='PASS_EXPLICIT_REAL_EXPORT_TO_NATIVE_INPUTS_PARTIAL_INTEGRATION_ONLY',
        pair='C2', video='0005', prior_physical_feature_receipt=str(
            PUBLIC/'historical_feature_cold_pilot_analysis.json'),
        previously_measured_student_export_reused=True,
        reused_export_not_another_GPU_forward_or_full_cold_cost=True,
        new_RGB_receipt=rgb_receipt, explicit_input_receipt=value.input_receipt,
        current_features_dtype='float32, exact promotion of actual exported float16 PCD',
        PCD_elements_checked_exact=checked, query_observation_rows=nq*n,
        native_action_feature_rows=len(order), finite_dense_and_pair_inputs=True,
        sampled_and_dense_rows_exact=True, partial_export_rejected_by_full_execution_default=True,
        opened_arrays=sorted(opened_arrays), historical_or_fitted_array_open_attempts=len(disallowed),
        source_features_or_NAS_history_modified=False, old_RGB_or_anchor_store_used=False,
        new_student_GPU_calls=0, new_teacher_calls=0, new_fits=0, new_RMOT_predictions=0,
        manual_GT_open_attempts=len(guard), new_HOTA_read=False,
        process_wall_seconds=time.perf_counter()-begin, goal_status='ACTIVE',
        remaining='Integrate retained solver and live teacher in a registered full-candidate matched quality/cold execution after complete current quality evidence.',
        source_sha256={p:file_sha(ROOT/p) for p in (
            'flex_uda/active/exported_observation_inputs.py',
            'scripts/a100_exported_observation_input_check.py')})
    atomic_json(OUT/'completion.json', report)
    atomic_json(PUBLIC/'exported_observation_input_check_20261124.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
