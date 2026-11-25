"""Actual frozen-prefix readout audit, no GT, fitting, or new teacher calls."""
import copy
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from flex_tta.runtime.pf_receipts import atomic_json, file_sha
from flex_uda.a100.mechanism import generation_guard
from flex_uda.active.joint_observation_inputs import JointObservationVideo, use_verified_local_features
from flex_uda.active.native_binary_inputs import inventory, NativeVisual
from flex_uda.active.native_binary_models import make_model, log_probabilities, pair_marginal_logs
from flex_uda.active.conditional_set_readout import at_donor_counts
from scripts.a100_native_binary_trust_six_setting import OUT as PARENT, PRIMARY, PUBLIC, RECIPES


def dense(model, visual, mean, std, *, permute_global_IDs=False):
    video = visual.video; n = len(video.obs); nq = len(video.queries)
    successor = np.empty(n, dtype=int); predecessor = np.empty(n, dtype=int)
    track_ids = np.unique(video.obs[:, 1])
    renamed_order = np.r_[track_ids[::2], track_ids[1::2]]
    renamed_labels = {int(t): i+1 for i, t in enumerate(renamed_order)}
    assert len(set(renamed_labels.values())) == len(track_ids)
    for frame in np.unique(video.obs[:, 0]):
        ids = np.flatnonzero(video.obs[:, 0] == frame)
        ids = ids[np.argsort(video.obs[ids, 1], kind='stable')]
        if permute_global_IDs:
            # ONE persistent ID bijection for the whole video, including
            # frames with a different set of visible candidates.
            ids = ids[np.argsort([renamed_labels[int(video.obs[i, 1])] for i in ids], kind='stable')]
        successor[ids] = np.roll(ids, -1); predecessor[ids] = np.roll(ids, 1)
    u = np.empty((nq, n), dtype='float64'); result = np.empty_like(u)
    with torch.no_grad():
        for qi in range(nq):
            logs = np.empty((n, 2, 2), dtype='float64')
            for begin in range(0, n, 1024):
                ids = np.arange(begin, min(begin+1024, n)); partners = successor[ids]
                a, b = video.columns[ids], video.columns[partners]
                raw = dict(anchors_A=video.raw_bags[qi, a], present_A=video.present[a],
                    anchors_B=video.raw_bags[qi, b], present_B=video.present[b],
                    first=video.dense_rows(np.full(len(ids), qi), ids),
                    second=video.dense_rows(np.full(len(ids), qi), partners))
                for name in ('anchors_A', 'anchors_B', 'first', 'second'):
                    raw[name] = ((raw[name]-mean)/std).astype('float32')
                raw['anchors_A'][~raw['present_A']] = 0
                raw['anchors_B'][~raw['present_B']] = 0
                data = {k: torch.as_tensor(v) for k, v in raw.items()}
                u[qi, ids] = model.utility(data['anchors_A'], data['present_A'], data['first']).double().numpy()
                logs[ids] = pair_marginal_logs(log_probabilities(model, data, dict(family='structured'))).numpy()
            binary = np.logaddexp(logs[:, 0, :], logs[predecessor, 1, :])-np.log(2.)
            result[qi] = binary[:, 0]-binary[:, 1]
    return u, result


def main():
    guard = generation_guard(); torch.set_num_threads(2); use_verified_local_features()
    rows = []
    for pair in ('C1', 'C6'):
        video_id = 'dancetrack0004'; root = PARENT/pair/'seed0/videos'/video_id
        saved = torch.load(root/'policies/LEARNED/closed.pt', map_location='cpu', weights_only=False)['checkpoint']
        recipe = RECIPES['FULL_CONDITIONAL_REFERENCE_LEARNED']
        assert saved['recipe'] == recipe
        model = make_model(recipe, 1101, saved['width']).eval(); model.load_state_dict(saved['model'])
        video = JointObservationVideo(pair, video_id); actions, _ = inventory(pair, video_id)
        visual = NativeVisual(video, actions); mean, std = saved['mean'], saved['std']
        u, margin = dense(model, visual, mean, std)
        renamed_u, renamed_margin = dense(model, visual, mean, std, permute_global_IDs=True)
        changed = copy.deepcopy(model)
        with torch.no_grad():
            for module in (changed.count_encoder, changed.base, changed.interaction):
                for parameter in module.parameters():
                    parameter.zero_()
        count_u, count_margin = dense(changed, visual, mean, std)
        assert np.array_equal(u, renamed_u) and np.array_equal(u, count_u)
        donor = np.load(PARENT/pair/'seed0'/PRIMARY/'videos'/video_id/'mask.npy')
        original_rank = at_donor_counts(margin, donor, video.obs)
        ordinal_rank = at_donor_counts(u, donor, video.obs)
        renamed_rank = at_donor_counts(renamed_margin, donor, video.obs)
        changed_rank = at_donor_counts(count_margin, donor, video.obs)
        assert np.array_equal(ordinal_rank, at_donor_counts(u+1024., donor, video.obs))
        assert np.array_equal(ordinal_rank, at_donor_counts(count_u, donor, video.obs))
        # Compare the actual original GPU output separately, so a CPU/GPU
        # numeric discrepancy is never counted as a count/ring intervention.
        raw_gpu = np.load(root/(PRIMARY+'_scores.npy'), mmap_mode='r')
        gpu_rank = at_donor_counts(raw_gpu, donor, video.obs)
        rows.append(dict(pair=pair, video=video_id, candidate_query_rows=u.size,
            donor_selected_rows=int(donor.sum()),
            CPU_vs_original_GPU_rank_replacements=int((original_rank ^ gpu_rank).sum()//2),
            CPU_vs_original_GPU_raw_decision_differences=int(((margin>0) ^ (raw_gpu>0)).sum()),
            original_vs_utility_rank_replacements=int((original_rank ^ ordinal_rank).sum()//2),
            ring_ID_order_intervention_rank_replacements=int((original_rank ^ renamed_rank).sum()//2),
            count_only_zero_intervention_rank_replacements=int((original_rank ^ changed_rank).sum()//2),
            pure_utility_unchanged_by_count_and_ring=True,
            ring_intervention_is_one_video_global_TrackID_bijection=True,
            conditional_set_common_offset_invariance_on_actual_rows=True,
            max_ring_margin_change=float(np.max(np.abs(margin-renamed_margin))),
            original_checkpoint=str(root/'policies/LEARNED/closed.pt')))
        print(json.dumps(rows[-1]), flush=True)
    assert not guard
    result = dict(status='COMPLETED_REAL_PREFIX_READOUT_COUPLING_AUDIT_NO_GT', rows=rows,
        scope='Fixed first Dance video, C1/C6 Trust LEARNED prefixes; all forward comparisons on CPU2threads.',
        interventions_not_learned_or_selected_with_GT=True, new_fits=0, new_teacher_calls=0,
        new_backbone_calls=0, manual_GT_open_attempts=0, new_HOTA_read=False,
        rank_changes_are_not_quality_or_semantic_learning_evidence=True,
        all_live_primaries_sources_masks_and_registrations_unchanged=True,
        source_sha256={name: file_sha(ROOT/name) for name in (
            'flex_uda/active/conditional_set_readout.py', 'scripts/a100_native_binary_readout_audit.py')},
        goal_status='ACTIVE')
    atomic_json(PUBLIC/'native_binary_readout_coupling_audit_20261124.json', result)
    print(json.dumps(dict(status=result['status'], rows=rows)), flush=True)


if __name__ == '__main__':
    main()
