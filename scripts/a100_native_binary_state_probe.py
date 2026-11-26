"""State-vs-identity on reserved teacher probes, before final all-paid fitting.

This is a fixed-prefix teacher-risk diagnostic, not RMOT quality or a newly
trained no-state ablation. Existing round2 checkpoints were fitted without
ALL current-video probe track groups. Both readouts run on the same CPU.
"""
from collections import defaultdict
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from flex_tta.runtime.pf_receipts import atomic_json, file_sha
from flex_uda.a100.mechanism import metadata, generation_guard
from flex_uda.active.joint_observation_inputs import JointObservationVideo, use_verified_local_features
from flex_uda.active.native_binary_models import make_model
from flex_uda.active.rsi_probe_conditioning import comparable_paid_probe
from scripts.a100_native_binary_utility_six_setting import PAIRS, PUBLIC, OUT, source_root


def opinions(model, video, paid, mean, std):
    q = np.asarray([r['local_query'] for r in paid], dtype=int)
    t = np.asarray([r['local_track'] for r in paid], dtype=int)
    rows = np.asarray([video.positions[int(r['case']['frame_id']), int(r['case']['track_id'])] for r in paid])
    assert all(video.queries[qi]['expression_id'] == r['case']['query_id'] for qi, r in zip(q, paid))
    assert all(int(video.tracks[ti]) == int(r['case']['track_id']) for ti, r in zip(t, paid))
    anchors = ((video.raw_bags[q, t]-mean)/std).astype('float32'); present = video.present[t]
    anchors[~present] = 0
    current = ((video.dense_rows(q, rows)-mean)/std).astype('float32')
    identity, full = [], []
    with torch.no_grad():
        for begin in range(0, len(paid), 1024):
            a = torch.as_tensor(anchors[begin:begin+1024]); p = torch.as_tensor(present[begin:begin+1024])
            x = torch.as_tensor(current[begin:begin+1024])
            context = model.context(model.utility_encoder(a), p)
            identity.append(model.identity(context).squeeze(-1).double().numpy())
            full.append(model.utility(a, p, x).double().numpy())
    return np.concatenate(identity), np.concatenate(full)


def risks(paid, identity, full):
    groups = defaultdict(list)
    for i, (a, b) in enumerate(zip(paid[::2], paid[1::2])):
        p, q = a['probability'], b['probability']; mass = p*(1-q)+q*(1-p)
        r = p*(1-q)/mass if mass else .5
        delta = np.asarray([identity[2*i]-identity[2*i+1], full[2*i]-full[2*i+1]])
        ce = mass*(r*np.logaddexp(0., -delta)+(1-r)*np.logaddexp(0., delta))
        entropy = -mass*(r*np.log(max(r, 1e-300))+(1-r)*np.log(max(1-r, 1e-300)))
        groups[a['local_query']].append([ce[0], ce[1], mass*np.log(2.), entropy])
    per_query = np.asarray([np.mean(v, axis=0) for v in groups.values()])
    values = per_query.mean(0)
    return dict(probe_queries=len(groups), paid_probe_nodes=len(paid),
        identity_CE=float(values[0]), full_CE=float(values[1]), constant_CE=float(values[2]),
        entropy_floor=float(values[3]), full_minus_identity_CE=float(values[1]-values[0]),
        weighted_conditional_teacher_information=float(values[2]-values[3]))


def main():
    guard = generation_guard(); torch.set_num_threads(2); use_verified_local_features()
    settings, details = {}, []; maximum = 0.
    for pair, (_, target) in PAIRS.items():
        root = source_root(pair); videos = sorted(metadata(target)['videos'])[:10]
        collected = {p: [] for p in ('LEARNED', 'FROZEN')}; skipped = defaultdict(int)
        for video_id in videos:
            local = root/'videos'/video_id
            receipt = json.loads((local/'receipt.json').read_text()); assert receipt['GT_open_attempts'] == 0
            video = None
            for policy in collected:
                folder = local/'policies'/policy
                paid_doc = json.loads((folder/'paid.json').read_text()); assert paid_doc['GT_open_attempts'] == 0
                paid = {r['uid']: r for r in paid_doc['paid']}
                sequence = json.loads((folder/'sequence.json').read_text())['sequence']
                rounds = receipt['fits']['FULL_CONDITIONAL_REFERENCE_'+policy]['rounds']
                stage = rounds[-1]['stage']; assert stage == 2
                probe = [paid[e['uid']] for e in sequence if e['role'] == 'probe' and e['stage'] == stage]
                probe, conditioning = comparable_paid_probe(probe)
                if not probe:
                    skipped[policy] += 1; continue
                path = folder/'round_latest.pt'
                saved = torch.load(path, map_location='cpu', weights_only=False)
                assert saved['stage'] == stage
                assert set(saved['training_UIDs']) == {e['uid'] for e in sequence if e['role'] == 'train'}
                train_groups = {(paid[u]['local_query'], paid[u]['local_track']) for u in saved['training_UIDs']}
                all_probe_groups = {(paid[e['uid']]['local_query'], paid[e['uid']]['local_track'])
                    for e in sequence if e['role'] == 'probe'}
                assert not train_groups & all_probe_groups
                checkpoint = saved['solver']; recipe = checkpoint['recipe']
                assert recipe == dict(family='structured', conditional=True, reference=True, identity=True, policy=policy)
                if video is None:
                    video = JointObservationVideo(pair, video_id)
                model = make_model(recipe, 1101, checkpoint['width']).eval()
                model.load_state_dict(checkpoint['model'])
                identity, full = opinions(model, video, probe, checkpoint['mean'], checkpoint['std'])
                row = risks(probe, identity, full)
                original = rounds[-1]['actor_update']['teacher_probe_CE_after']
                error = abs(row['full_CE']-original); maximum = max(maximum, error)
                assert error < 1e-4, (pair, video_id, policy, error)
                row.update(pair=pair, video=video_id, policy=policy,
                    CPU_original_vs_logged_GPU_CE_error=error, conditioning=conditioning,
                    current_video_probe_groups_never_fit_in_this_checkpoint=True,
                    checkpoint=str(path), checkpoint_sha256=file_sha(path))
                collected[policy].append(row); details.append(row)
            print(json.dumps(dict(pair=pair, completed_video=video_id,
                evaluated_rounds=sum(len(v) for v in collected.values()))), flush=True)
        policies = {}
        for policy, values in collected.items():
            assert values
            weights = np.asarray([r['probe_queries'] for r in values]); total = weights.sum()
            aggregate = {name: float(sum(w*r[name] for w, r in zip(weights, values))/total)
                for name in ('identity_CE', 'full_CE', 'constant_CE', 'entropy_floor', 'full_minus_identity_CE',
                             'weighted_conditional_teacher_information')}
            policies[policy] = dict(**aggregate, scored_rounds=len(values), empty_last_rounds=skipped[policy],
                query_video_probe_episodes=int(total), paid_probe_nodes=sum(r['paid_probe_nodes'] for r in values),
                full_better_rounds=sum(r['full_minus_identity_CE'] < -1e-10 for r in values),
                identity_better_rounds=sum(r['full_minus_identity_CE'] > 1e-10 for r in values),
                tied_rounds=sum(abs(r['full_minus_identity_CE']) <= 1e-10 for r in values))
        settings[pair] = dict(videos=videos, source_root=str(root), policies=policies)
    assert not guard
    destination = OUT.parent/'state_probe_diagnosis_20261124'
    atomic_json(destination/'per_video.json', dict(rows=details))
    result = dict(status='COMPLETED_FIXED_PREFIX_RESERVED_TEACHER_STATE_DIAGNOSIS', settings=settings,
        scope='First min(10,N) completed videos, third-round checkpoints BEFORE final all-paid fitting; current-video whole probe groups excluded from training.',
        risk='Current exactly-one-mass-weighted conditional teacher CE, averaged within query then across query-video probe episodes.',
        lower_CE_is_better=True, inference_only_identity_deletion_not_a_retrained_no_state_model=True,
        no_claim_of_true_temporal_semantics_or_RMOT_accuracy=True,
        changing_readout_on_a_probe_is_descriptive_not_an_independent_validation_or_checkpoint_selection=True,
        same_CPU_for_both_models_with_separate_logged_GPU_reproduction_error=True,
        maximum_CPU_original_vs_logged_GPU_CE_error=maximum,
        no_GT_no_new_teacher_no_training_no_new_RMOT_predictions=True,
        local_details=str(destination/'per_video.json'), local_details_sha256=file_sha(destination/'per_video.json'),
        source_sha256=file_sha(Path(__file__)), all_live_sources_and_primaries_unchanged=True, goal_status='ACTIVE')
    atomic_json(PUBLIC/'native_binary_state_probe_diagnosis_20261124.json', result)
    print(json.dumps(dict(status=result['status'], full_minus_identity_CE={p:{k:v['full_minus_identity_CE']
        for k,v in r['policies'].items()} for p,r in settings.items()})), flush=True)


if __name__ == '__main__':
    main()
