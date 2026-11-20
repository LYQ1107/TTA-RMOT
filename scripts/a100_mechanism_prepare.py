"""Register samples and stopping rules before any new GT analysis."""
import hashlib
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
from flex_tta.runtime.pf_receipts import atomic_json, file_sha
from flex_uda.a100.mechanism import OUT, OLD, PAIRS, metadata, history, generation_guard
from flex_uda.active.acquisition import expression_types


def main():
    attempts = generation_guard(); OUT.mkdir(parents=True, exist_ok=True)
    samples = []; inputs = {}; rng = np.random.default_rng(20261121)
    for pair, (model, target) in PAIRS.items():
        m = metadata(target); base = history(pair); eligible = {}; parents = {}
        qindex = {(q['video_id'], q['expression_id']): q for q in m['episodes']}
        for video, bank in sorted(m['videos'].items()):
            parent = video.rsplit('_c', 1)[0] if target == 'SportsMOT' else video
            parents[video] = parent
            track_frames = {}
            for frame, tid in bank['observations']:
                track_frames.setdefault(int(tid), []).append(int(frame))
            p = base / 'videos' / video / 'questions.json'
            answers = json.loads(p.read_text())
            for a in answers:
                key = (video, a['query_id'], int(a['track_id']), int(a['frame_id']))
                if key in eligible or len([t for t in track_frames[key[2]] if t <= key[3]]) < 4:
                    continue
                q = qindex[video, a['query_id']]
                types = expression_types(q['query_text'])
                kind = 'MOTION' if 'MOTION' in types else 'SPATIAL' if 'SPATIAL' in types else 'ROLE' if 'ROLE' in types else 'APPEARANCE' if 'APPEARANCE' in types else 'OTHER'
                eligible[key] = dict(pair=pair, target=target, model=model, video_id=video,
                    query_id=key[1], track_id=key[2], frame_id=key[3], query_text=q['query_text'],
                    query_type=kind, prior_stratum=a.get('prior_stratum', -1), parent_video=parent,
                    original_answer_path=str(p), original_view=a['effective_view'],
                    original_yes_probability=a['yes_probability'])
        # V1: round robin across video/type/prior buckets, no GT. The100-point
        # cutoff may stop within the first pass, concentrating sorted videos.
        # Retain this registered selection; a future study needs parent quotas.
        buckets = {}
        for item in eligible.values():
            key = (item['video_id'], item['query_type'], item['prior_stratum'])
            buckets.setdefault(key, []).append(item)
        for values in buckets.values():
            rng.shuffle(values)
        selected = []
        while len(selected) < 100 and any(buckets.values()):
            for key in sorted(buckets, key=str):
                if buckets[key]: selected.append(buckets[key].pop())
                if len(selected) == 100: break
        all_parents = sorted(set(parents.values()))
        calibration = set(all_parents[::3])
        for row in selected:
            row['split'] = 'calibration' if row['parent_video'] in calibration else 'evaluation'
            row['sample_id'] = hashlib.sha256(json.dumps({k: row[k] for k in
                ('pair', 'video_id', 'query_id', 'track_id', 'frame_id')}, sort_keys=True).encode()).hexdigest()[:20]
        samples.extend(selected)
        inputs[pair] = dict(target=target, model=model, videos=len(m['videos']), queries=len(m['episodes']),
            candidate_query_rows=m['candidate_query_rows'], history=str(base),
            historical_prediction_seal_sha256=file_sha(base / 'predictions_seal.json'),
            eligible_locations=len(eligible), selected=len(selected),
            calibration=sum(x['split'] == 'calibration' for x in selected),
            evaluation=sum(x['split'] == 'evaluation' for x in selected),
            original_parent_groups=len(all_parents))
    assert len({r['sample_id'] for r in samples}) == len(samples)
    assert len(samples) <= 300 and not attempts
    plan = dict(version='A100_MECHANISM_20261121_V1',registered_before_new_GT=True,
        premise='visual intuition motivates testable routing, reliability, existence and temporal hypotheses; no preselected positive conclusion',
        physical_GPUs=[0, 1, 2, 3], inputs=inputs, new_teacher_call_cap=900,
        evidence_locations=len(samples), seed=20261121, evidence_views=['CROP', 'CLIP', 'FRAME_BOX'],
        sampling='historical actual requested pairs, >=4 past observations; round robin video/type/prior stratum, seeded shuffle, no GT',
        split='sorted original parent-video groups every third group calibration, others evaluation; no same-parent leakage',
        mechanism=dict(primary=['CROP_FIXED', 'ROUTED_FIXED', 'CROP_DS', 'ROUTED_DS'],
            supplementary=['CLIP_FIXED', 'FRAME_BOX_FIXED', 'ALL_MEAN', 'ALL_DS'],
            DS_fit='calibration evidence only, Beta(2,2), 20EM steps, source prior .5, no GT',
            fixed_reliability=[.8, .8], posterior_threshold=.5,
            prior='shared frozen source-calibrated frame-topk selection frequency clipped .05/.95, logit clipped [-2,2]',
            no_parameter_search=True, neural_optimizer_updates=0,
            claim='small bank mechanistic diagnosis; not full-domain deployment or independent confirmation'),
        audit=dict(primary_pair='C4', secondary=['C1', 'C5'],
            frame_assignment='maximum cardinality then IoU at .5; original boxes and physical IDs unchanged',
            oracles=['FRAME_ORACLE', 'TRACK_MAJORITY_ORACLE'],
            oracle_not_strict_theoretical_bound=True, GT_offline_only=True,
            flicker='adjacent observed frames <=2 apart; raw margin sign, frame topk and final mask; stable GT membership and physical identity separated',
            no_additive_HOTA_error_percentages=True),
        temporal=dict(gate='stable-membership frame-topk false flips >0; final v8P track-level outputs audited separately',
            controls=['RAW_MARGIN', 'EMA_MARGIN', 'HYSTERESIS_MARGIN'],alpha=.2,on_probability=.6,off_probability=.4,
            reset_after_frame_gap=2, no_GT_parameter_selection=True),
        protocol=dict(conditions=['AUTO_BATCH', 'EXPLICIT_TASK', 'UNIFIED_CHECKLIST'],
            batch_invariance=True, posterior_one_is_video_level_only=True,
            fixed_answer_replay_not_live_policy=True),
        acquisition=dict(conditional=True,methods=['UNIFORM','INFORMATION_GAIN','UNCERTAINTY','UNCERTAINTY_EXPLORE'],
            exploration_fraction=.2, nominal_bank_budget_fractions=[.1,.3,1.],
            no_live_compute_claim_from_prepaid_bank=True),
        profile=dict(physical_GPU=1, max_frames=16, cache_mib=[512,8192],
            exact_numerical_equivalence_required=True, new_models_not_needed=True),
        independent_validation=dict(require_unseen_compatible_assets=True,
            existing_three_validation_sets_are_development=True,
            no_recut_disguised_as_blind_test=True,
            methods=['FINAL_FROZEN','YT_PRIOR','UNIFORM_K'],seeds=[0,17,42],
            conditional_on_assets_and_pilot=True),
        GT_open_attempts=len(attempts), NAS_write=False,
        source_commit=__import__('subprocess').check_output(['git','rev-parse','HEAD'],text=True).strip())
    for name, data in [('preregistration.json', plan), ('evidence_manifest.json', dict(samples=samples, GT_used=False))]:
        p = OUT / name
        if p.exists(): assert json.loads(p.read_text()) == data, 'Do not overwrite a different preregistration'
        else: atomic_json(p, data)
    print(json.dumps(dict(status='REGISTERED',inputs=inputs,locations=len(samples),new_teacher_call_cap=900), ensure_ascii=False),flush=True)


if __name__ == '__main__': main()
