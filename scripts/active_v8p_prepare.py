"""GT-free metadata classification, v8 reuse references and Stage-A cost."""
import json
from pathlib import Path
import shutil
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json, file_sha
from flex_uda.presence.runtime import RUN, runtime_guard, target_metadata, sealed_margins
from flex_uda.presence.predictions import verify_predictions
from flex_uda.active.v8p import classify_protocol, CHECKLIST, PRESUPPOSED
from scripts.active_run import PAIRS


def prepare():
    attempts = runtime_guard()
    public = ROOT/'results/presence_active/redesign_v8p'
    registration = json.loads((public/'manifest.json').read_text())
    old = json.loads((ROOT/'results/presence_active/redesign_v8/execution.json').read_text())
    runtime = RUN/'redesign_v8p'
    protocols, dimensions, inputs, reuses = {}, {}, [], []
    expected = {'Dance': (CHECKLIST, 1, 25), 'KITTI-V2': (PRESUPPOSED, 4, 4),
                'SportsMOT': (PRESUPPOSED, 41, 45)}
    for target in expected:
        meta = target_metadata(target)
        querysets = {v: [q['query_text'] for q in meta['episodes'] if q['video_id'] == v]
                     for v in sorted(meta['videos'])}
        r = classify_protocol(querysets)
        assert (r['protocol'], r['unique_query_sets'], r['videos']) == expected[target], r
        protocols[target] = r
    atomic_json(public/'protocols.json', dict(targets=protocols, GT_open_attempts=len(attempts),
        rule_registration_public_commit='008a9028c645fa06008cb3416ba07b0c1d5482cb',
        raw_text_only=True, computed_once_before_real_runs=True))
    for pair, (model, target) in PAIRS.items():
        meta = target_metadata(target); margin, seal = sealed_margins(target, model)
        budget = 0
        for video, bank in sorted(meta['videos'].items()):
            _, counts = np.unique(np.asarray(bank['observations'])[:, 1], return_counts=True)
            nq = sum(q['video_id'] == video for q in meta['episodes'])
            budget += int(np.floor(.3*nq*int((counts >= 3).sum())))
        dimensions[pair] = dict(target=target, model=model, videos=len(meta['videos']),
            episodes=len(meta['episodes']), candidate_query_rows=len(margin), teacher_budget=budget)
        inputs.append(dict(pair=pair, margin=seal))
        if pair in ('C1', 'C6'):
            folder = Path(old['active_root'])/pair/'MINUS_SLOW/rho_0.3'
            sealed = verify_predictions(folder)
            reuses.append(dict(pair=pair, method='MINUS_SLOW', rho=.3,
                folder=str(folder), seal_sha256=file_sha(folder/'predictions_seal.json'),
                source_public_commit=registration['base_public_commit'],
                episodes=sealed['episode_count'], rows=sealed['prediction_rows'],
                protocol=CHECKLIST, code_path='BeliefStateV8 + questions_v8 (exact objects)',
                new_teacher_calls=0, rerun=False))
    units = json.loads((ROOT/'results/presence_active/redesign_v5/active_map_v1_critical28/remaining_budget_review.json').read_text())['target_unit_costs']
    jobs = registration['stage_A']['new_jobs']
    estimates = [dict(j, total_GPU_hours_estimate=dimensions[j['pair']]['teacher_budget']*
        units[dimensions[j['pair']]['target']]['seconds_per_teacher_call']/3600) for j in jobs]
    cost = dict(version='v8P', jobs=estimates, dimensions=dimensions,
        total_GPU_hours_estimate=sum(j['total_GPU_hours_estimate'] for j in estimates),
        no_cache_discount=True, LoRA_enabled=False, rescore_GPU_hours=0,
        data2_available_bytes=shutil.disk_usage('/data2').free,
        maximum_total_project_GPUs=2, original_v8_stage3_unchanged=True)
    assert cost['data2_available_bytes'] > 2 << 30
    atomic_json(public/'BC_cost_estimate.json', cost)
    atomic_json(public/'inputs.json', dict(files=inputs, reused_v8=reuses,
        GT_open_attempts=len(attempts)))
    manifest = dict(old)
    manifest.update(version='v8P', active_root=str(runtime/'active'), public_root=str(public),
        launcher_root=str(runtime/'launchers'), allowed_registered_jobs=jobs,
        prereg_path=str(public/'PREREG.md'), prereg_sha256=file_sha(public/'PREREG.md'),
        registration_path=str(public/'manifest.json'), registration_sha256=file_sha(public/'manifest.json'),
        amendments=registration['amendments'], maximum_GPUs=2,
        coordinator_script='scripts/active_v8p_continue.py',
        launcher_script='scripts/active_v8p_launch.py', report_script='scripts/active_v8p_report.py',
        protocol_path=str(public/'protocols.json'), protocol_sha256=file_sha(public/'protocols.json'),
        prior_v8_active_root=old['active_root'], prior_v8_public_root=old['public_root'],
        prior_v8_durable_completion=str(RUN/'redesign_v8/launchers/durable_1793959350550541895/completion.json'),
        prior_v8_admission_path=str(RUN/'redesign_v8/launchers/workers_1793959360708780690/admission.json'),
        user_authorization='v8P stage A; B only upon the preregistered stage-A PASS',
        registered_tier='V8P_STAGE_A', full_matrix_authorized=False)
    atomic_json(public/'execution.json', manifest)
    atomic_json(public/'registered_jobs.json', dict(tier='V8P_STAGE_A', jobs=jobs,
        prereg_path=manifest['prereg_path'], prereg_sha256=manifest['prereg_sha256'],
        cost_estimate_sha256=file_sha(public/'BC_cost_estimate.json'), maximum_GPUs=2,
        original_v8_stage3_must_not_be_interrupted=True))
    assert not attempts
    print(json.dumps(dict(event='V8P_PREPARED', protocols={t:r['protocol'] for t,r in protocols.items()},
        total_GPU_hours_estimate=cost['total_GPU_hours_estimate'], reused_v8=reuses,
        GT_open_attempts=0)), flush=True)


if __name__ == '__main__':
    prepare()
