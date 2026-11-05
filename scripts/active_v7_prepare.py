"""GT-free v7 cost gate, immutable input references, and isolated run plan."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from flex_uda.presence.runtime import RUN, runtime_guard, target_metadata, sealed_margins
from scripts.active_run import PAIRS
from flex_tta.runtime.pf_receipts import atomic_json, file_sha


def prepare(estimate_only=False):
    opened = runtime_guard()
    public = ROOT/'results/presence_active/redesign_v7'; public.mkdir(exist_ok=True)
    old_public = ROOT/'results/presence_active/redesign_v5/active_map_v1_critical28'
    v6 = ROOT/'results/presence_active/redesign_v6'
    units = json.loads((old_public/'remaining_budget_review.json').read_text())['target_unit_costs']
    full_cost = {x['pair']: x for x in json.loads((v6/'budget_preflight.json').read_text())['estimates'] if x['method']=='FULL'}
    inputs, dimensions, jobs = [], {}, []
    for pair, (model, target) in PAIRS.items():
        meta = target_metadata(target); margin, seal = sealed_margins(target, model)
        budget = 0
        for video, bank in sorted(meta['videos'].items()):
            _, counts = np.unique(np.asarray(bank['observations'])[:, 1], return_counts=True)
            nq = sum(q['video_id']==video for q in meta['episodes'])
            budget += int(np.floor(.3*nq*int((counts>=3).sum())))
        dimensions[pair] = dict(target=target, model=model, videos=len(meta['videos']),
            episodes=len(meta['episodes']), candidate_query_rows=len(margin), teacher_budget=budget)
        inputs.append(dict(pair=pair, target=target, model=model, margin=seal))
        teacher_hours = budget*units[target]['seconds_per_teacher_call']/3600
        # No cache/early-stop discount; retain the previous conservative batch8
        # full-frame and all-50-step upper estimates for FULL and fixed replay.
        slow = full_cost[pair]
        for method in ('MINUS_SLOW','FULL'):
            phase = teacher_hours if method=='MINUS_SLOW' else slow['total_GPU_hours_estimate']
            jobs.append(dict(pair=pair, method=method, rho=.3, total_GPU_hours_estimate=phase,
                teacher_equivalent_GPU_hours=teacher_hours,
                LoRA_GPU_hours=0. if method=='MINUS_SLOW' else slow['LoRA_GPU_hours'],
                rescore_GPU_hours=0. if method=='MINUS_SLOW' else slow['full_frame_score_GPU_hours']+slow['sample32_score_GPU_hours']))
        if pair in ('C1','C3','C4'):
            phase=slow['full_frame_score_GPU_hours']+slow['sample32_score_GPU_hours']+slow['LoRA_GPU_hours']
            jobs.append(dict(pair=pair, method='SLOW_REPLAY', rho=.3, total_GPU_hours_estimate=phase,
                teacher_equivalent_GPU_hours=0., LoRA_GPU_hours=slow['LoRA_GPU_hours'],
                rescore_GPU_hours=slow['full_frame_score_GPU_hours']+slow['sample32_score_GPU_hours']))
    assert len(jobs)==15
    conditional_bound=sum(dimensions[p]['teacher_budget']*units[PAIRS[p][1]]['seconds_per_CROP_call']/3600 for p in PAIRS)
    total=sum(r['total_GPU_hours_estimate'] for r in jobs)+conditional_bound
    selected=[r for r in jobs if total<=20 or r['method']=='MINUS_SLOW' or r['pair'] in ('C1','C3','C4')]
    assert len(selected)==(15 if total<=20 else 12)
    selected_pairs={r['pair'] for r in selected if r['method']=='FULL'}
    selected_conditional=sum(dimensions[p]['teacher_budget']*units[PAIRS[p][1]]['seconds_per_CROP_call']/3600 for p in selected_pairs)
    frame_bytes=sum(dimensions[r['pair']]['candidate_query_rows']*4*769 for r in selected if r['method']!='MINUS_SLOW')
    storage=Path('/data1/user/TTA-RMOT-assets/presence_active_20261102/redesign_v7')
    available=shutil.disk_usage('/data1').free
    record=dict(status='ALL15_WITHIN20H' if total<=20 else 'TWELVE_JOB_FALLBACK',
        requested_15_GPU_hours_estimate=total, conditional_baseline_upper_GPU_hours=conditional_bound,
        selected_GPU_hours_estimate=sum(r['total_GPU_hours_estimate'] for r in selected)+selected_conditional,
        user_gate_GPU_hours=20, estimates=jobs, dimensions=dimensions,
        selected_jobs=[{k:r[k] for k in ('pair','method','rho')} for r in selected],
        selected_conditional_YT_PRIOR_upper_GPU_hours=selected_conditional,
        frame_cache_bytes_estimate=frame_bytes, data1_available_bytes=available, cache_root=str(storage),
        new_GPU_or_teacher_calls=0, GT_open_attempts=len(opened),
        estimate_scope='conservative shared-device elapsed GPU phases; no cache/skip discount; not a utilization integral',
        teacher_loading_and_CPU_costs_not_in_phase_estimate=True,
        replay_teacher_cost_incremental_zero_but_MINUS_teacher_dependency_included=True,
        prior_v6_cost_estimate_sha256=file_sha(v6/'budget_preflight.json'))
    atomic_json(public/'budget_preflight.json',record); atomic_json(public/'inputs.json',dict(files=inputs,GT_open_attempts=len(opened)))
    print(json.dumps({k:v for k,v in record.items() if k not in ('estimates','dimensions')},indent=2),flush=True)
    assert not opened
    if estimate_only:return
    assert frame_bytes < available, 'Insufficient disk for registered caches; do not delete or move historical assets'
    storage.mkdir(parents=True,exist_ok=True)
    runtime=RUN/'redesign_v7'
    if not runtime.exists():runtime.symlink_to(storage,target_is_directory=True)
    else:assert runtime.resolve()==storage
    manifest=json.loads((v6/'execution.json').read_text())
    prereg=public/'PREREG.md'
    manifest.update(version='v7', fast_mode='empirical_bayes_v7', active_root=str(runtime/'active'),
        public_root=str(public), launcher_root=str(runtime/'launchers'), prereg_path=str(prereg),
        prereg_sha256=file_sha(prereg), full_matrix_authorized=False,
        allowed_registered_jobs=record['selected_jobs']+[dict(pair=p,method='YESTRACK_BUDGET_PRIOR',rho=.3) for p in sorted(selected_pairs)],
        registered_tier=record['status'], maximum_GPUs=2,
        coordinator_script='scripts/active_v7_continue.py', launcher_script='scripts/active_v7_launch.py',
        report_script='scripts/active_v7_report.py',
        user_authorization='2026-11-05 final consolidated v7, 15 jobs or >20h twelve-job fallback; v6 C4-C6 and remaining80 denied',
        continuation_of=str(v6/'execution.json'),
        prior_v6_public_root=str(v6), prior_v6_active_root=str(RUN/'redesign_v6/active'),
        presence_logLR_cap=2., pi_soft_count_likelihood_unclipped=True, student_frozen_only_for_MINUS_SLOW=True,
        single_answer_identity_log_odds_cap=2., none_class_weight_floor=.2,
        original_BC_hyperparameters_unchanged=False,
        matrix_GPU_hour_gate=20,first_jobs='all six MINUS_SLOW, then registered FULL and fixed-question SLOW_REPLAY',
        registered_change='final v7 EB presence, asked-only .85/.15, positive rechecks, balanced v6 LoRA and fixed-evidence replay')
    cost=dict(total_GPU_hours_estimate=total,jobs=jobs,source='v7 budget_preflight.json; conditional YT upper included')
    atomic_json(public/'BC_cost_estimate.json',cost); atomic_json(public/'execution.json',manifest)
    atomic_json(public/'registered_jobs.json',dict(tier='MAIN_AND_CRITICAL_ONLY',scope='V7_'+record['status'],
        jobs=record['selected_jobs'],selected_total_GPU_hours_estimate=record['selected_GPU_hours_estimate'],
        cost_estimate_sha256=file_sha(public/'BC_cost_estimate.json'), prereg_path=str(prereg),
        prereg_sha256=file_sha(prereg), maximum_GPUs=2, remaining_matrix_not_authorized=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--estimate-only',action='store_true')
    prepare(parser.parse_args().estimate_only)
