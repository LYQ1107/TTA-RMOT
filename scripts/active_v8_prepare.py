"""v8 fixed stage2/3 plan, cost and storage admission; no GPU/GT input."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from flex_uda.presence.runtime import RUN,runtime_guard,target_metadata,sealed_margins
from scripts.active_run import PAIRS


def prepare(estimate_only=False):
    attempts=runtime_guard();public=ROOT/'results/presence_active/redesign_v8'
    registration=json.loads((public/'manifest.json').read_text())
    stage0=public/'amendments_A2_A3'
    assert json.loads((stage0/'stage0_v7_reproduction.json').read_text())['status']=='PASS'
    assert json.loads((stage0/'stage0_offline_em.json').read_text())['status']=='PASS'
    assert json.loads((stage0/'stage0_GT_diagnostics.json').read_text())['status']=='COMPLETE'
    assert json.loads((stage0/'cpu_tests.json').read_text())['failed']==0
    old=json.loads((ROOT/'results/presence_active/redesign_v7/execution.json').read_text())
    units=json.loads((ROOT/'results/presence_active/redesign_v5/active_map_v1_critical28/remaining_budget_review.json').read_text())['target_unit_costs']
    bounds={r['pair']:r for r in json.loads((ROOT/'results/presence_active/redesign_v6/budget_preflight.json').read_text())['estimates'] if r['method']=='FULL'}
    dimensions={};inputs=[];jobs=[]
    for pair,(model,target) in PAIRS.items():
        meta=target_metadata(target);margin,seal=sealed_margins(target,model)
        budget=0
        for video,bank in sorted(meta['videos'].items()):
            _,counts=np.unique(np.asarray(bank['observations'])[:,1],return_counts=True)
            nq=sum(q['video_id']==video for q in meta['episodes'])
            budget+=int(np.floor(.3*nq*int((counts>=3).sum())))
        dimensions[pair]=dict(target=target,model=model,videos=len(meta['videos']),episodes=len(meta['episodes']),
            candidate_query_rows=len(margin),teacher_budget=budget)
        inputs.append(dict(pair=pair,margin=seal))
    for registered in registration['allowed_registered_jobs']:
        pair=registered['pair'];method=registered['method'];d=dimensions[pair]
        teacher=d['teacher_budget']*units[d['target']]['seconds_per_teacher_call']/3600
        bound=bounds[pair]
        jobs.append(dict(pair=pair,method=method,rho=.3,
            total_GPU_hours_estimate=teacher if method=='MINUS_SLOW' else
                bound['LoRA_GPU_hours']+bound['full_frame_score_GPU_hours']+bound['sample32_score_GPU_hours']+(0 if method=='SLOW_REPLAY' else teacher)))
    frame_bytes=sum(dimensions[j['pair']]['candidate_query_rows']*4*769 for j in jobs if j['method']!='MINUS_SLOW')
    storage=Path('/data1/user/TTA-RMOT-assets/presence_active_20261102/redesign_v8')
    available=shutil.disk_usage(storage.parent).free
    cost=dict(version='v8',jobs=jobs,total_GPU_hours_estimate=sum(j['total_GPU_hours_estimate'] for j in jobs),
        dimensions=dimensions,frame_cache_bytes_estimate=frame_bytes,data1_available_bytes=available,
        data2_available_bytes=shutil.disk_usage('/data2').free,cache_root=str(storage),
        scope='six MINUS, FULL C1/C4/C6, REPLAY C1/C4; no discount for cache/skip/pilots',
        maximum_GPUs=2,GT_open_attempts=len(attempts),GPU_or_teacher_calls=0,
        stage4_not_authorized_until_main_gate_passes=True)
    atomic_json(public/'budget_preflight.json',cost);atomic_json(public/'inputs.json',dict(files=inputs,GT_open_attempts=len(attempts)))
    atomic_json(public/'BC_cost_estimate.json',cost)
    print(json.dumps({k:v for k,v in cost.items() if k not in ('jobs','dimensions')},indent=2),flush=True)
    if estimate_only:return
    assert frame_bytes+(4<<30)<available,'Insufficient new-cache storage; preserve all historical assets'
    storage.mkdir(parents=True,exist_ok=True);runtime=RUN/'redesign_v8'
    if not runtime.exists():runtime.symlink_to(storage,target_is_directory=True)
    else:assert runtime.resolve()==storage
    manifest=dict(old)
    manifest.update(version='v8',active_root=str(runtime/'active'),public_root=str(public),
        launcher_root=str(runtime/'launchers'),fast_mode='presence_em_v8',
        prereg_path=str(public/'PREREG.md'),prereg_sha256=file_sha(public/'PREREG.md'),
        registration_path=str(public/'manifest.json'),registration_sha256=file_sha(public/'manifest.json'),
        amendments=registration['amendments'],maximum_GPUs=2,allowed_registered_jobs=[{k:j[k] for k in ('pair','method','rho')} for j in jobs],
        coordinator_script='scripts/active_v8_continue.py',launcher_script='scripts/active_v8_launch.py',report_script='scripts/active_v8_report.py',
        user_authorization='v8 original stages0-3 plus pre-real-data A1/A2/A3; stage4 only if registered main gate passes',
        prior_v7_public_root=str(ROOT/'results/presence_active/redesign_v7'),prior_v7_active_root=old['active_root'],
        original_v8_registration_remote_commit='c22d9fbda8d1a71e1a85163cf3ff29acf6d52d41',
        registered_tier='V8_STAGE2_3_ELEVEN_JOBS',full_matrix_authorized=False,
        EM=dict(iters=5000,tol=1e-7,Beta_prior=[2,2],clip=[.05,.95],track_cap=2),
        pi_identification_minimum_answered_pairs=200)
    atomic_json(public/'execution.json',manifest)
    atomic_json(public/'pilot_execution.json',dict(manifest,coordinator_script='scripts/active_v8_smoke.py'))
    plan=dict(tier='V8_STAGE2_3',jobs=manifest['allowed_registered_jobs'],
        maximum_GPUs=2,prereg_path=str(public/'PREREG.md'),prereg_sha256=file_sha(public/'PREREG.md'),
        cost_estimate_sha256=file_sha(public/'BC_cost_estimate.json'),remaining_matrix_not_authorized=True)
    atomic_json(public/'registered_jobs.json',plan)
    atomic_json(public/'pilot_jobs.json',dict(plan,jobs=[j for j in plan['jobs'] if j['pair'] in ('C1','C2') and j['method']=='MINUS_SLOW']))
    assert not attempts


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--estimate-only',action='store_true')
    prepare(p.parse_args().estimate_only)
