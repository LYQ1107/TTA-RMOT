"""Estimate all ten v6 jobs first; enforce the user20h five-job fallback."""
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flex_uda.presence.runtime import RUN,target_metadata,runtime_guard
from scripts.active_run import PAIRS
from flex_tta.runtime.pf_receipts import atomic_json,file_sha


def sample32(frames):
    frames=np.asarray(sorted(frames),dtype=int)
    anchors=np.linspace(frames[0],frames[-1],min(32,len(frames)))
    return sorted({int(min(frames,key=lambda t:(abs(t-a),t))) for a in anchors})


def main():
    opened=runtime_guard();base=ROOT/'results/presence_active/redesign_v6';base.mkdir(exist_ok=True)
    previous=ROOT/'results/presence_active/redesign_v5/active_map_v1_critical28'
    review=json.loads((previous/'remaining_budget_review.json').read_text())
    rows=[];dimensions={}
    for pair,(model,target) in PAIRS.items():
        meta=target_metadata(target);work=sampled=budget=0;videos=[]
        for video,bank in sorted(meta['videos'].items()):
            md=json.loads((RUN/'features'/target/model/'videos'/video/'metadata.json').read_text())
            selected=sample32(md['frames']);lengths={i:0 for _,i in bank['observations']}
            for _,i in bank['observations']:lengths[i]+=1
            q=len(md['queries']);work+=q*len(bank['observations'])
            sample_work=q*sum(t in selected for t,_ in bank['observations']);sampled+=sample_work
            calls=int(np.floor(.3*q*sum(n>=3 for n in lengths.values())));budget+=calls
            videos.append(dict(video=video,queries=q,observations=len(bank['observations']),
                fixed_sample_frames=selected,sampled_candidate_query_rows=sample_work,budget=calls))
        unit=review['target_unit_costs'][target]
        dimensions[pair]=dict(target=target,model=model,videos=videos,all_candidate_query_rows=work,
                             sample32_candidate_query_rows=sampled,teacher_budget=budget)
        for method in ('FULL','MINUS_SLOW') if pair in ('C1','C3','C4','C6') else ('FULL',):
            full_seconds=work*unit['seconds_per_rescore_candidate_query']
            sample_seconds=4*sampled*unit['seconds_per_rescore_candidate_query'] if method=='FULL' else 0.
            slow_seconds=4*50*len(videos)*unit['seconds_per_update'] if method=='FULL' else 0.
            teacher_seconds=budget*unit['seconds_per_teacher_call']
            rows.append(dict(pair=pair,method=method,rho=.3,full_frame_score_GPU_hours=full_seconds/3600,
                sample32_score_GPU_hours=sample_seconds/3600,LoRA_GPU_hours=slow_seconds/3600,
                teacher_equivalent_GPU_hours=teacher_seconds/3600,
                total_GPU_hours_estimate=(full_seconds+sample_seconds+slow_seconds+teacher_seconds)/3600))
    assert len(rows)==10
    total=sum(r['total_GPU_hours_estimate'] for r in rows)
    fallback=[dict(pair=p,method=m,rho=.3) for p,m in
              [('C1','FULL'),('C2','FULL'),('C3','FULL'),('C1','MINUS_SLOW'),('C3','MINUS_SLOW')]]
    jobs=fallback if total>20 else [{k:r[k] for k in ('pair','method','rho')} for r in rows]
    result=dict(status='FIVE_JOB_FALLBACK' if total>20 else 'TEN_JOBS_WITHIN20H',
        ten_job_GPU_hours_estimate=total,user_gate_GPU_hours=20,selected_jobs=jobs,
        selected_GPU_hours_estimate=sum(r['total_GPU_hours_estimate'] for r in rows
            if {k:r[k] for k in ('pair','method','rho')} in jobs),estimates=rows,dimensions=dimensions,
        estimate_source=str(previous/'remaining_budget_review.json'),estimate_source_sha256=file_sha(previous/'remaining_budget_review.json'),
        limitations=['Conservative v5 same-query-batch8 measured rescore unit, extrapolated to all initial frames and fixed32 refreshes.',
            'Shared GPU elapsed phase estimate, not utilization integral; no cache/skip discount; loading/CPU decoding overhead excluded.',
            'No new GPU timing before the user scope gate; actual counters will be reported separately.'],
        GT_open_attempts=len(opened),new_GPU_or_teacher_calls=0,
        estimated_frame_cache_bytes=sum(dimensions[j['pair']]['all_candidate_query_rows']*4*769 for j in jobs))
    assert not opened
    atomic_json(base/'budget_preflight.json',result)
    old_path=ROOT/'results/presence_active/redesign_v5/execution_map_v1_critical18.json'
    manifest=json.loads(old_path.read_text())
    prior_root=manifest['active_root'];prereg=base/'PREREG.md'
    manifest.update(version='v6',fast_mode='fraction_prior_v6',active_root=str(RUN/'redesign_v6/active'),
        public_root=str(base),launcher_root=str(RUN/'redesign_v6/launchers'),
        prereg_path=str(prereg),prereg_sha256=file_sha(prereg),
        allowed_registered_jobs=jobs+[dict(pair=p,method='YESTRACK_BUDGET_PRIOR',rho=.3)
            for p in sorted({j['pair'] for j in jobs if j['method']=='FULL'})],
        conditional_YT_PRIOR_call_change_threshold=.05,prior_v5_active_root=prior_root,
        prior_v5_public_root=str(previous),full_matrix_authorized=False,
        coordinator_script='scripts/active_v6_continue.py',report_script='scripts/active_v6_report.py',
        user_authorization='2026-11-04 v6 four repairs and explicit20h/five-job fallback; v5 remaining80 denied',
        pilot_only=False,registered_tier=result['status'],continuation_of=str(old_path),
        frame_refresh='video-start full current-model cache; fixed32 after LoRA; cached re-decode after fast',
        maximum_GPUs=2)
    cost=dict(total_GPU_hours_estimate=total,jobs=rows,source='v6 budget_preflight.json; conservative no cache/skip discount')
    atomic_json(base/'BC_cost_estimate.json',cost)
    plan=dict(tier='MAIN_AND_CRITICAL_ONLY',scope='V6_'+result['status'],jobs=jobs,
        selected_total_GPU_hours_estimate=result['selected_GPU_hours_estimate'],
        cost_estimate_sha256=file_sha(base/'BC_cost_estimate.json'),prereg_path=str(prereg),
        prereg_sha256=file_sha(prereg),maximum_GPUs=2,remaining_matrix_not_authorized=True)
    atomic_json(base/'execution.json',manifest);atomic_json(base/'registered_jobs.json',plan)
    print(json.dumps({k:v for k,v in result.items() if k not in ('dimensions','estimates')},indent=2))


if __name__=='__main__':main()
