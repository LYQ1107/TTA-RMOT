"""Prepare only the explicitly authorized next 18 registered jobs; no GPU/GT."""
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.active_plan import jobs
from flex_tta.runtime.pf_receipts import atomic_json,file_sha


def additional_jobs():
    methods=('FULL','MINUS_SLOW','UNIFORM_K','YESTRACK_BUDGET_RANDOM','YESTRACK_BUDGET_PRIOR')
    selected=[j for j in jobs() if j['pair'] in ('C2','C4','C5','C6') and j['rho']==.3 and j['method'] in methods]
    assert len(selected)==18
    return selected


def main():
    base=ROOT/'results/presence_active/redesign_v5'
    previous=base/'active_map_v1_full';public=base/'active_map_v1_critical28'
    manifest_path=base/'execution_map_v1_critical18.json'
    m=json.loads((base/'execution_map_v1_full.json').read_text())
    review=json.loads((previous/'measured_budget_review.json').read_text())
    prior_plan=json.loads((previous/'registered_jobs.json').read_text())
    selected=additional_jobs()
    estimates=[r for r in review['estimates'] if {k:r[k] for k in ('pair','method','rho')} in selected]
    assert len(estimates)==18
    total=sum(r['total_GPU_hours_estimate'] for r in estimates)
    assert total==review['additional_main_critical_subset_GPU_hours_estimate']
    preserved=[]
    for j in prior_plan['jobs']:
        path=Path(m['active_root'])/j['pair']/j['method']/('rho_'+str(j['rho']))/'predictions_seal.json'
        seal=json.loads(path.read_text())
        assert seal['GT_open_attempts']==0 and seal['episode_count'] in (425,466)
        preserved.append(dict(j,path=str(path),sha256=file_sha(path)))
    cost_path=public/'BC_cost_estimate.json'
    if not cost_path.exists():shutil.copyfile(previous/'BC_cost_estimate.json',cost_path)
    assert file_sha(cost_path)==file_sha(previous/'BC_cost_estimate.json')
    prereg=public/'PREREG.md'
    m.update(public_root=str(public),launcher_root=str(Path(m['launcher_root']).with_name('active_map_critical18_launchers')),
        registered_tier='ADDITIONAL_CRITICAL18',allowed_registered_jobs=selected,
        prereg_path=str(prereg),prereg_sha256=file_sha(prereg),
        continuation_of=str(base/'execution_map_v1_full.json'),full_matrix_authorized=False,
        user_authorization='2026-11-04 user: 先跑18项',prior_ten_results_immutable=True)
    plan=dict(tier='MAIN_AND_CRITICAL_ONLY',scope='ADDITIONAL_CRITICAL18',jobs=selected,
        cost_estimate_sha256=file_sha(cost_path),selected_total_GPU_hours_estimate=total,
        measured_cost_review_path=str(previous/'measured_budget_review.json'),
        measured_cost_review_sha256=file_sha(previous/'measured_budget_review.json'),
        full_matrix_GPU_hours_estimate=review['updated_full108_GPU_hours_estimate'],
        estimate_is_conservative_no_cache_and_does_not_deduct_finished_first_videos=True,
        prereg_path=str(prereg),prereg_sha256=file_sha(prereg),maximum_GPUs=2,
        remaining_matrix_not_authorized=True,remaining_after_this_tier=80,
        user_authorization=m['user_authorization'],prior_completed_jobs=preserved,
        after_completion_registered_jobs_covered=28)
    for path,record in ((manifest_path,m),(public/'registered_jobs.json',plan)):
        if path.exists() and json.loads(path.read_text())!=record:
            raise ValueError('Do not overwrite a different sealed plan: '+str(path))
        atomic_json(path,record)
    print(json.dumps(dict(manifest=str(manifest_path),plan=str(public/'registered_jobs.json'),
        jobs=len(selected),estimated_GPU_hours=total,GT_or_GPU_calls=0,preserved_jobs=len(preserved)),indent=2))


if __name__=='__main__':main()
