"""Critical28 accounting from sealed reports/receipts; no new GT/model calls."""
import csv
from collections import Counter,defaultdict
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from flex_uda.active.execution import active_paths
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from scripts.active_plan import jobs,dimensions
from scripts.active_run import PAIRS
from scripts.report_active import write_table


def read(path):
    with Path(path).open() as stream:return list(csv.DictReader(stream))


def main():
    p=active_paths();plan=json.loads((p.public/'registered_jobs.json').read_text())
    start=json.loads((p.public/'execution_started.json').read_text())
    end=json.loads((Path(start['durable_dir'])/'completion.json').read_text())
    assert end['exit_code']==0
    rows=[r for r in read(p.public/'main.csv') if r['method']!='GATE_REASSIGN_REF']
    costs=[r for r in read(p.public/'costs.csv') if r['method']!='GATE_REASSIGN_REF']
    comparisons=json.loads((p.public/'bootstrap.json').read_text())['comparisons']
    confidence=read(p.public/'final_confidence.csv');calibration=read(p.public/'calibration_calls.csv')
    recovery=read(p.public/'recovery_FP_attribution.csv');video=read(p.public/'video_differences.csv')
    direction=[];slow=[];mechanism=[];units=defaultdict(list)
    full=[r for r in rows if r['method']=='FULL']
    for r in full:
        pair,target=r['pair'],r['target'];hota=float(r['HOTA_unrounded'])
        method_rows={v['method']:v for v in rows if v['pair']==pair}
        compare={k.split('FULL_minus_')[-1]:v for k,v in comparisons.items() if k.startswith(pair+'/')}
        record=dict(pair=pair,source=r['model'],target=target,
            NATIVE=hota-compare['NATIVE']['delta_HOTA'],PRES_TOPK=hota-compare['PRES_TOPK']['delta_HOTA'],
            **{m:float(v['HOTA_unrounded']) for m,v in method_rows.items()},
            delta_NATIVE=compare['NATIVE']['delta_HOTA'],delta_NATIVE_CI95=compare['NATIVE']['CI95'],
            delta_YT_PRIOR=compare['YESTRACK_BUDGET_PRIOR']['delta_HOTA'],
            delta_YT_PRIOR_CI95=compare['YESTRACK_BUDGET_PRIOR']['CI95'])
        record['above_all_simple_controls']=all(hota>float(method_rows[m]['HOTA_unrounded']) for m in
            ('UNIFORM_K','YESTRACK_BUDGET_RANDOM','YESTRACK_BUDGET_PRIOR'))
        direction.append(record)
        if 'MINUS_SLOW' in method_rows:
            slow.append(dict(pair=pair,target=target,**compare['MINUS_SLOW']))
        f=next(v for v in confidence if v['pair']==pair and v['method']=='FULL')
        c=[v for v in calibration if v['pair']==pair and v['method']=='FULL']
        a=next(v for v in recovery if v['pair']==pair and v['method']=='FULL')
        gains={m:Counter('up' if float(v['delta_HOTA'])>0 else 'down' if float(v['delta_HOTA'])<0 else 'equal'
            for v in video if v['pair']==pair and v['comparison']==m) for m in sorted({v['comparison'] for v in video if v['pair']==pair})}
        mechanism.append(dict(pair=pair,target=target,final_confidence=f,recovery=a,
            fast_calls=len(c),identity_status_counts=dict(Counter(v['identity_status'] for v in c)),
            presence_status_counts=dict(Counter(v['presence_status'] for v in c)),video_gains=gains))
        cost=next(v for v in costs if v['pair']==pair and v['method']=='FULL')
        receipts=[json.loads(path.read_text()) for path in (Path(r['path'])/'videos').glob('*/receipt.json')]
        updates=int(cost['optimizer_updates']);work=sum(v['rescore_candidate_query_rows'] for v in receipts)
        units[target].append(dict(pair=pair,seconds_per_update=float(cost['LoRA_GPU_hours'])*3600/updates,
            seconds_per_rescore_candidate_query=float(cost['rescore_GPU_hours'])*3600/work,
            seconds_per_teacher_call=float(cost['committed_teacher_equivalent_forward_GPU_hours'])*3600/int(cost['teacher_calls'])))
    target_units={}
    for target,values in units.items():
        u={k:max(v[k] for v in values) for k in ('seconds_per_update','seconds_per_rescore_candidate_query','seconds_per_teacher_call')}
        pairs={v['pair'] for v in values}
        u['seconds_per_CROP_call']=max(float(v['committed_teacher_equivalent_forward_GPU_hours'])*3600/int(v['teacher_calls'])
            for v in costs if v['pair'] in pairs and v['method'].startswith('YESTRACK_BUDGET'))
        target_units[target]=u
    completed={(r['pair'],r['method'],float(r['rho'])) for r in rows};estimates=[]
    for job in jobs():
        target=PAIRS[job['pair']][1];ds=dimensions(target);u=target_units[target];method=job['method']
        calls=(sum(d['yestrack_full_d10_calls'] for d in ds) if method=='YESTRACK_FULL_D10' else
            sum(int(np.floor(job['rho']*d['eligible_tracks']*d['queries'])) for d in ds))
        no_slow=method.startswith('YESTRACK') or method in ('UNIFORM_K','MINUS_SLOW')
        teacher=calls*u['seconds_per_CROP_call' if method.startswith('YESTRACK') else 'seconds_per_teacher_call']/3600
        training=0 if no_slow else 4*50*len(ds)*u['seconds_per_update']/3600
        rescore=0 if no_slow else 5*sum(d['rescore_rows'] for d in ds)*u['seconds_per_rescore_candidate_query']/3600
        estimates.append(dict(job,target=target,completed=(job['pair'],method,job['rho']) in completed,
            total_GPU_hours_estimate=teacher+training+rescore))
    new_keys={(j['pair'],j['method'],j['rho']) for j in plan['jobs']}
    new_costs=[v for v in costs if (v['pair'],v['method'],float(v['rho'])) in new_keys]
    cost_fields=('teacher_forward_GPU_hours','committed_teacher_equivalent_forward_GPU_hours','LoRA_GPU_hours',
        'rescore_GPU_hours','assigned_worker_wall_hours','optimizer_updates')
    totals=lambda selected:{k:sum(float(v[k]) for v in selected) for k in cost_fields}
    launch=json.loads((Path(start['durable_dir'])/'launch.json').read_text())
    budget=dict(status='BATCH18_FINISHED_NEXT80_NOT_AUTHORIZED',completed=28,remaining=80,
        old_cost_estimates_not_overwritten=True,original_registered_full108_GPU_hours=292.88630367713597,
        previous_full108_GPU_hours=277.4612638401637,previous_remaining80_GPU_hours=241.6491770744947,
        target_specific_updated_full108_GPU_hours=sum(v['total_GPU_hours_estimate'] for v in estimates),
        target_specific_updated_remaining80_GPU_hours=sum(v['total_GPU_hours_estimate'] for v in estimates if not v['completed']),
        original_cost_gate_GPU_hours=40,newly_authorized_remaining_jobs=0,
        new18_totals=totals(new_costs),all28_totals=totals(costs),
        new18_wall_hours=(end['finished_time_ns']-launch['launched_time_ns'])/3.6e12,
        new18_failed_attempts=sum(int(v['failed_worker_attempts']) for v in new_costs),
        target_unit_costs=target_units,unit_measurements=units,estimates=estimates,
        limitations=['no-cache equivalent teacher cost, all4x50 steps and5 rescores/video; no skip discounts',
            'worst of two measured sources per target; no longer substitutes Dance/Sports for unmeasured KITTI',
            'shared-device elapsed phase time, not a GPU utilization integral',
            'new18 wall includes post-seal CPU evaluation/reporting; no matched serial baseline, so not measured parallel speedup',
            'shared preparation/discarded historical heads and unmeasured failure costs remain separate',
            'cost-only metadata, no target GT or HOTA used to change experiment parameters'])
    verdict=dict(status='DESCRIPTIVE_CRITICAL28_NOT_FULL108_ACCEPTANCE',above_NATIVE=sum(v['delta_NATIVE']>0 for v in direction),
        below_NATIVE=sum(v['delta_NATIVE']<0 for v in direction),above_all_simple_controls=sum(v['above_all_simple_controls'] for v in direction),
        directions=direction,slow_comparisons=slow,mechanism=mechanism,
        mechanism_oracle_note='GT in these tables was computed only offline after sealing; final belief is not optimizer sample labels',
        labels=['Dance-only advantage over all current simple controls; not a generally validated TTA method',
            'slow loop has stable positive conditional CI only in C4 among the4 tested directions; C3 is negative',
            'KITTI4 videos point estimates only; no seed/order uncertainty or blind-test claim'],
        new_model_teacher_or_GT_calls=0)
    atomic_json(p.public/'computed_verdict.json',verdict);atomic_json(p.public/'remaining_budget_review.json',budget)
    write_table(p.public/'direction_summary.csv',direction);write_table(p.public/'slow_summary.csv',slow)
    print(json.dumps(dict(directions=direction,slow_comparisons=slow,
        remaining80_GPU_hours=budget['target_specific_updated_remaining80_GPU_hours'],
        full108_GPU_hours=budget['target_specific_updated_full108_GPU_hours'],new18_cost=budget['new18_totals'],
        new18_wall_hours=budget['new18_wall_hours'],new18_failed_attempts=budget['new18_failed_attempts']),indent=2))


if __name__=='__main__':main()
