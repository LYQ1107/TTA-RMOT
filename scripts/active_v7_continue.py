"""Exactly the v7 cost-gated tier, followed only by conditional YT_PRIOR."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flex_uda.active.execution import authorize_active
from flex_tta.runtime.pf_receipts import atomic_json
from scripts.active_continue import run


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--jobs-json',type=Path,required=True);a=p.parse_args()
    execution,_=authorize_active();run(execution,a.jobs_json)
    plan=json.loads(a.jobs_json.read_text());decisions=[];follow=[]
    for job in plan['jobs']:
        if job['method']!='FULL':continue
        pair=job['pair'];new=execution.active/pair/'FULL/rho_0.3'
        old=Path(execution.manifest['prior_v5_active_root'])/pair/'YESTRACK_BUDGET_PRIOR/rho_0.3'
        n=json.loads((new/'predictions_seal.json').read_text())['teacher']['logical_calls']
        previous=json.loads((old/'predictions_seal.json').read_text())['teacher']['logical_calls']
        change=abs(n-previous)/previous;required=change>.05
        decisions.append(dict(pair=pair,v7_FULL_calls=n,reference_YT_PRIOR_calls=previous,
            relative_change=change,rerun_YT_PRIOR=required))
        if required:follow.append(dict(pair=pair,method='YESTRACK_BUDGET_PRIOR',rho=.3))
    atomic_json(execution.public/'baseline_budget_decisions.json',dict(decisions=decisions,jobs=follow))
    if follow:
        plan.update(scope='V7_CONDITIONAL_YT_PRIOR',jobs=follow)
        path=execution.public/'conditional_YT_PRIOR_jobs.json';atomic_json(path,plan);run(execution,path)
    from scripts.active_v7_report import report
    report(execution)
    atomic_json(execution.public/'queue_completion.json',dict(status='V7_AUTHORIZED_TIER_COMPLETE',
        primary_jobs=len(json.loads(a.jobs_json.read_text())['jobs']),conditional_YT_PRIOR_jobs=len(follow),
        v6_C4_C6_started=False,remaining_v5_80_started=False,unselected_v7_jobs_started=False))


if __name__=='__main__':main()
