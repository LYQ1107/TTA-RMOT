"""Run only the cost-gated v6 set and conditionally rematch YT_PRIOR."""
from argparse import ArgumentParser
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flex_uda.active.execution import authorize_active
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from scripts.active_continue import run


def main():
    p=ArgumentParser(description=__doc__);p.add_argument('--jobs-json',type=Path,required=True);a=p.parse_args()
    execution,_=authorize_active();run(execution,a.jobs_json)
    old=Path(execution.manifest['prior_v5_active_root']);decisions=[];follow=[]
    for job in json.loads(a.jobs_json.read_text())['jobs']:
        if job['method']!='FULL':continue
        pair=job['pair'];new=execution.active/pair/'FULL/rho_0.3';previous=old/pair/'FULL/rho_0.3'
        n=json.loads((new/'predictions_seal.json').read_text())['teacher']['logical_calls']
        b=json.loads((previous/'predictions_seal.json').read_text())['teacher']['logical_calls']
        delta=abs(n-b)/b;required=delta>.05
        decisions.append(dict(pair=pair,v6_calls=n,v5_calls=b,absolute_relative_change=delta,
            rerun_YT_PRIOR=required,rule='strict >5 percent total change; rematch per-video if required'))
        if required:follow.append(dict(pair=pair,method='YESTRACK_BUDGET_PRIOR',rho=.3))
    atomic_json(execution.public/'baseline_budget_decisions.json',dict(decisions=decisions,jobs=follow,
        GT_or_model_calls_for_decision=0,remaining_v5_80_not_authorized=True))
    if follow:
        plan=json.loads(a.jobs_json.read_text());plan.update(scope='V6_CONDITIONAL_YT_PRIOR',jobs=follow)
        path=execution.public/'conditional_YT_PRIOR_jobs.json';atomic_json(path,plan);run(execution,path)
    from scripts.active_v6_report import report
    report(execution)
    atomic_json(execution.public/'queue_completion.json',dict(status='V6_AUTHORIZED_SET_COMPLETE',
        primary_jobs=len(json.loads(a.jobs_json.read_text())['jobs']),conditional_YT_PRIOR_jobs=len(follow),
        remaining_v5_80_started=False,unapproved_v6_jobs_started=False))


if __name__=='__main__':main()
