"""Same-CPU original SGD control for the separate observer optimizer pilot.

GPU-vs-CPU solver fits are not a pure optimizer intervention. Preserve the
running two-branch pilot and add an explicitly registered original CPU arm.
"""
import hashlib
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json
from scripts import a100_rsi_policy_distribution_pilot as pilot

PLAN=pilot.PUBLIC/'rsi_policy_distribution_CPU_control_registration_20261124.json'


def main():
    plan=json.loads(PLAN.read_text())
    for n,s in plan['source_sha256'].items():assert hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==s,n
    pilot.check_plan()
    from flex_uda.active import rsi_policy_distribution as policy
    policy.trust_update=policy.SGD_UPDATE
    policy.gated_update=lambda update,credit,*args,**kwargs:update(*args,**kwargs)
    print(json.dumps(dict(status='RUNNING_MATCHED_CPU_ORIGINAL_SGD_CONTROL',pid=os.getpid(),GT_used=False)),flush=True)
    pilot.worker('CPU_ORIGINAL_SGD')
    print(json.dumps(dict(status='COMPLETE_MATCHED_CPU_ORIGINAL_SGD_CONTROL_NOT_RMOT_QUALITY')),flush=True)


if __name__=='__main__':main()
