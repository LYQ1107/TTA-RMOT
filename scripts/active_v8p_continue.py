"""Stage-A continuation; the preregistered gate decides B, never old stage4."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flex_uda.active.execution import authorize_active
from flex_tta.runtime.pf_receipts import atomic_json
from scripts.active_continue import run


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--jobs-json',type=Path,required=True);a=p.parse_args()
    execution,_=authorize_active();assert execution.manifest['version']=='v8P'
    run(execution,a.jobs_json)
    gate=json.loads((execution.public/'stage_A_gate.json').read_text())
    atomic_json(execution.public/'queue_completion.json',dict(status='V8P_STAGE_A_COMPLETE',
        gate=gate,original_v8_stage3_unmodified=True,old80_launched=False))
    print(json.dumps(dict(event='V8P_STAGE_A_COMPLETE',gate=gate)),flush=True)
    if gate['result']=='PASS':
        # This continuation is installed before any A result. B has its own
        # immutable plan and reuses A rather than resuming the old 80 jobs.
        from scripts.active_v8p_stage_b import run_conditional
        run_conditional(execution)
    else:
        print(json.dumps(dict(event='V8P_STOP_STAGE_A_FAILED_NO_STAGE_B')),flush=True)


if __name__=='__main__':main()
