"""One immutable eleven-job v8 tier; no v7 conditional extra methods."""
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
    execution,_=authorize_active();assert execution.manifest['version']=='v8'
    assert json.loads((execution.public/'gpu_smoke.json').read_text())['status']=='V8_TWO_DIRECTION_TWO_VIDEO_PILOTS_PASS'
    run(execution,a.jobs_json)
    from scripts.active_v8_validate import validate
    validate(execution)
    from scripts.active_v8_report import report
    result=report(execution)
    atomic_json(execution.public/'queue_completion.json',dict(status='V8_STAGE2_3_COMPLETE',**result,
        stage4_launched=False,old80_or_v6_expansion_started=False))
    print(json.dumps(dict(event='V8_STAGE2_3_COMPLETE',**result)),flush=True)


if __name__=='__main__':main()
