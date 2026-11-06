"""Post-tier v8 close; conditional stage4 requires a genuine main PASS."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flex_uda.active.execution import authorize_active
from flex_tta.runtime.pf_receipts import atomic_json


def close(execution):
    completion=json.loads((execution.public/'queue_completion.json').read_text())
    assert completion['status']=='V8_STAGE2_3_COMPLETE'
    assert json.loads((execution.public/'validation.json').read_text())['status']=='PASS'
    gate=json.loads((execution.public/'verdict.json').read_text())
    state=dict(status='STAGE2_3_COMPLETE_GATE_PASS' if gate['main']['result']=='PASS' else 'COMPLETE_STOPPED_MAIN_GATE_FAIL',
        main_gate=gate['main'],slow_gate=gate['slow'],stage4_launched=False,
        old80_or_v6_extension_started=False)
    atomic_json(execution.public/'final_stage_status.json',state)
    print(json.dumps(state),flush=True);return state


if __name__=='__main__':execution,_=authorize_active();close(execution)
