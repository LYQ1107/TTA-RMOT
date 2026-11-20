#!/usr/bin/env python3
"""Audit actual smoke losses/updates, not just a worker's exit status."""
import argparse
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json
from flex_uda.training import METHODS


def audit(root):
    rows=[]
    for method in METHODS:
        folder=Path(root)/method
        receipt=json.loads((folder/'training_receipt.json').read_text())
        records=[json.loads(line) for line in (folder/'metrics_train.jsonl').read_text().splitlines()]
        bad=[r['step'] for r in records if any(not math.isfinite(r[k]) for k in ('source_loss','gradient_norm') if k in r)
            or ('target_loss' in r and not math.isfinite(r['target_loss']))]
        target_ok=method=='SOURCE_REHEARSAL' or any(x>0 and math.isfinite(x) for x in receipt['target_smoke_gradient_norms'])
        valid=not bad and target_ok and receipt['steps']==20 and receipt['frozen_source_unchanged'] and not receipt['target_GT_open_attempts']
        rows.append(dict(method=method,status='PASS' if valid else 'INVALID_SMOKE',nonfinite_loss_steps=bad,
            source_loss_range=[min(r['source_loss'] for r in records),max(r['source_loss'] for r in records)] if not bad else None,
            actual_updates=receipt['counters']['optimizer_updates'],target_gradients_valid=target_ok,
            adapter_delta_L2=receipt['adapter_delta_L2'],frozen_source_unchanged=receipt['frozen_source_unchanged'],
            target_GT_open_attempts=receipt['target_GT_open_attempts']))
    result=dict(status='PASS_ALL_FOUR_REAL_SMOKES' if all(r['status']=='PASS' for r in rows) else 'DO_NOT_START_FULL_TRAINING',methods=rows)
    atomic_json(Path(root)/'independent_smoke_audit.json',result)
    print(json.dumps(result,indent=2));return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    raise SystemExit(0 if audit(a.root)['status']=='PASS_ALL_FOUR_REAL_SMOKES' else 1)
