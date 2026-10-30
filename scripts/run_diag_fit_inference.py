#!/usr/bin/env python3
"""Only after verified Table A publication: final-step full425 then seal."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flex_uda.diag_fit import require_published_fit_table
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from scripts.run_psr_matrix import run


if __name__=='__main__':
    root=ROOT/'results/runs/diag_fit_20261030_v1'
    gate=require_published_fit_table(root)
    completion=json.loads((root/'full_completion.json').read_text())
    methods=[m for m,r in completion['arms'].items() if r['status']=='COMPLETE']
    if methods:run(root,methods,workers=min(3,len(methods)))
    sealed={}
    for method in methods:
        folder=root/'evaluation/merged'/method;receipt=json.loads((folder/'run_metadata.json').read_text())
        if receipt['valid_episodes']!=425 or receipt['selected_step']!=8973 or receipt['selection']!='FINAL_STEP' or receipt['target_GT_open_attempts']:
            raise ValueError('not a complete final-step-only D-FIT evaluation')
        sealed[method]=dict(run_metadata=str(folder/'run_metadata.json'),run_metadata_sha256=file_sha(folder/'run_metadata.json'),
            full_audit_sha256=file_sha(folder/'full_audit.json'),checkpoint_sha256=receipt['checkpoint_sha256'],
            episodes=[dict(prediction_path=r['prediction_path'],prediction_sha256=r['prediction_sha256'],
                margins_path=r['margins_path'],margins_sha256=r['margins_sha256']) for r in receipt['episodes']])
    atomic_json(root/'predictions_seal.json',dict(status='ALL_TRAINABLE_ARMS_25VIDEO_425QUERY_FINAL_STEP_SEALED',
        methods=sealed,untrainable_methods={m:r for m,r in completion['arms'].items() if r['status']!='COMPLETE'},
        target_GT_open_attempts=0,prior_Table_A_publication=gate,full_predictions_not_for_git=True))
    print(json.dumps(dict(event='D_FIT_ALL_FULL_PREDICTIONS_SEALED',methods=methods)),flush=True)
