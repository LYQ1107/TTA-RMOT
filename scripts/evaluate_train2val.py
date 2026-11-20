#!/usr/bin/env python3
"""Independent post-seal official evaluation, reusing the v2 evaluator unchanged."""
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from scripts import diag_attrib_v2_common as common
from scripts import evaluate_diag_attrib_v2 as evaluator
from scripts.train2val_controls import REFERENCES,OLD,verify_native_predictions
from scripts.run_train2val_training import RUN,PUBLIC


def gate():
    value=json.loads((RUN/'fit_table_publication_gate.json').read_text())
    if not value['remote_HEAD_verified'] or file_sha(PUBLIC/'TABLE_A.md')!=value['FIT_TABLE_sha256']:
        raise ValueError('prior Train table publication changed')
    return value


def evaluate():
    gate();verify_native_predictions()
    planned=json.loads((RUN/'controls_seal.json').read_text())
    if planned['GT_free_target_open_attempts']:raise ValueError('control generator read GT')
    # Only output/gate bindings differ; TrackEval command, metrics, filtering,
    # seal verification and completion-resume logic are the existing v2 code.
    common.EVAL=RUN/'evaluation';evaluator.RUN=RUN;evaluator.EVAL=common.EVAL;evaluator.require_prereg=gate
    for name,original in REFERENCES.items():common.reuse_metrics(name,original)
    common.reuse_metrics('FROZEN_AT_OFFICIAL_COUNT',OLD/'M_FROZEN__OFFICIAL_DANCE_COMPLETE')
    new=[m for m in planned['methods'] if m not in planned['reused_reference_methods']]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for future in as_completed([pool.submit(evaluator.evaluate,m) for m in new]):future.result()
    results={m:json.loads((common.EVAL/m/'official_completion.json').read_text()) for m in planned['methods']}
    atomic_json(RUN/'official_evaluation_completion.json',dict(status='COMPLETE_ALL_PREDICTION_SETS',
        new_official_jobs=len(new),reused_exactly_equivalent_jobs=len(results)-len(new),methods=results))
    print(json.dumps(dict(event='T2V_OFFICIAL_ALL_COMPLETE',new_jobs=len(new),reused_jobs=len(results)-len(new))),flush=True)


if __name__=='__main__':evaluate()
