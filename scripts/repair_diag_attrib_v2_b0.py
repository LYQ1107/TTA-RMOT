#!/usr/bin/env python3
"""Preserve invalid initial B0 artifacts, repair only four registered points."""
import json
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.diag_attrib_v2_common import *


def repair():
    require_prereg()
    start,cpu=time.perf_counter(),time.process_time()
    retained=RUN/'retained_failures/b0_scale_clamp_1'
    if retained.exists():
        raise FileExistsError('do not overwrite retained first failure')
    retained.mkdir(parents=True)
    shutil.copytree(PUBLIC,retained/'initial_public_report')
    shutil.copy2(RUN/'ORACLE_predictions.json',retained/'ORACLE_predictions_initial.json')
    shutil.copy2(RUN/'completion.json',retained/'completion_initial_revoked.json')
    record=json.loads((RUN/'ORACLE_predictions.json').read_text())
    metadata,scores,groups,historical=load_bank()
    lines=candidate_lines(metadata)
    fixed=[]
    for scale in (1.25,1.5,2.,3.):
        method='B0_SCALE_'+str(scale).replace('.','p')
        old=EVAL/method
        old_seal=verify_seal(method)
        if old_seal['prediction_rows'] != 212541:
            raise ValueError('not the identified initial B0 clamp failure')
        saved=retained/'evaluation'/method
        saved.parent.mkdir(exist_ok=True)
        old.rename(saved)
        jobs=RUN/'official_jobs'/method
        target=retained/'official_jobs'/method
        target.parent.mkdir(exist_ok=True)
        jobs.rename(target)
        budget=scale_budget(scale,212541,len(groups))
        mask,detail=top_budget(scores['FROZEN'],budget)
        seal=write_selection(method,mask,metadata,lines,protocol='B0_GT_INFORMED_SCALE_CURVE_NOT_A_METHOD')
        if seal['prediction_rows']!=budget:
            raise ValueError('corrected B0 count differs from preregistration')
        new=dict(method=method,registered_budget=budget,**detail)
        record['B0_boundaries']=[new if r['method']==method else r for r in record['B0_boundaries']]
        record['prediction_seals'][method]=file_sha(EVAL/method/'predictions_seal.json')
        fixed.append(dict(method=method,invalid_initial_rows=212541,correct_registered_rows=budget,
                          retained_prediction_and_evaluation=str(saved)))
    recovery=dict(status='FOUR_B0_COUNTS_REPAIRED_AWAIT_OFFICIAL_EVALUATION',
        first_actionable_root_cause='round_budget(scale,GT_count) capped an absolute scale budget to GT_count instead of available candidates',
        definition_changed=False,preregistered_scale_points_unchanged=True,affected=fixed,
        unaffected='all30 matrix cells, all8 B1-B4 baselines, all7 historical sensitivity and all bootstrap/verdict comparisons',
        invalid_initial_report_retained=str(retained/'initial_public_report'),
        invalid_initial_report_was_not_pushed=True,CPU_wall_seconds=time.perf_counter()-start,
        CPU_process_seconds=time.process_time()-cpu,GT_free_methods_changed=False,GPU_calls=0,optimizer_steps=0)
    record['B0_recovery']=recovery
    record['CPU_wall_seconds']+=recovery['CPU_wall_seconds']
    record['CPU_process_seconds']+=recovery['CPU_process_seconds']
    atomic_json(RUN/'ORACLE_predictions.json',record)
    atomic_json(RUN/'B0_recovery.json',recovery)
    atomic_json(RUN/'completion.json',dict(status='INITIAL_REPORT_REVOKED_B0_FOUR_POINTS_PENDING',recovery=str(RUN/'B0_recovery.json')))
    print(json.dumps(recovery,indent=2),flush=True)


if __name__=='__main__':
    repair()
