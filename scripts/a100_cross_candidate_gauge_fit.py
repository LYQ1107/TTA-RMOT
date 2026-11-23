"""Prospective teacher-risk engineering check, no task GT or predictions."""
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from flex_uda.a100.mechanism import generation_guard
from flex_uda.active.cross_candidate_gauge import GaugeFixedCrossCandidateHead
from scripts import a100_cross_candidate_fit_pilot as protocol
from scripts.a100_cross_candidate_teacher_pilot import OUT as TEACHER
from scripts.a100_six_direction import OUT as SIX,PUBLIC

OUT=SIX/'cross_candidate_gauge_fit_20261123'
PLAN=PUBLIC/'cross_candidate_gauge_prospective_plan.json'


def contract():
    generator=torch.Generator().manual_seed(0)
    ua,ub,b,interaction=torch.randn(4,17,generator=generator,dtype=torch.float64)
    zero=torch.zeros_like(b);offset=(ua+ub)/2
    before=torch.log_softmax(torch.stack((b+ua,b+ub,2*b+ua+ub+interaction,zero),-1),-1)
    va,vb,common=ua-offset,ub-offset,b+offset
    after=torch.log_softmax(torch.stack((common+va,common+vb,2*common+va+vb+interaction,zero),-1),-1)
    error=float((before-after).abs().max())
    assert error<1e-12 and torch.allclose(ua-ub,va-vb)
    return dict(classical_gauge_family_equivalence_max_error=error,
        preserved_candidate_difference=True,
        identical_function_requires_intercept_coordinate_change=True,
        not_identical_forward_at_untransformed_old_parameters=True)


def main():
    plan=json.loads(PLAN.read_text())
    for name,sha in plan['source_sha256'].items():assert file_sha(ROOT/name)==sha,name
    guard=generation_guard();torch.set_num_threads(2)
    algebra=contract();protocol.use_verified_local_features()
    data=json.loads((TEACHER/'responses.json').read_text())
    original=json.loads((protocol.OUT/'receipt.json').read_text())
    # Explicit constructor injection in this process only. The pinned original
    # source/optimizer remains unchanged; ordinary constructors are untouched.
    protocol.CrossCandidateHead=GaugeFixedCrossCandidateHead
    started=time.perf_counter();reports={}
    for pair,video in sorted({(c['pair'],c['video']) for c in data['cases']}):
        arrays,target,source,mean,std=protocol.load_video(data,pair,video)
        own={}
        for name in ('CONDITIONAL_IDENTITY_REFERENCE','FULL_LIKELIHOOD_IDENTITY_REFERENCE'):
            model,receipt=protocol.fit(name,arrays,target,64)
            own[name]=receipt
            folder=OUT/pair;folder.mkdir(parents=True,exist_ok=True)
            torch.save(dict(model=model.state_dict(),mean=mean,std=std,receipt=receipt),folder/(name+'.pt'))
            print(json.dumps(dict(pair=pair,arm=name,joint_KL=receipt['joint_KL'],
                exactly_one_conditional_KL=receipt['exactly_one_conditional_KL'],
                fit_gate=receipt['ordinary_gate_passed'])),flush=True)
        reports[pair]=dict(gauge=own,original=original['targets'][pair]['fits'],
            ordinary_original_fit_reused_no_new_baseline_training=True)
    assert not guard
    result=dict(status='COMPLETE_GAUGE_PARAMETERIZATION_TEACHER_FIT_ENGINEERING_ONLY',
        targets=reports,algebra=algebra,source_sha256=plan['source_sha256'],
        wall_seconds=time.perf_counter()-started,neural_seeds=[0],
        no_new_teacher_GPU_GT_or_RMOT_outputs=True,
        no_new_task_quality_or_module_contribution_claim=True,
        selection_scope='Teacher-training-risk engineering after the original pilot; not a preregistered task-quality improvement.')
    atomic_json(OUT/'receipt.json',result)
    atomic_json(PUBLIC/'cross_candidate_gauge_fit_receipt.json',result)
    print(json.dumps(dict(status=result['status'],wall_seconds=result['wall_seconds'])),flush=True)


if __name__=='__main__':main()
