"""Explicit numerical recovery of a failed ordinary baseline; old26pins intact.

The exact same seed0 network, teacher targets and BCE objective are retained.
Only a failing ordinary optimizer is restarted with damped Adam (.0001).
The .005 adequacy gate, minimum6000/max16000 per-attempt, original primary,
all completed cases, query budgets and both-count comparisons stay fixed.
This gives ordinary KD extra computation; no equal-compute claim is made.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json, file_sha
from scripts.a100_residual_rsi_primary import OUT, PUBLIC, PLAN, registration

AMEND = OUT/'numerical_recovery_registration.json'
SELF = Path(__file__).resolve()


def prepare():
    assert not AMEND.exists()
    plan = registration()
    diagnosis = json.loads((OUT/'fit_failure_diagnosis/diagnosis.json').read_text())
    end = diagnosis['records'][-1]
    assert end['proper_teacher_excess_CE'] == .04745238236393415
    assert end['numerically_saturated_count']==22
    assert not (OUT/'all66_sealed_before_GT.json').exists()
    preserved = {}
    for pair in ('C1','C2','C4'):
        for relative in ('primary/final.pt','ordinary/UNIFORM_SOFT_KD.pt'):
            name = pair+'/'+relative
            preserved[name] = file_sha(OUT/name)
        if pair!='C4':
            preserved[pair+'/main_completion.json'] = file_sha(OUT/pair/'main_completion.json')
    doc = dict(status='PREREGISTERED_C4_ORDINARY_NUMERICAL_RECOVERY_BEFORE_GT',
        original_registration_sha256=file_sha(PLAN), original26_source_pins_unchanged=True,
        source_sha256={str(SELF.relative_to(ROOT)):file_sha(SELF)},
        preserved_files=preserved, diagnosed_original_final=end,
        root_cause='Reproduced original C4softKL exactly; raw[-140.30,389.18] drives float32 tanh saturation. Exact-input contradictions absent. This is optimization failure, not semantic comparison.',
        correction='Only a failing ordinary fit retries same seed0 and phase schedule with Adam rate.0001 instead of.001. Other ordinary fits first use original rate. Original primary is never refit.',
        unchanged='Same9352D/normalization/model/boundedlogit8/targets/weights/clip10/cyclicbatch512/optimizercontinuity/min6000/max16000/.005 properKL. No targetGT or quality-based checkpoint selection.',
        compute='Original failed attempts and damping retries counted separately; stronger ordinary gets extra optimization, no equal-time claim.',
        all66seal_then_isolated_GT=True, no_new_sampling_or_neural_seed=True,
        fullsix_threecontribution_fresh_cost_goal_complete=False, goal_status='ACTIVE')
    atomic_json(AMEND,doc)
    atomic_json(PUBLIC/'residual_primary_numerical_recovery_20261125.json',doc)
    atomic_json(PUBLIC/'residual_primary_fit_failure_20261125.json',dict(
        status=diagnosis['status'],records=diagnosis['records'],GT_open_attempts=0,
        original_registration_sha256=file_sha(PLAN)))
    print(json.dumps(dict(status=doc['status'],preserved=len(preserved))),flush=True)


def verify():
    registration()
    doc=json.loads(AMEND.read_text())
    for name,sha in doc['source_sha256'].items():assert file_sha(ROOT/name)==sha,name
    for name,sha in doc['preserved_files'].items():assert file_sha(OUT/name)==sha,name
    assert file_sha(PLAN)==doc['original_registration_sha256']
    return doc


def worker():
    from flex_uda.active.deterministic_observation_runtime import configure_before_cuda
    configure_before_cuda()
    import torch
    import flex_uda.active.residual_rsi_learning as learning
    import scripts.a100_residual_rsi_primary as original
    verify()
    ordinary,imputed,fit=learning.fit_ordinary,learning.fit_imputed,learning.fit

    def recover(function,case,*args):
        # Case is still C4; a known exact failure need not be executed again.
        mode = args[1] if function is ordinary else 'IMPUTED_KD'
        known_failed = function is ordinary and mode=='SOFT_KD'
        if not known_failed:
            try:return function(case,*args)
            except AssertionError as error:
                value=error.args[0] if error.args else None
                assert isinstance(value,tuple) and value[:2]==('C4',mode),repr(error)
                atomic_json(OUT/'C4'/(mode+'_original_fit_failure.json'),dict(error=repr(error),GT=0))
        print(json.dumps(dict(status='DAMPED_ORDINARY_NUMERICAL_RETRY',pair='C4',mode=mode,lr=.0001,adequacy=.005)),flush=True)
        def damped(model,x,y,weights,steps,device,optimizer=None):
            if optimizer is None:optimizer=torch.optim.Adam(model.parameters(),lr=.0001)
            return fit(model,x,y,weights,steps,device,optimizer)
        learning.fit=damped
        try:model,receipt=function(case,*args)
        finally:learning.fit=fit
        receipt.update(numerical_recovery=True,Adam_learning_rate=.0001,
            optimizer_rate_matches_primary=False,ordering_initialization_and_loss_match=True,
            original_failed_attempt_preserved=True,recovery_registration_sha256=file_sha(AMEND),
            ordinary_gets_extra_compute_no_equal_time_claim=True)
        folder=Path(args[2] if function is ordinary else args[1])
        path=folder/(mode+'.pt')
        saved=torch.load(path,map_location='cpu',weights_only=False)
        saved['receipt']=receipt;learning.save(path,saved)
        print(json.dumps(dict(status='DAMPED_ORDINARY_ADEQUATE',pair='C4',mode=mode,fit=receipt)),flush=True)
        return model,receipt
    learning.fit_ordinary=lambda case,*args:recover(ordinary,case,*args)
    learning.fit_imputed=lambda case,*args:recover(imputed,case,*args)
    original.worker('C4')
    verify()
    completion=OUT/'C4/main_completion.json'
    done=json.loads(completion.read_text());done['numerical_recovery_registration_sha256']=file_sha(AMEND)
    atomic_json(completion,done)


def main():
    verify()
    if not (OUT/'C4/main_completion.json').exists():
        owners=subprocess.check_output(['nvidia-smi','-i','3','--query-compute-apps=pid,process_name','--format=csv,noheader,nounits'],text=True).strip()
        assert not owners,owners
        proc=subprocess.Popen([sys.executable,'-u',str(SELF),'--worker'],cwd=ROOT,
            stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='3',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'))
        atomic_json(OUT/'numerical_recovery_launch.json',dict(pid=proc.pid,GPU=3,original_GPU2_service_preserved=True))
        with (OUT/'numerical_recovery.log').open('a',buffering=1) as stream:
            for line in proc.stdout:stream.write(line);print(line,end='',flush=True)
        assert proc.wait()==0,'C4 recovery failed; no GT gate'
    verify()
    from scripts.a100_residual_rsi_primary import main as complete_original
    complete_original()
    verify()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--worker',action='store_true');args=p.parse_args()
    if args.prepare:prepare()
    elif args.worker:worker()
    else:main()
