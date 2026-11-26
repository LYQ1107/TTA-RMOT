"""Explicit full-objective numerical recovery, then unfinished registered units.

Both the primary and its same-architecture ordinary control failed on two
large KITTI units. This amendment preserves the original question sequence,
targets, architectures and .005 gate. Deterministic full-batch L-BFGS is a
numerical rescue, not a new RMOT contribution or a compute-matched claim.
The original42running sources and healthyGPU0/2 workers are never edited.
"""
import argparse
import concurrent.futures
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json, file_sha
from scripts.a100_fullscope_visual_six import OUT as PARENT, PLAN, registration

OUT = PARENT / 'full_objective_recovery_20261126'
OWN = OUT / 'registration.json'
MAX_ITERATIONS = 400


def prepare():
    plan = registration()
    diagnosis = PARENT / 'fit_failure_diagnosis_20261126/completion.json'
    evidence = json.loads(diagnosis.read_text())
    assert not OWN.exists() and len(evidence['cases']) == 2
    assert all(not c['control']['adequate'] and c['GT_open_attempts'] == 0 for c in evidence['cases'])
    cases = [c for c in plan['cases'] if c['pair'] in ('C2', 'C5') and c['evaluated_video'] == '0019']
    assert len(cases) == 2
    prior = {c['pair']: c for c in evidence['cases']}
    inputs = {}
    for case in cases:
        unit = PARENT / 'units' / case['pair'] / case['evaluated_video']
        assert prior[case['pair']]['feasibility']['conflicting_groups'] == 0
        assert not (unit / 'generation_complete.json').exists()
        for name in ('primary/closed.pt', 'primary/anchor_projection.pt',
                     'ordinary/SAME_ARCH_KD128/SOFT_KD_partial.pt', 'active_answers.jsonl'):
            p = unit / name
            inputs[str(p)] = file_sha(p)
    atomic_json(OWN, dict(status='REGISTERED_EXPLICIT_NUMERICAL_RECOVERY_NOT_NEW_METHOD',
        parent_registration_sha256=file_sha(PLAN), source_sha256=file_sha(__file__),
        original42sources=plan['source_sha256'], diagnosis_sha256=file_sha(diagnosis),
        original_failed_inputs_sha256=inputs, initial_cases=cases,
        first_failure='Original primary and SAME_ARCH_KD128 both failed original .005 after16000Adamsteps; exact-input floor is zero but does not prove capacity. No old tanh dead zone.',
        hypothesis='Deterministic full-objective gradients and line search may resolve minibatch/conditioning stagnation; failure of this rescue is retained, not a reason to weaken the gate.',
        optimizer=dict(name='torch.optim.LBFGS', lr=1., max_iter_per_call=1,
            max_eval=20, history_size=10, tolerance_grad=1e-9, tolerance_change=1e-12,
            line_search_fn='strong_wolfe', maximum_outer_iterations=MAX_ITERATIONS,
            teacher_loss='Exactly original bounded-logit Bernoulli BCE over all paid rows, normalized by positive weight mass.',
            primary_weights='Mean of original LURE and original ordinary weights, as in original final projection.'),
        trigger='Only a failed original teacher-fit gate; same numerical rescue available to every failing primary or ordinary head in the fixed study.',
        original_Adam_attempts_preserved=True, original_max_Adam_steps=16000,
        extra_fullbatch_optimizer_work_explicitly_additional=True,
        teacher_fit_gate=.005, original_min_steps=6000, seed=0,
        no_GT_teacher_questions_architecture_or_hyperparameter_grid=True,
        healthy_GPU0_and2_preserved=True, original_GPU2_service_preserved=True,
        full148_and4588_and_all_teacher_sessions_before_current_study_GT=True,
        goal_status='ACTIVE'))
    print('REGISTERED_FULL_OBJECTIVE_RESCUE_ORIGINAL_GATE_AND_ALL_FAILED_ATTEMPTS_RETAINED', flush=True)


def validate():
    plan = registration()
    own = json.loads(OWN.read_text())
    assert own['source_sha256'] == file_sha(__file__)
    assert own['parent_registration_sha256'] == file_sha(PLAN)
    return plan, own


def rescue(context, model, x, y, training_weights, gate_weights, folder, device):
    """Optimize an unchanged differentiable objective, with all attempts billed."""
    import torch
    import flex_uda.active.evidence_visual_learning as core
    destination = Path(folder)
    destination.mkdir(parents=True, exist_ok=True)
    checkpoint = destination / 'full_objective_checkpoint.pt'
    optimizer = torch.optim.LBFGS(model.parameters(), lr=1., max_iter=1, max_eval=20,
        history_size=10, tolerance_grad=1e-9, tolerance_change=1e-12, line_search_fn='strong_wolfe')
    tx = core.tensors(x, device)
    ty = torch.as_tensor(y, device=device, dtype=torch.float64)
    tw = torch.as_tensor(training_weights, device=device, dtype=torch.float64)
    assert torch.isfinite(tx).all() and torch.isfinite(ty).all() and torch.isfinite(tw).all()
    assert ((ty >= 0) & (ty <= 1)).all() and (tw > 0).all()
    mass = tw.sum()
    calls, iterations, history, prior_wall = 0, 0, [], 0.
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        assert saved['recovery_registration_sha256'] == file_sha(OWN)
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        calls, iterations, history, prior_wall = saved['closure_calls'], saved['iterations'], saved['history'], saved['wall_seconds']
    model.train()
    started = time.perf_counter()
    def closure():
        nonlocal calls
        calls += 1
        optimizer.zero_grad(set_to_none=True)
        total = torch.zeros((), device=device, dtype=torch.float64)
        for begin in range(0, len(tx), 512):
            z = model(tx[begin:begin+512]).to(torch.float64)
            loss = ((torch.nn.functional.softplus(z) - ty[begin:begin+512]*z)*tw[begin:begin+512]).sum()/mass
            assert torch.isfinite(loss)
            loss.backward()
            total += loss.detach()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        return total
    def errors():
        return {name: core.excess(model, x, y, w, device) for name, w in gate_weights.items()}
    measured = errors()
    before = dict(measured)
    while max(measured.values()) > .005 and iterations < MAX_ITERATIONS:
        optimizer.step(closure)
        iterations += 1
        if iterations % 10 == 0 or iterations == MAX_ITERATIONS:
            torch.cuda.synchronize()
            measured = errors()
            history.append(dict(iteration=iterations, teacher_KL=measured, closure_calls=calls))
            core.save(checkpoint, dict(model=core.state(model), optimizer=optimizer.state_dict(),
                iterations=iterations, closure_calls=calls, history=history,
                wall_seconds=prior_wall+time.perf_counter()-started,
                recovery_registration_sha256=file_sha(OWN)))
            print(json.dumps(dict(status='FULL_OBJECTIVE_NUMERICAL_RESCUE_PROGRESS',
                pair=context.case['pair'], video=context.case['evaluated_video'],
                head=destination.name, **history[-1])), flush=True)
    torch.cuda.synchronize()
    measured = errors()
    receipt = dict(status='ADEQUATE' if max(measured.values()) <= .005 else 'FAILED_ORIGINAL_GATE',
        original_gate=.005, before_teacher_KL=before, final_teacher_KL=measured,
        fullbatch_outer_iterations=iterations, full_dataset_closure_calls=calls,
        equivalent_row_forward_backward_passes=int(calls*len(x)),
        additional_wall_seconds=prior_wall+time.perf_counter()-started,
        original_teacher_targets_architecture_and_draws_unchanged=True,
        full_objective_optimizer_is_not_a_scientific_module=True,
        no_new_GT_teacher_or_seeds=True, recovery_registration_sha256=file_sha(OWN))
    atomic_json(destination/'receipt.json', receipt)
    assert max(measured.values()) <= .005, (context.case['pair'], context.case['evaluated_video'],
        destination.name, 'FULL_OBJECTIVE_RECOVERY_FAILED', measured)
    return receipt


def recovered_primary(original, context, folder, device, registration_sha256):
    import torch
    import flex_uda.active.evidence_visual_learning as core
    from flex_uda.active.fullscope_visual_runtime import teacher_table
    try:
        return original(context, folder, device, registration_sha256)
    except AssertionError as error:
        args = error.args[0] if error.args else ()
        if not isinstance(args, tuple) or len(args) != 4 or args[2] != 'PRIMARY_TEACHER_FIT':
            raise
    folder = Path(folder)
    closed = torch.load(folder/'closed.pt', map_location='cpu', weights_only=False)
    failed = torch.load(folder/'anchor_projection.pt', map_location='cpu', weights_only=False)
    assert closed['registration_sha256'] == registration_sha256
    assert sum(row['steps'] for row in failed['fits']) == 16000
    draws = closed['draws']
    assert len(draws) == len({d['uid'] for d in draws}) == sum(context.quotas.values())
    assert all(context.primary.history[d['uid']]['probability'] == d['probability'] for d in draws)
    s = core.make(context.mean, context.std, device)
    r = core.make(context.mean, context.std, device, outputs=2)
    s.load_state_dict(failed['solver']); r.load_state_dict(closed['observer'])
    x,y,w = teacher_table(context,draws,'IPW_KD')
    _,_,plain = teacher_table(context,draws,'SOFT_KD')
    receipt = rescue(context,s,x,y,(w+plain)/2,dict(positive_LURE=w,ordinary=plain),
        folder/'full_objective_recovery',device)
    weighted,ordinary = core.excess(s,x,y,w,device),core.excess(s,x,y,plain,device)
    result = dict(solver=core.state(s),observer=core.state(r),draws=draws,
        rounds=closed['rounds'],solver_fits=failed['fits'],
        proper_known_teacher_KL=max(weighted,ordinary),weighted_teacher_KL=weighted,
        unweighted_teacher_KL=ordinary,positive_LURE_teacher_risk=True,
        unique_questions=len(draws),broker=context.primary.receipt(complete=True),
        original_teacher_targets_unchanged=True,full_physical_content_support=True,
        no_signed_semantic_targets=True,registration_sha256=registration_sha256,
        explicit_additional_numerical_recovery=receipt,original_failed_Adam_steps=16000,
        do_not_interpret_solver_fits_as_all_actual_optimizer_work=True)
    core.save(folder/'final.pt',result)
    atomic_json(folder/'draws.json',dict(draws=draws))
    return s,r,result


def recovered_control(original, context, case, draws, mode, folder, device):
    import torch
    import flex_uda.active.evidence_visual_learning as core
    from flex_uda.active.fullscope_visual_runtime import teacher_table
    try:
        return original(case,draws,mode,folder,device)
    except AssertionError as error:
        args = error.args[0] if error.args else ()
        if not isinstance(args,tuple) or len(args) != 3 or args[:2] != (case['pair'],mode):
            raise
    folder = Path(folder)
    saved = torch.load(folder/(mode+'_partial.pt'),map_location='cpu',weights_only=False)
    assert saved['completed'] == 3 and sum(row['steps'] for row in saved['receipts']) == 16000
    model = core.make(context.mean,context.std,device,kind='visual' if mode=='SSL_KD' else 'plain',width=288)
    model.load_state_dict(saved['model'])
    x,y,w = teacher_table(context,draws,'HARD_KD' if mode=='HARD_KD' else ('IPW_KD' if mode=='IPW_KD' else 'SOFT_KD'))
    receipt = rescue(context,model,x,y,w,dict(original_teacher_risk=w),folder/'full_objective_recovery',device)
    adequate = dict(mode=mode,total_steps=16000,proper_teacher_excess_CE=receipt['final_teacher_KL']['original_teacher_risk'],
        adequacy_limit=.005,initial_and_phase_order_matches_primary=True,
        Adam_learning_rate=core.LR,parameter_count=sum(p.numel() for p in model.parameters()),
        phase_fits=saved['receipts'],SSL_gets_same_pixels_and_prior_VICReg_pretext=mode=='SSL_KD',
        original_failed_Adam_steps=16000,explicit_additional_numerical_recovery=receipt,
        do_not_interpret_total_steps_as_all_actual_optimizer_work=True)
    core.save(folder/(mode+'.pt'),dict(model=core.state(model),receipt=adequate))
    return model,adequate


def worker(gpu):
    from flex_uda.active.deterministic_observation_runtime import configure_before_cuda
    execution=configure_before_cuda()
    import torch
    import flex_uda.active.evidence_visual_learning as core
    import flex_uda.active.fullscope_visual_runtime as runtime
    from scripts.a100_fullscope_visual_six import run_unit
    plan,own=validate()
    assert os.environ['CUDA_VISIBLE_DEVICES'] == str(gpu)
    torch.set_num_threads(2);torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.35)
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
    original_train,original_control=runtime.train,core.fit_control
    runtime.train=lambda context,folder,device,sha:recovered_primary(original_train,context,folder,device,sha)
    current=[None]
    original_context=runtime.Context
    def context_factory(*args,**kwargs):
        result=original_context(*args,**kwargs);current[0]=result;return result
    runtime.Context=context_factory
    core.fit_control=lambda case,draws,mode,folder,device:recovered_control(original_control,current[0],case,draws,mode,folder,device)
    dispatcher=runtime.TeacherDispatcher(ROOT,PARENT/'teacher_cache/actual_inputs.sqlite')
    first='C2' if gpu==3 else 'C5'
    ordered=sorted(plan['cases'],key=lambda c:(c['pair']!=first or c['evaluated_video']!='0019',-c['logical_budget'],c['pair'],c['evaluated_video']))
    started=time.perf_counter();completed=[]
    try:
        for case in ordered:
            folder=PARENT/'units'/case['pair']/case['evaluated_video'];folder.mkdir(parents=True,exist_ok=True)
            if (folder/'generation_complete.json').exists():continue
            with (folder/'unit.lock').open('a') as lock:
                try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:continue
                if (folder/'generation_complete.json').exists():continue
                atomic_json(folder/f'recovery_assignment_{os.getpid()}.json',dict(GPU=gpu,pid=os.getpid(),source_registration_sha256=file_sha(PLAN),recovery_registration_sha256=file_sha(OWN)))
                run_unit(case,dispatcher,gpu);completed.append([case['pair'],case['evaluated_video']])
    finally:
        sealed=dispatcher.close()
        receipt=dict(teacher=sealed,execution=execution,GPU=gpu,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            recovery_registration_sha256=file_sha(OWN),wall_seconds=time.perf_counter()-started,completed=completed)
        atomic_json(PARENT/'supervisor'/f'GPU{gpu}_teacher_session_{os.getpid()}.json',receipt)
        final=PARENT/'supervisor'/f'GPU{gpu}_teacher_final.json'
        if sealed is not None or not final.exists():atomic_json(final,receipt)
        atomic_json(OUT/f'GPU{gpu}_worker_final.json',receipt)


def drain(proc,path):
    with path.open('a',buffering=1) as stream:
        for line in proc.stdout:stream.write(line);print(line,end='',flush=True)


def main():
    plan,own=validate()
    healthy=[]
    for gpu in (0,2):
        row=json.loads((PARENT/'supervisor'/f'GPU{gpu}_launch.json').read_text())
        healthy.append(row['pid'])
    jobs=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        for gpu in (1,3):
            owners=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-compute-apps=pid,process_name','--format=csv,noheader,nounits'],text=True).strip()
            used=int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
            assert not owners and used<512,(gpu,owners,used)
            proc=subprocess.Popen([sys.executable,'-u',str(Path(__file__)),'--worker',str(gpu)],cwd=ROOT,
                stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'))
            future=pool.submit(drain,proc,OUT/f'GPU{gpu}.log');jobs.append((gpu,proc,future))
            atomic_json(OUT/f'GPU{gpu}_launch.json',dict(GPU=gpu,pid=proc.pid,owners_before=owners,used_MiB_before=used,healthy_original_workers=healthy))
            print(json.dumps(dict(status='STARTED_FULL_OBJECTIVE_RECOVERY_AND_UNFINISHED_REGISTERED_UNITS',GPU=gpu,pid=proc.pid)),flush=True)
        failures=[]
        for gpu,proc,future in jobs:
            code=proc.wait();future.result()
            if code:failures.append([gpu,code])
    assert not failures,failures
    # One blocking completion coordinator, never a second agent-level observer.
    # Original healthy workers keep their locks, paid questions, and T0 sessions.
    for pid in healthy:
        while Path('/proc') .joinpath(str(pid)).exists():time.sleep(2)
    from flex_uda.presence.predictions import verify_predictions
    tasks=[]
    for case in plan['cases']:
        folder=PARENT/'units'/case['pair']/case['evaluated_video']
        done=json.loads((folder/'generation_complete.json').read_text())
        assert done['GT_open_attempts']==0 and done['registration_sha256']==file_sha(PLAN)
        tasks.extend(done['tasks'])
    assert len(tasks)==4588
    for task in tasks:verify_predictions(task['folder'])
    sessions=[]
    for p in sorted((PARENT/'supervisor').glob('GPU*_teacher_session_*.json')):
        row=json.loads(p.read_text())
        if row['teacher'] is not None:
            assert row['teacher']['backend_unmodified']
            sessions.append(dict(path=str(p),sha256=file_sha(p)))
    assert len(sessions)>=4
    for gpu in (0,1,2,3):
        row=json.loads((PARENT/'supervisor'/f'GPU{gpu}_teacher_final.json').read_text())
        assert row['teacher'] is not None and row['teacher']['backend_unmodified']
    atomic_json(PARENT/'all4588_sealed_before_GT.json',dict(tasks=tasks,GT_open_attempts=0,
        teacher_final_weight_seals=4,all_attempt_teacher_sessions=sessions,
        registration_sha256=file_sha(PLAN),explicit_numerical_recovery_registration_sha256=file_sha(OWN)))
    print('ALL4588_FULL_SCOPE_OUTPUTS_AND_ALL_ATTEMPT_TEACHER_SEALS_BEFORE_ISOLATED_GT',flush=True)
    from scripts.a100_fullscope_visual_six import evaluate_and_collect
    evaluate_and_collect(plan,tasks)
    atomic_json(OUT/'completion.json',dict(status='COMPLETE_RECOVERED_WHOLE_SIX_STUDY_ORIGINAL_GATE',
        original42sources_unchanged=True,all_failed_attempts_preserved=True,goal_status='ACTIVE'))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--worker',type=int);args=parser.parse_args()
    if args.prepare:prepare()
    elif args.worker is not None:worker(args.worker)
    else:main()
