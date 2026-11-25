"""Diagnose two failed primary fits; complete their planned ordinary heads.

No GT, new questions, changed fit threshold, or edits to the running42files.
Only the original SAME_ARCH_KD128 controls run on the two freed GPUs.
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

OUT = PARENT / 'fit_failure_diagnosis_20261126'
OWN = OUT / 'registration.json'


def prepare():
    parent = registration()
    cases = [c for c in parent['cases'] if c['pair'] in ('C2', 'C5') and c['evaluated_video'] == '0019']
    assert len(cases) == 2 and not OWN.exists()
    records = []
    for case in cases:
        folder = PARENT / 'units' / case['pair'] / '0019'
        assignment = json.loads((folder / 'assignment.json').read_text())
        assert assignment['GPU'] in (1, 3)
        assert not (folder / 'generation_complete.json').exists()
        assert (folder / 'primary/anchor_projection.pt').exists()
        records.append(dict(case=case, GPU=assignment['GPU'], original_failed_pid=assignment['pid']))
    atomic_json(OWN, dict(status='REGISTERED_GT_FREE_FAILED_FIT_DIAGNOSIS_AND_PLANNED_CONTROL',
        cases=records, source_sha256=file_sha(__file__), parent_registration_sha256=file_sha(PLAN),
        original42sources=parent['source_sha256'], teacher_fit_gate=.005,
        diagnosis='Exact input conflicts and input-conditioned Jensen floor; original bootstrap/closed/failed-projection logits and positive-weight teacher KL. No neural capacity theorem from a zero conflict floor.',
        planned_control='Existing SAME_ARCH_KD128, exact original factory/phase order/Adam.0001/min6000/max16000/seed0. Save original expected checkpoint for later reuse; do not replace primary by this control.',
        healthy_GPU0_and2_workers_preserved=True, no_new_teacher_GT_predictions_or_seeds=True,
        goal_status='ACTIVE'))
    print('REGISTERED_TWO_FAILED_FITS_WITHOUT_CHANGING_GATE_OR_MAIN_SOURCES', flush=True)


def floor_report(x, y, weights):
    canonical = np.ascontiguousarray(x).copy()
    canonical[canonical == 0] = 0.  # +0/-0 are numerically identical inputs.
    keys = canonical.view(np.dtype((np.void, canonical.shape[1] * canonical.dtype.itemsize))).ravel()
    unique, group = np.unique(keys, return_inverse=True)
    n = len(unique); del unique, keys, canonical
    lo, hi = np.full(n, np.inf), np.full(n, -np.inf)
    np.minimum.at(lo, group, y); np.maximum.at(hi, group, y)
    def entropy(p):
        return -p * np.log(np.clip(p, 1e-15, 1)) - (1-p) * np.log(np.clip(1-p, 1e-15, 1))
    result = {}
    for name, w in weights.items():
        mass = np.bincount(group, weights=w, minlength=n)
        mean = np.bincount(group, weights=w*y, minlength=n) / mass
        result[name] = float(np.sum(w * (entropy(mean[group]) - entropy(y))) / w.sum())
    return dict(rows=len(x), unique_exact_inputs=n, conflicting_groups=int(((hi-lo) > 1e-7).sum()),
        maximum_same_input_target_span=float((hi-lo).max()), irreducible_individual_teacher_KL=result,
        zero_floor_does_not_prove_neural_capacity_or_optimization=True)


def measure(model, x, y, weights):
    import torch
    import flex_uda.active.evidence_visual_learning as core
    model.eval(); raw, values, jacobian = [], [], []
    with torch.no_grad():
        for begin in range(0, len(x), 512):
            batch = torch.as_tensor(x[begin:begin+512])
            z = model.output(model.representation(batch)).flatten()
            raw.append(z.numpy()); values.append(core.bounded(z).numpy())
            jacobian.append(model.last_layer_jacobian_norm(batch).numpy())
    raw, z, jacobian = np.concatenate(raw), np.concatenate(values).astype('float64'), np.concatenate(jacobian)
    h = -y*np.log(np.clip(y, 1e-15, 1)) - (1-y)*np.log(np.clip(1-y, 1e-15, 1))
    loss = np.logaddexp(0, z) - y*z - h
    return dict(teacher_KL={k: float(np.sum(w*loss)/w.sum()) for k,w in weights.items()},
        raw_logit_quantiles=np.quantile(raw, [0,.01,.5,.99,1]).tolist(),
        raw_absolute_over={str(t): int((np.abs(raw)>t).sum()) for t in (8,64,256,1024)},
        last_layer_jacobian_quantiles=np.quantile(jacobian, [0,.01,.5,.99,1]).tolist(),
        known_teacher_class_disagreements=int(((z>0)!=(y>.5)).sum()),
        scope='CPU descriptive teacher fit; not GT semantic accuracy or HOTA.')


def worker(pair):
    from flex_uda.active.deterministic_observation_runtime import configure_before_cuda
    execution = configure_before_cuda()
    import torch
    import flex_uda.active.evidence_visual_learning as core
    from flex_uda.active.fullscope_visual_runtime import Context, teacher_table, fit, safe_content_loss
    from flex_uda.a100.mechanism import generation_guard
    guard = generation_guard(); registration()
    own = json.loads(OWN.read_text()); assert own['source_sha256'] == file_sha(__file__)
    record = next(r for r in own['cases'] if r['case']['pair'] == pair)
    assert os.environ['CUDA_VISIBLE_DEVICES'] == str(record['GPU'])
    torch.set_num_threads(2); torch.cuda.set_device(0); torch.cuda.set_per_process_memory_fraction(.35)
    torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False); torch.backends.cuda.enable_math_sdp(True)
    folder = PARENT / 'units' / pair / '0019'; destination = OUT / pair; destination.mkdir(parents=True, exist_ok=True)
    with (folder / 'unit.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        started = time.perf_counter(); context = Context(record['case'], folder, None, record['GPU'])
        closed = torch.load(folder/'primary/closed.pt', map_location='cpu', weights_only=False)
        failed = torch.load(folder/'primary/anchor_projection.pt', map_location='cpu', weights_only=False)
        draws = closed['draws']; assert len(draws) == len({d['uid'] for d in draws}) == 11937
        assert all(context.primary.history[d['uid']]['probability'] == d['probability'] for d in draws)
        assert sum(r['steps'] for r in failed['fits']) == 16000
        x,y,w = teacher_table(context, draws, 'IPW_KD'); _,_,plain = teacher_table(context, draws, 'SOFT_KD')
        weights = dict(positive_LURE=w, ordinary=plain)
        feasibility = floor_report(x,y,weights)
        measurements = {}
        for stage, saved in [('bootstrap',torch.load(folder/'primary/bootstrap.pt',map_location='cpu',weights_only=False)),('closed6000',closed),('failed16000',failed)]:
            model = core.make(context.mean,context.std,'cpu'); model.load_state_dict(saved['solver'])
            measurements[stage] = measure(model,x,y,weights)
        pre = dict(feasibility=feasibility, primary_measurements=measurements,
            original_GPU_failed_max_teacher_KL=failed['proper_teacher_KL'],
            original_checkpoints_preserved=True, GT_open_attempts=len(guard))
        atomic_json(destination/'input_and_primary_diagnosis.json',pre)
        print(json.dumps(dict(status='FAILED_PRIMARY_INPUT_AND_LOGIT_DIAGNOSED',pair=pair,**pre)),flush=True)
        core.data = lambda unused: context.data()
        core.teacher_table = lambda features, query, used, quotas, auxiliary, ay, mode='SOFT_KD': teacher_table(context,used,mode)
        def logged_fit(model, x, y, w, steps, device, optimizer=None, **kwargs):
            opt, rec = fit(context,model,x,y,w,steps,device,optimizer,**kwargs)
            print(json.dumps(dict(status='PLANNED_SAME_ARCH_CONTROL_FIT_PHASE',pair=pair,**rec)),flush=True)
            return opt,rec
        core.fit = logged_fit; core.content_loss = safe_content_loss
        original = core.make
        core.make = lambda mean,std,device,**kwargs: original(mean,std,device,kind='visual',width=128,outputs=kwargs.get('outputs',1))
        control_folder = folder/'ordinary/SAME_ARCH_KD128'; control_folder.mkdir(parents=True,exist_ok=True)
        try:
            head,receipt = core.fit_control(record['case'],draws,'SOFT_KD',control_folder,'cuda:0')
            receipt.update(mode='SAME_ARCH_KD128',width=128,architecture='visual',ordinary_direct_teacher_BCE=True)
            saved = torch.load(control_folder/'SOFT_KD.pt',map_location='cpu',weights_only=False)
            saved['receipt']=receipt;core.save(control_folder/'SOFT_KD.pt',saved)
            control = dict(adequate=True,receipt=receipt,planned_checkpoint_saved_for_reuse=True)
        except AssertionError as error:
            control = dict(adequate=False,exception=str(error),original_gate_not_weakened=True,
                original_partial_checkpoint_preserved=True,primary_not_replaced=True)
        finally:
            core.make = original
        assert not guard
        result = dict(status='COMPLETE_GT_FREE_INPUT_DIAGNOSIS_AND_PLANNED_CONTROL_ATTEMPT',pair=pair,
            **pre,control=control,wall_seconds=time.perf_counter()-started,execution=execution,
            new_teacher_calls=0,current_study_HOTA_read=False,primary_not_replaced_by_control=True,goal_status='ACTIVE')
        atomic_json(destination/'completion.json',result)
        print(json.dumps(dict(status=result['status'],pair=pair,feasibility=feasibility,control=control)),flush=True)


def drain(proc,path):
    with path.open('a',buffering=1) as stream:
        for line in proc.stdout:stream.write(line);print(line,end='',flush=True)


def main():
    own=json.loads(OWN.read_text());assert own['source_sha256']==file_sha(__file__);registration()
    jobs=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        for row in own['cases']:
            gpu=row['GPU'];pair=row['case']['pair']
            owners=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-compute-apps=pid,process_name','--format=csv,noheader,nounits'],text=True).strip()
            used=int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
            assert not owners and used<512,(gpu,owners,used)
            prior=json.loads((PARENT/'supervisor'/f'GPU{gpu}_teacher_final.json').read_text())
            assert prior['teacher']['backend_unmodified']
            proc=subprocess.Popen([sys.executable,'-u',str(Path(__file__)),'--worker',pair],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'))
            future=pool.submit(drain,proc,OUT/(pair+'.log'));jobs.append((pair,proc,future))
            atomic_json(OUT/(pair+'_launch.json'),dict(pid=proc.pid,GPU=gpu,owners_before=owners,used_MiB_before=used,healthy_GPU0_and2_preserved=True))
            print(json.dumps(dict(status='STARTED_FREED_GPU_PLANNED_CONTROL_AND_FIT_DIAGNOSIS',pair=pair,GPU=gpu,pid=proc.pid)),flush=True)
        failures=[]
        for pair,proc,future in jobs:
            code=proc.wait();future.result()
            if code:failures.append([pair,code])
    assert not failures,failures
    results=[json.loads((OUT/r['case']['pair']/'completion.json').read_text()) for r in own['cases']]
    atomic_json(OUT/'completion.json',dict(status='COMPLETE_TWO_FAILED_FIT_DIAGNOSES_NO_GT',cases=results,original42sources_unchanged=True,goal_status='ACTIVE'))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--worker');args=parser.parse_args()
    if args.prepare:prepare()
    elif args.worker:worker(args.worker)
    else:main()
