"""ONE fixed source-only LR mitigation; default artifacts remain untouched.

Chosen after all default heads selected epoch1 and then overfit their SOURCE
validation loss. Trial LR1e-5 is fixed before running it; every other given
setting and seed stays identical. Select by source loss BEFORE any trial HOTA.
No target-domain metric selects the source head, no new teacher or TTA runs.
"""
from concurrent.futures import ThreadPoolExecutor
from queue import Queue, Empty
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flex_uda.presence.runtime import RUN, PUBLIC, IN_DOMAIN
from flex_tta.runtime.pf_receipts import atomic_json, file_sha


def main():
    from scripts.select_gpu import query
    registered = json.loads((RUN / 'resources/current.json').read_text())['selected']
    out = RUN / 'source_lr_trial' / str(time.time_ns()); out.mkdir(parents=True)
    plan = dict(status='FIXED_SOURCE_ONLY_ONE_LR_TRIAL', lr=1e-5, default_lr=3e-4,
        architecture_and_other_hyperparameters_unchanged=True, seed=0, max_epochs=20, patience=3,
        reason='all default heads choose epoch1 by own-source loss and deteriorate on later epochs',
        selection='minimum own-source validation BCE+0.5 conditional count CE; tie=default',
        no_target_HOTA_for_selection=True, default_outputs_not_overwritten=True, maximum_physical_GPUs=2)
    atomic_json(PUBLIC / 'source_lr_trial_plan.json', plan)
    queue = Queue()
    for target in IN_DOMAIN: queue.put(target)
    def slot(_):
        result = []
        while True:
            try: target = queue.get_nowait()
            except Empty: return result
            name = IN_DOMAIN[target]; destination = RUN / 'heads_source_lr1e5' / name
            if (destination / 'training_receipt.json').exists(): continue
            while True:
                eligible = [g for g in query() if g['uuid'] in {r['uuid'] for r in registered} and g['free_mib'] >= 8192]
                if eligible:
                    gpu = min(eligible, key=lambda g:(g['compute_processes'],-g['free_mib'],g['index'])); break
                time.sleep(20)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu['uuid'], PRESENCE_PHYSICAL_GPU=str(gpu['index']),
                OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1',
                PYTHONPATH=f'{ROOT}/compat:{ROOT}:/data2/user/TTA-RMOT-assets/flexhook/pydeps')
            command = [sys.executable, str(ROOT / 'scripts/presence_train.py'), '--target', target,
                '--learning-rate', '0.00001', '--output', str(destination)]
            log = out / (target+'.log')
            with log.open('a') as handle:
                child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                print(json.dumps(dict(event='SOURCE_LR_TRIAL_STARTED', target=target, model=name,
                    physical_gpu=gpu['index'], pid=child.pid, log=str(log))), flush=True)
                for line in child.stdout:
                    handle.write(line); handle.flush()
                    if line.startswith('{'): print(line, end='', flush=True)
                code = child.wait()
            if code: raise RuntimeError('Source-only LR trial failed: '+str(log))
            result.append(dict(target=target, model=name, log=str(log)))
    with ThreadPoolExecutor(max_workers=2) as pool: completed = [r for rr in pool.map(slot, registered) for r in rr]
    selection = {}
    for target, name in IN_DOMAIN.items():
        default = RUN / 'heads' / name / 'training_receipt.json'
        trial = RUN / 'heads_source_lr1e5' / name / 'training_receipt.json'
        candidates = [('default',default), ('source_lr1e5',trial)]
        values = [(key,path,json.loads(path.read_text())) for key,path in candidates]
        key, path, receipt = min(values, key=lambda v:(v[2]['best_source_loss'],v[0] != 'default'))
        selection[name] = dict(source_domain=target, selected=key, learning_rate=receipt.get('learning_rate',3e-4),
            checkpoint=receipt['checkpoint'], checkpoint_sha256=receipt['checkpoint_sha256'],
            best_source_loss=receipt['best_source_loss'],
            candidates=[dict(variant=k, loss=r['best_source_loss'], path=str(p), sha256=file_sha(p)) for k,p,r in values])
    atomic_json(PUBLIC / 'source_lr_trial_selection.json', dict(models=selection,
        selection_uses_source_validation_loss_only=True, trial_HOTA_not_computed_at_selection=True))
    atomic_json(out / 'completion.json', dict(status='SOURCE_ONLY_LR_TRIAL_TRAINING_COMPLETE', completed=completed))
    print(json.dumps(dict(status='SOURCE_ONLY_SELECTION_SEALED_BEFORE_TRIAL_HOTA', models=selection)), flush=True)


if __name__ == '__main__': main()
