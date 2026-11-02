"""Blocking source-head pipeline, confined to the two already registered GPUs.

Waits/checks live inside this command, never agent-level ps/tail polling. Head
training may share the feature export's two physical GPUs if admission permits;
it cannot allocate a third physical GPU. All raw-GT work is a separate process.
"""
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
from queue import Queue, Empty
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flex_uda.presence.runtime import RUN, IN_DOMAIN
from flex_tta.runtime.pf_receipts import atomic_json


def main():
    p = ArgumentParser(description=__doc__)
    p.add_argument('--feature-admission', type=Path, required=True)
    p.add_argument('--targets', nargs='+', choices=list(IN_DOMAIN), default=list(IN_DOMAIN))
    a = p.parse_args()
    assigned = json.loads(a.feature_admission.read_text())['selected']
    if len(assigned) > 2: raise ValueError('At most two physical GPUs including feature exports')
    folder = RUN / 'head_pipeline' / str(time.time_ns()); folder.mkdir(parents=True)
    parent_pids = []
    for proc in Path('/proc').glob('[0-9]*/cmdline'):
        try: command = proc.read_bytes().replace(b'\0', b' ')
        except (FileNotFoundError, PermissionError): continue
        exporter = any(name in command for name in (b'presence_launch.py --phase A-export', b'presence_rescue_queue.py',
                                                    b'presence_priority.py', b'presence_export_target.py'))
        if exporter and b'python' in command and b'/bin/bash' not in command:
            parent_pids.append(int(proc.parent.name))
    atomic_json(folder / 'plan.json', dict(registered_physical_GPUs=assigned, maximum_physical_GPUs=2,
        requested_targets=a.targets,
        feature_parent_pids=parent_pids, head_training_memory_budget_mib=4096,
        admission_required_free_mib=8192, shared_with_own_frozen_export=True, other_processes_stopped=0))
    def parents_alive():
        return any(Path('/proc', str(pid)).exists() for pid in parent_pids)
    def child(command, env, log):
        with log.open('a') as handle:
            process = subprocess.Popen([sys.executable, *command], cwd=ROOT, env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            print(json.dumps(dict(event='HEAD_PIPELINE_CHILD', command=command, pid=process.pid, log=str(log))), flush=True)
            for line in process.stdout:
                handle.write(line); handle.flush()
                if line.startswith('{'): print(line, end='', flush=True)
            code = process.wait()
        if code: raise RuntimeError('Pipeline child failed: ' + str(log))
    queue = Queue()
    for target in a.targets: queue.put(target)
    def slot(gpu):
        result = []
        while True:
            try: target = queue.get_nowait()
            except Empty: return result
            model = IN_DOMAIN[target]
            ready = [RUN / 'features/source_train' / target / model / 'export_receipt.json',
                     RUN / 'features' / target / model / 'export_receipt.json']
            while not all(path.exists() for path in ready):
                if not parents_alive():
                    raise RuntimeError('Feature exporter ended with unfinished source/domain bank: ' + target)
                time.sleep(20)
            from scripts.select_gpu import query
            while True:
                eligible = [g for g in query() if g['uuid'] in {x['uuid'] for x in assigned} and g['free_mib'] >= 8192]
                if eligible:
                    current = min(eligible, key=lambda g: (g['compute_processes'], -g['free_mib'], g['index']))
                    break
                time.sleep(20)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=current['uuid'], PRESENCE_PHYSICAL_GPU=str(current['index']),
                OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false',
                HF_HUB_OFFLINE='1', PYTHONPATH=f'{ROOT}/compat:{ROOT}:/data2/user/TTA-RMOT-assets/flexhook/pydeps')
            atomic_json(folder / (target+'_admission.json'), dict(current=current, shared_with_export=True,
                allocator_limit_mib=4096, reserved_other_jobs=False, physical_GPU_set=[g['physical_gpu'] for g in assigned]))
            child([str(ROOT / 'scripts/presence_prepare_validation.py'), '--target', target], env, folder / (target+'_validation.log'))
            head = RUN / 'heads' / model
            if not (head / 'training_receipt.json').exists():
                command = [str(ROOT / 'scripts/presence_train.py'), '--target', target]
                if (head / 'checkpoint_latest.pt').exists(): command.append('--resume')
                child(command, env, folder / (target+'_training.log'))
            child([str(ROOT / 'scripts/presence_eval_offline.py'), '--target', target, '--model', model, '--phase', 'decode'], env, folder / (target+'_decode.log'))
            child([str(ROOT / 'scripts/presence_eval_offline.py'), '--target', target, '--model', model, '--phase', 'evaluate'], env, folder / (target+'_evaluate.log'))
            result.append(dict(target=target, model=model, status='SOURCE_HEAD_AND_SAME_DOMAIN_EVAL_COMPLETE'))
    with ThreadPoolExecutor(max_workers=len(assigned)) as pool:
        results = [r for rows in pool.map(slot, assigned) for r in rows]
    atomic_json(folder / 'completion.json', dict(status='REQUESTED_SOURCE_HEADS_AND_SAME_DOMAIN_EVALUATIONS_COMPLETE',
        requested_targets=a.targets, results=results))


if __name__ == '__main__': main()
