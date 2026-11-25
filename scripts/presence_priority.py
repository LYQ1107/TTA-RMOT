"""Prioritize the two small own-domain banks on the already registered GPU set.

This is one blocking command. Export-file locks coordinate with the original
queue; it never stops that queue or opens a third physical GPU.
"""
from argparse import ArgumentParser
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flex_uda.presence.runtime import RUN
from flex_tta.runtime.pf_receipts import atomic_json


def main():
    p = ArgumentParser(description=__doc__)
    p.add_argument('--feature-admission', type=Path, required=True)
    a = p.parse_args()
    from scripts.select_gpu import query
    registered = json.loads(a.feature_admission.read_text())['selected']
    uuids = {g['uuid'] for g in registered}
    out = RUN / 'priority_export' / str(time.time_ns()); out.mkdir(parents=True)
    receipts = []
    for target, model in [('Dance', 'Dance-best'), ('SportsMOT', 'LaMOT-best')]:
        if (RUN / 'features' / target / model / 'export_receipt.json').exists(): continue
        while True:
            available = [g for g in query() if g['uuid'] in uuids and g['free_mib'] >= 20480]
            if available: break
            time.sleep(20)
        gpu = min(available, key=lambda g: (g['compute_processes'], -g['free_mib'], g['index']))
        atomic_json(out / (target+'_admission.json'), dict(selected=gpu,
            registered_physical_GPUs=[g['physical_gpu'] for g in registered], maximum_physical_GPUs=2,
            shared_with_healthy_export=True, foreign_processes_stopped=0))
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu['uuid'], PRESENCE_PHYSICAL_GPU=str(gpu['index']),
            OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false',
            HF_HUB_OFFLINE='1', PYTHONPATH=f'{ROOT}/compat:{ROOT}:/data2/user/TTA-RMOT-assets/flexhook/pydeps')
        command = [sys.executable, str(ROOT / 'scripts/presence_export_target.py'), '--target', target, '--model', model]
        log = out / (target+'.log'); start = time.perf_counter()
        with log.open('x') as stream:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            print(json.dumps(dict(event='PRIORITY_OWN_DOMAIN_EXPORT', target=target, physical_gpu=gpu['index'], pid=child.pid, log=str(log))), flush=True)
            for line in child.stdout:
                stream.write(line); stream.flush()
                if line.startswith('{'): print(line, end='', flush=True)
            code = child.wait()
        if code: raise RuntimeError('Priority bank failed; preserve source/original queue: '+str(log))
        receipts.append(dict(target=target, model=model, seconds=time.perf_counter()-start, log=str(log)))
    atomic_json(out / 'completion.json', dict(status='PRIORITY_OWN_DOMAIN_BANKS_COMPLETE', results=receipts))


if __name__ == '__main__': main()
