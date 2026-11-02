"""One blocking coordinator for the authorized fixed residual count-head trial."""
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
from flex_uda.presence.runtime import RUN, PUBLIC, IN_DOMAIN
from flex_tta.runtime.pf_receipts import atomic_json, file_sha
from scripts.diag_attrib_v2_common import write_csv

OUT = RUN / 'redesign_v3'
RECORD = PUBLIC / 'redesign_v3'


def run(command, log, env, **context):
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('a') as handle:
        child = subprocess.Popen([sys.executable, *command], cwd=ROOT, env=env,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        print(json.dumps(dict(event='START', pid=child.pid, log=str(log), command=command, **context)), flush=True)
        for line in child.stdout:
            handle.write(line); handle.flush()
            if line.startswith('{'): print(line, end='', flush=True)
        code = child.wait()
    if code: raise RuntimeError('Retain completed units; failing command exit '+str(code)+': '+str(log))


def environment(gpu=None):
    env = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1',
        HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
        PYTHONPATH=f'{ROOT}/compat:{ROOT}:/data2/user/TTA-RMOT-assets/flexhook/pydeps')
    if gpu: env.update(CUDA_VISIBLE_DEVICES=gpu['uuid'], PRESENCE_PHYSICAL_GPU=str(gpu['physical_gpu']))
    else: env['CUDA_VISIBLE_DEVICES'] = ''
    return env


def train(logs):
    from scripts.launch_uda_workers import audit_gpus
    from scripts.select_gpu import query
    # No earlier project workers remain at entry to this phase; never migrate
    # a running context or borrow a third card when shared memory changes.
    selected, all_gpus, apps = audit_gpus(2, minimum_free_mib=8192)
    admission = dict(selected=selected, all_gpus=all_gpus, processes=apps, maximum_physical_GPUs=2,
        sharing_authorized=True, foreign_processes_stopped=0, phase='redesign_v3_heads_only',
        resource_snapshot_not_reservation=True)
    atomic_json(logs / 'admission.json', admission)
    atomic_json(RECORD / 'admission.json', admission)
    atomic_json(RUN / 'resources/current.json', admission)
    jobs = Queue()
    for target in IN_DOMAIN: jobs.put(target)
    def worker(gpu):
        while True:
            try: target = jobs.get_nowait()
            except Empty: return
            name = IN_DOMAIN[target]
            destination = OUT / 'heads' / name
            if (destination / 'training_receipt.json').exists(): continue
            while next(g for g in query() if g['uuid'] == gpu['uuid'])['free_mib'] < 8192: time.sleep(20)
            command = ['scripts/presence_train.py', '--target', target, '--architecture', 'residual_prefix_v3',
                       '--learning-rate', '.0001', '--output', str(destination)]
            if target == 'SportsMOT':
                command += ['--source-label-root', str(RUN / 'source_labels_by_expression_v2/SportsMOT/LaMOT-best')]
            run(command, logs / ('train_'+target+'.log'), environment(gpu), target=target, physical_gpu=gpu['physical_gpu'])
    with ThreadPoolExecutor(max_workers=2) as pool: list(pool.map(worker, selected))
    selection = {}
    for target, name in IN_DOMAIN.items():
        path = OUT / 'heads' / name / 'training_receipt.json'
        r = json.loads(path.read_text())
        selection[name] = dict(source_domain=target, checkpoint=r['checkpoint'], checkpoint_sha256=r['checkpoint_sha256'],
            best_source_loss=r['best_source_loss'], architecture=r['architecture'], receipt_path=str(path),
            receipt_sha256=file_sha(path), initial_metrics=json.loads((path.parent / 'epoch0_native_prior.json').read_text()))
    atomic_json(RECORD / 'selection.json', dict(models=selection, selection='minimum own-source validation count_NLL+.1_KL',
        source_validation_only=True, new_HOTA_not_computed_at_selection=True,
        prereg_sha256=file_sha(RECORD / 'PREREG.md')))
    print(json.dumps(dict(event='SOURCE_SELECTION_SEALED_BEFORE_HOTA', models=selection)), flush=True)


def evaluate(logs, full=False):
    selection_file = RECORD / 'selection.json'; selection_sha = file_sha(selection_file)
    selection = json.loads(selection_file.read_text())['models']
    jobs = list(IN_DOMAIN.items()) if not full else [(t, m) for t in IN_DOMAIN for m in IN_DOMAIN.values()]
    def cell(job):
        target, name = job
        choice = selection[name]
        if file_sha(choice['checkpoint']) != choice['checkpoint_sha256']: raise ValueError('Selected head changed')
        folder = OUT / 'stage_A' / target / name
        commands = [['scripts/presence_eval_offline.py', '--target', target, '--model', name,
                     '--phase', 'decode', '--head-checkpoint', choice['checkpoint'], '--output-dir', str(folder)]]
        for method in ('PRES_TOPK', 'PRES_TOPK_SMOOTH'):
            if (folder / method / 'offline_summary.json').exists():
                from flex_uda.presence.predictions import verify_predictions
                verify_predictions(folder / method)
                continue
            command = ['scripts/active_eval.py', '--target', target, '--folder', str(folder / method)]
            if target == 'SportsMOT': command.append('--sports418')
            commands.append(command)
        for command in commands:
            run(command, logs / ('eval_'+target+'_'+name+'.log'), environment(), target=target, model=name)
    with ThreadPoolExecutor(max_workers=3) as pool: list(pool.map(cell, jobs))
    if file_sha(selection_file) != selection_sha: raise ValueError('Head selection changed during evaluation')
    collect(full=full)


def collect(full=False):
    from scripts.presence_finish_a import references
    from scripts.pf_diag_common import official_metrics
    from flex_uda.presence.predictions import verify_predictions
    selection = json.loads((RECORD / 'selection.json').read_text())['models']
    rows, gates, artifacts, costs = [], [], [], []
    jobs = list(IN_DOMAIN.items()) if not full else [(t, m) for t in IN_DOMAIN for m in IN_DOMAIN.values()]
    for target, name in jobs:
        native_folder = dict(references(target, name))['NATIVE']
        native = official_metrics(native_folder / 'results')['HOTA']
        folder = OUT / 'stage_A' / target / name
        for method in ('PRES_TOPK', 'PRES_TOPK_SMOOTH'):
            path = folder / method
            seal = verify_predictions(path)
            if seal['head_checkpoint_sha256'] != selection[name]['checkpoint_sha256']: raise ValueError('Wrong head in sealed decode')
            row = json.loads((path / 'offline_summary.json').read_text())
            rows.append(dict(row, native_HOTA=native, delta_HOTA=row['HOTA']-native,
                architecture='residual_prefix_v3', own_domain=IN_DOMAIN[target] == name))
            artifacts.append(dict(target=target, model=name, method=method, path=str(path),
                prediction_seal_sha256=file_sha(path / 'predictions_seal.json'),
                detailed_sha256=file_sha(path / 'results/pedestrian_detailed.csv')))
            if method == 'PRES_TOPK' and IN_DOMAIN[target] == name:
                gates.append(dict(target=target, model=name, native_HOTA=native, HOTA=row['HOTA'],
                    delta_HOTA=row['HOTA']-native, passed=row['HOTA'] >= native-1, path=str(path)))
    for target, name in IN_DOMAIN.items():
        r = json.loads((OUT / 'heads' / name / 'training_receipt.json').read_text())
        costs.append(dict(target=target, model=name, epochs=r['epochs'], source_loss=r['best_source_loss'],
            parameters=r['parameters'], optimizer_updates=r['optimizer_updates'], GPU_hours=r['wall_seconds']/3600,
            wall_seconds=r['wall_seconds'], peak_allocated_bytes=r['peak_allocated_bytes'], physical_gpu=r['physical_gpu']))
    fields = list(dict.fromkeys(k for row in rows for k in row))
    write_csv(RECORD / ('stage_A.csv' if full else 'same_domain.csv'), [{k: r.get(k) for k in fields} for r in rows])
    write_csv(RECORD / 'training_costs.csv', costs)
    failures = [g['target'] for g in gates if not g['passed']]
    gate = dict(status='FAIL_STOP_BEFORE_BC' if failures else 'PASS_SAME_DOMAIN', failed_targets=failures,
        criterion='Every own-source PRES_TOPK HOTA >= complete NATIVE HOTA -1', same_domain=gates,
        selection_path=str(RECORD / 'selection.json'), selection_sha256=file_sha(RECORD / 'selection.json'),
        completed_full_A=full, new_teacher_calls=0, target_LoRA_updates=0,
        historical_gate_and_results_unchanged=True)
    atomic_json(RECORD / 'gate.json', gate)
    atomic_json(RECORD / 'artifacts.json', dict(predictions=artifacts, source_heads=selection))
    print(json.dumps(dict(event='REDESIGN_GATE', **gate)), flush=True)
    return gate


def main():
    p = ArgumentParser(description=__doc__); p.add_argument('--phase', choices=['train', 'evaluate', 'full-A', 'collect'], required=True)
    a = p.parse_args()
    logs = OUT / 'launchers' / str(time.time_ns()); logs.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    if a.phase == 'train': train(logs)
    elif a.phase == 'evaluate': evaluate(logs)
    elif a.phase == 'full-A':
        if json.loads((RECORD / 'gate.json').read_text())['failed_targets']: raise RuntimeError('Same-domain gate not passed')
        evaluate(logs, full=True)
    else: collect()
    atomic_json(logs / 'completion.json', dict(status='COMPLETE', phase=a.phase, wall_seconds=time.perf_counter()-started))


if __name__ == '__main__': main()
