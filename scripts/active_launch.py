"""One blocking, at-most-two-GPU active queue; gate/cost plan checked first."""
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
from queue import Queue, Empty
from threading import Event
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flex_uda.presence.runtime import RUN, PUBLIC
from flex_tta.runtime.pf_receipts import atomic_json, file_sha
from scripts.active_plan import jobs as requested_jobs


def main():
    p = ArgumentParser(description=__doc__)
    p.add_argument('--phase', choices=['pilot', 'registered'], required=True)
    p.add_argument('--jobs-json', type=Path)
    p.add_argument('--pilot-videos',type=int,default=1)
    p.add_argument('--pilot-pairs',nargs='+',choices=['C1','C3'],default=['C1','C3'])
    a = p.parse_args()
    from flex_uda.active.execution import authorize_active
    execution,publication=authorize_active()
    if execution.manifest and execution.manifest.get('pilot_only'):
        if a.phase!='pilot' or a.pilot_videos!=1:
            raise PermissionError('Only first-video small pilots are authorized for this construction')
    if a.phase == 'pilot':
        selected_jobs = [dict(pair=pair, method='FULL', rho=.3, pilot_videos=a.pilot_videos) for pair in a.pilot_pairs]
    else:
        if not a.jobs_json: p.error('Explicit documented job list required after real-pilot cost estimate')
        plan = json.loads(a.jobs_json.read_text()); cost_path = execution.public / 'BC_cost_estimate.json'
        if plan['cost_estimate_sha256'] != file_sha(cost_path): raise ValueError('Changed cost estimate')
        cost = json.loads(cost_path.read_text())
        if cost['total_GPU_hours_estimate'] > 40 and plan['tier'] != 'MAIN_AND_CRITICAL_ONLY':
            raise PermissionError('Over40h: do not expand the full matrix before user direction')
        selected_jobs = plan['jobs']
        fixed = requested_jobs()
        if any(job not in fixed for job in selected_jobs): raise ValueError('Unregistered method or budget')
    def output(job):
        return execution.active / job['pair'] / job['method'] / (
            'original_delta10' if job['method'] == 'YESTRACK_FULL_D10' else 'rho_'+str(job['rho']))
    def finished(job):
        path=output(job) / ('pilot_receipt.json' if a.phase=='pilot' else 'predictions_seal.json')
        if not path.exists():return False
        return a.phase!='pilot' or len(json.loads(path.read_text())['videos'])>=job['pilot_videos']
    selected_jobs = [job for job in selected_jobs if not finished(job)]
    if not selected_jobs:
        print(json.dumps(dict(status='ALREADY_COMPLETE_NO_RERUN', phase=a.phase)), flush=True); return
    for job in selected_jobs:
        if job['method'].startswith('YESTRACK_BUDGET'):
            dependency = execution.active / job['pair'] / 'FULL' / ('rho_'+str(job['rho'])) / 'predictions_seal.json'
            queued = any(j['method'] == 'FULL' and j['pair'] == job['pair'] and j['rho'] == job['rho'] for j in selected_jobs)
            if not dependency.exists() and not queued: raise ValueError('Budget control has no sealed/queued FULL dependency')
    selected_jobs.sort(key=lambda j: j['method'] != 'FULL')
    from scripts.select_gpu import query
    current = query()
    eligible = sorted([g for g in current if g['free_mib'] >= 43008],
        key=lambda g: (g['compute_processes'], -g['free_mib'], g['index']))[:min(2, len(selected_jobs))]
    if not eligible: raise RuntimeError('No shared GPU fits fixed teacher20GiB+student16GiB+headroom; do not evict other processes')
    out = execution.launchers / str(time.time_ns()); out.mkdir(parents=True)
    atomic_json(out / 'admission.json', dict(selected=eligible, all_gpus=current, jobs=selected_jobs,
        stage_A_publication=publication, maximum_physical_GPUs=2,
        teacher_memory_cap_GiB=20, student_memory_cap_GiB=16, admission_required_mib=43008,
        sharing_authorized=True, foreign_processes_stopped=0, resources_not_reserved=True))
    queue = Queue()
    failed = Event()
    for job in selected_jobs: queue.put(job)
    def slot(gpu):
        receipts = []
        while True:
            try: job = queue.get_nowait()
            except Empty: return receipts
            # Budget controls only start once the corresponding FULL run is
            # sealed. This internal wait never starts a duplicate FULL stream.
            if job['method'].startswith('YESTRACK_BUDGET'):
                dependency = execution.active / job['pair'] / 'FULL' / ('rho_'+str(job['rho'])) / 'predictions_seal.json'
                while not dependency.exists():
                    if failed.is_set(): raise RuntimeError('FULL dependency failed; do not invent a matching budget')
                    time.sleep(10)
            # Admission is rechecked inside this blocking coordinator. Keep
            # the registered cards; do not evict other work or borrow a third.
            while next(g for g in query() if g['uuid']==gpu['uuid'])['free_mib']<43008:
                time.sleep(20)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu['uuid'], PRESENCE_PHYSICAL_GPU=str(gpu['index']),
                OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', HF_HUB_OFFLINE='1',
                TOKENIZERS_PARALLELISM='false', PYTHONPATH=f'{ROOT}/compat:{ROOT}:/data2/user/TTA-RMOT-assets/flexhook/pydeps')
            baseline = job['method'].startswith('YESTRACK') or job['method'] == 'UNIFORM_K'
            command = [sys.executable, str(ROOT / ('scripts/active_baseline_run.py' if baseline else 'scripts/active_run.py')),
                '--pair', job['pair'], '--method' if baseline else '--variant', job['method']]
            if job['rho'] is not None: command += ['--rho', str(job['rho'])]
            if job.get('pilot_videos'): command += ['--pilot-videos', str(job['pilot_videos'])]
            name = '_'.join([job['pair'], job['method'], str(job['rho'])])
            start = time.perf_counter(); log = out / (name+'.log')
            with log.open('x') as stream:
                child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                print(json.dumps(dict(event='ACTIVE_JOB_STARTED', job=job, physical_gpu=gpu['index'], pid=child.pid, log=str(log))), flush=True)
                for line in child.stdout:
                    stream.write(line); stream.flush()
                    if line.startswith('{'): print(line, end='', flush=True)
                code = child.wait()
            row = dict(job, seconds=time.perf_counter()-start, exit_code=code, physical_gpu=gpu['index'], log=str(log))
            receipts.append(row); atomic_json(out / (name+'_completion.json'), row)
            if code:
                failed.set()
                raise RuntimeError('Preserve finished videos; first failure in '+str(log))
    with ThreadPoolExecutor(max_workers=len(eligible)) as pool:
        receipts = [r for rr in pool.map(slot, eligible) for r in rr]
    atomic_json(out / 'completion.json', dict(status='REGISTERED_ACTIVE_QUEUE_COMPLETE', phase=a.phase, jobs=receipts))


if __name__ == '__main__': main()
