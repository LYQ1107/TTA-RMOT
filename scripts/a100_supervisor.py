"""One blocking supervisor for the four explicitly authorized A100 devices."""
import concurrent.futures
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=Path(os.environ.get('TTA_RMOT_A100_RUN_ROOT',ROOT/'a100_runs/three_stage_20261120')).resolve()
PYTHON=ROOT/'.venv/bin/python'
FAILED=threading.Event()


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.partial')
    tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def run(command,gpu,folder,budget=0):
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/'supervisor_receipt.json').exists():
        r=json.loads((folder/'supervisor_receipt.json').read_text())
        if r['status'] in ('COMPLETE','BUDGET_EXHAUSTED'):
            print(json.dumps(dict(event='SKIP_COMPLETED_JOB',folder=str(folder))),flush=True);return
    if FAILED.is_set():return
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONPATH=str(ROOT))
    begin=time.perf_counter();command=[str(PYTHON),'-u',*command]
    if budget:command+=['--start',str(begin)]
    print(json.dumps(dict(event='JOB_START',gpu=gpu,folder=str(folder),budget=budget)),flush=True)
    attempt=sum(1 for _ in folder.glob('process_*.log'))
    log=(folder/f'process_{attempt:02d}.log').open('w')
    child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
    def drain():
        for line in child.stdout:
            log.write(line);log.flush()
            if line.startswith('{'):
                try:r=json.loads(line)
                except ValueError:continue
                if r.get('event')=='ANYTIME_SNAPSHOT':
                    # The parent can observe the fully written prediction now.
                    # Use this conservative timestamp for hard-budget credit.
                    with (folder/'availability.jsonl').open('a') as available:
                        available.write(json.dumps(dict(index=r['index'],parent_observed_seconds=time.perf_counter()-begin))+'\n')
                if r.get('event') in ('VIDEO_DONE','CAUSAL_ARM_SEALED','WALL_WORKER_DONE','STUDENT_LOAD_DONE'):
                    print(json.dumps(dict(gpu=gpu,**r)),flush=True)
    reader=threading.Thread(target=drain,daemon=True);reader.start()
    expired=False
    try:code=child.wait(timeout=budget if budget else None)
    except subprocess.TimeoutExpired:
        expired=True;child.kill();code=child.wait()
    reader.join();log.close()
    elapsed=time.perf_counter()-begin
    status='BUDGET_EXHAUSTED' if expired else ('COMPLETE' if code==0 else 'FAILED')
    worker_path=folder/'worker_receipt.json'
    if status=='COMPLETE' and worker_path.exists() and json.loads(worker_path.read_text())['status']=='BUDGET_EXHAUSTED':
        status='BUDGET_EXHAUSTED'
    save(folder/'supervisor_receipt.json',dict(status=status,returncode=code,wall_seconds=elapsed,budget_seconds=budget,
        physical_GPU=gpu,process_started=begin,hard_deadline_terminated=expired,log=str(log.name),command=command))
    print(json.dumps(dict(event='JOB_END',gpu=gpu,status=status,wall_seconds=elapsed,folder=str(folder))),flush=True)
    if status=='FAILED':
        FAILED.set();raise RuntimeError('Experiment failed; inspect '+str(log.name))


def main():
    stage1=json.loads((OUT/'stage1/receipt.json').read_text())
    assert stage1['status']=='PASS','Actual minimal loop is a prerequisite'
    for p in (OUT/'stage2/preregistration.json',OUT/'stage3/preregistration.json'):assert p.exists()
    jobs=queue.Queue()
    for repeat in range(3):
        order=list(('MINUS_v8P','YT_PRIOR','UNIFORM_K'))
        order=order[repeat:]+order[:repeat]
        for method in order:jobs.put(('short',repeat,method))
    for method in ('MINUS_v8P','YT_PRIOR','UNIFORM_K'):jobs.put(('full',0,method))
    def benchmark_lane(gpu):
        while not FAILED.is_set():
            try:scope,repeat,method=jobs.get_nowait()
            except queue.Empty:return
            run([str(ROOT/'scripts/a100_wallclock.py'),'--method',method,'--scope',scope,'--repeat',str(repeat),
                 '--budget','600' if scope=='short' else '0'],gpu,OUT/'stage3'/scope/f'repeat_{repeat}'/method,
                 budget=600 if scope=='short' else 0)
            jobs.task_done()
    def causal_lane(gpu,method):
        run([str(ROOT/'scripts/a100_causal.py'),'--method',method],gpu,OUT/'stage2'/method)
        benchmark_lane(gpu)
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        futures=[pool.submit(causal_lane,0,'SLOW_REPLAY_MATCHED'),pool.submit(causal_lane,1,'MINUS_RECALC_ONLY'),
            pool.submit(benchmark_lane,2),pool.submit(benchmark_lane,3),
            pool.submit(run,[str(ROOT/'scripts/a100_causal.py'),'--method','MINUS'],'',OUT/'stage2/MINUS')]
        for future in concurrent.futures.as_completed(futures):future.result()
    if FAILED.is_set():raise RuntimeError('A child failed')
    save(OUT/'compute_complete.json',dict(status='ALL_COMPUTE_JOBS_FINISHED',physical_GPUs=[0,1,2,3]))
    print(json.dumps(dict(event='COMPUTE_COMPLETE')),flush=True)


if __name__=='__main__':main()
