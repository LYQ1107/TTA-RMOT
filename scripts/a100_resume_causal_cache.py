"""Checkpoint handoff for exact frozen-feature memoization, never GT tuning.

The two self-contained stage3 jobs continue under their original supervisor.
This helper replaces only the two causal GPU workers, preserving the existing
CPU MINUS process and forwarding progress into the original blocking pipe.
"""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flex_uda.a100.runtime import OUTPUT
from flex_tta.runtime.pf_receipts import atomic_json

METHODS={0:'SLOW_REPLAY_MATCHED',1:'MINUS_RECALC_ONLY'}


def owned_python(script):
    result=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            if p.stat().st_uid!=os.getuid() or (p/'cwd').resolve()!=ROOT:continue
            args=(p/'cmdline').read_bytes().decode().split('\0')
            if args and 'python' in Path(args[0]).name and any(Path(a).name==script for a in args):result.append((int(p.name),args))
        except (FileNotFoundError,PermissionError,ProcessLookupError):continue
    return result


def launch():
    import torch
    parents=owned_python('a100_supervisor.py');assert len(parents)==1
    parent=parents[0][0]
    found={}
    for pid,args in owned_python('a100_causal.py'):
        method=args[args.index('--method')+1]
        if method in METHODS.values():found[method]=pid
    assert set(found)==set(METHODS.values())
    # Briefly freeze only our two workers to choose a shared valid video
    # checkpoint. An in-flight video is never promoted to a completed one.
    while True:
        for pid in found.values():os.kill(pid,signal.SIGSTOP)
        states={m:torch.load(OUTPUT/'stage2'/m/'stream.pt',map_location='cpu',weights_only=False) for m in found}
        if len({s['next_video'] for s in states.values()})==1:break
        for pid in found.values():os.kill(pid,signal.SIGCONT)
        time.sleep(1)
    next_video=next(iter(states.values()))['next_video']
    amendment=dict(reason='50 LoRA steps took742.102s; the legacy512MiB exact frozen-encoder LRU holds only79 frames at6,773,760 bytes/frame.',
        change='Use8192MiB exact frozen-encoder memoization on both causal GPU arms; stage3 unchanged512MiB.',
        completed_videos_preserved=next_video,learning_rate_steps_batch_seed_questions_and_fixed32_unchanged=True,
        canonical_regression='Existing canonical margin, presence, utilities, raw_k must be bitwise equal when recomputed.',
        GT_has_been_read_this_round=False,source_worker_pids=found,old_supervisor_pid=parent,time_unix=time.time())
    atomic_json(OUTPUT/'stage2/runtime_cache_amendment.json',amendment)
    for pid in found.values():os.kill(pid,signal.SIGKILL)
    for method in found:
        out=OUTPUT/'stage2'/method
        # Original supervisor accounts and journals the deliberate interruption.
        while not (out/'supervisor_receipt.json').exists():time.sleep(.1)
        receipt=json.loads((out/'supervisor_receipt.json').read_text());assert receipt['status']=='FAILED'
        archive=out/'attempts/cache_handoff_20261120';archive.mkdir(parents=True,exist_ok=True)
        for name in ('supervisor_receipt.json','model_load.json','stream.pt','rounds.jsonl'):
            p=out/name
            if p.exists():shutil.copy2(p,archive/name)
        ordered=sorted((out/'videos').iterdir())
        for vi,p in enumerate(ordered):
            # Existing directories form a prefix, including the unfinished unit.
            if vi>=next_video:shutil.move(str(p),str(archive/p.name))
        complete={p.name for p in (out/'videos').iterdir()}
        if (out/'rounds.jsonl').exists():
            rows=[json.loads(line) for line in (out/'rounds.jsonl').read_text().splitlines()]
            (out/'rounds.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows if r['video'] in complete))
    try:destination=os.open(f'/proc/{parent}/fd/1',os.O_WRONLY)
    except OSError:destination=os.open(OUTPUT/'cache_handoff_supervisor.log',os.O_CREAT|os.O_APPEND|os.O_WRONLY,0o600)
    child=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-u',str(Path(__file__).resolve()),'--worker'],
        cwd=ROOT,env=dict(os.environ,PYTHONPATH=str(ROOT)),stdout=destination,stderr=subprocess.STDOUT,start_new_session=True)
    os.close(destination)
    atomic_json(OUTPUT/'cache_handoff_process.json',dict(pid=child.pid,parent_supervisor=parent,next_video=next_video,
        original_blocking_pipe_retained=True))
    print(json.dumps(dict(status='CAUSAL_CACHE_HANDOFF_LAUNCHED',next_video=next_video,pid=child.pid)),flush=True)


def worker():
    from scripts.a100_supervisor import run,FAILED
    def lane(gpu,method):
        out=OUTPUT/'stage2'/method
        run([str(ROOT/'scripts/a100_causal.py'),'--method',method,'--feature-cache-mib','8192'],gpu,out)
        if gpu==0 and not FAILED.is_set():
            run([str(ROOT/'scripts/a100_wallclock.py'),'--scope','full','--repeat','0','--method','UNIFORM_K','--budget','0'],
                gpu,OUTPUT/'stage3/full/repeat_0/UNIFORM_K')
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(lane,g,m) for g,m in METHODS.items()]
            for f in concurrent.futures.as_completed(futures):f.result()
        atomic_json(OUTPUT/'cache_handoff_complete.json',dict(status='COMPLETE',GPU_workers=list(METHODS)))
    except BaseException as e:
        atomic_json(OUTPUT/'cache_handoff_FAILED.json',dict(error=repr(e)));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--worker',action='store_true');a=p.parse_args()
    worker() if a.worker else launch()
