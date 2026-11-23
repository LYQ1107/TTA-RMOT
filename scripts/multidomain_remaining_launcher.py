"""At most two admitted GPUs, blocking workers, recover only unfinished units."""
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
from threading import Event
from queue import Queue,Empty
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.multidomain_runtime import RUN,PUBLIC,require_plan,require_remaining_p0
from flex_tta.runtime.pf_receipts import atomic_json

JOBS=[('SportsMOT','LaMOT-best'),('KITTI-V2','LaMOT-best'),('SportsMOT','V2-best'),
      ('Dance','LaMOT-best'),('SportsMOT','Dance-best')]


def output_folder(target,model):
    return RUN/({'SportsMOT':'sports','KITTI-V2':'kitti','Dance':'dance'}[target])/model


def main():
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=['smoke','full'],required=True)
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--target',choices=['SportsMOT','KITTI-V2','Dance'])
    parser.add_argument('--model',choices=['V2-best','Dance-best','LaMOT-best'])
    args=parser.parse_args();require_plan()
    if args.worker:
        if args.target=='SportsMOT':
            from scripts.multidomain_sports_smoke import main as sports
            sys.argv=[sys.argv[0],'--model',args.model,*(['--full'] if args.phase=='full' else [])]
            sports()
        else:
            from scripts.multidomain_kitti_scores import worker
            worker(args.model,args.phase=='smoke',args.target)
        return
    if args.phase=='full':require_remaining_p0()
    pending=[]
    for target,model in JOBS:
        marker=output_folder(target,model)/('smoke_receipt.json' if args.phase=='smoke' else 'scores_seal.json')
        if marker.exists():
            r=json.loads(marker.read_text())
            if args.phase=='smoke' and not r.get('cost_gate_pass'):
                raise RuntimeError('Existing smoke failed the original 6-GPU-hour gate')
            continue
        pending.append((target,model))
    if not pending:
        print(json.dumps(dict(status='ALREADY_COMPLETE_NO_RERUN',phase=args.phase)),flush=True);return
    from scripts.launch_uda_workers import audit_gpus
    n=min(2,len(pending));selected,all_gpus,processes=audit_gpus(n,minimum_free_mib=20480)
    directory=RUN/'remaining_launchers'/str(time.time_ns());directory.mkdir(parents=True)
    atomic_json(directory/'admission.json',dict(phase=args.phase,selected=selected,all_gpus=all_gpus,
        existing_processes=processes,max_GPUs=2,sharing_authorized=True,no_foreign_process_stopped=True))
    start=time.perf_counter();failed=Event();results=[];queue=Queue()
    for item in pending:queue.put(item)
    def slot(gpu):
        records=[]
        while not failed.is_set():
            try:target,model=queue.get_nowait()
            except Empty:break
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu['uuid'],MULTIDOMAIN_PHYSICAL_GPU=str(gpu['physical_gpu']),
                OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',
                HF_HUB_OFFLINE='1',PYTHONPATH=f'{ROOT}/compat:{ROOT}:/data2/user/TTA-RMOT-assets/flexhook/pydeps')
            log=directory/(target+'_'+model+'_'+args.phase+'.txt');begin=time.perf_counter()
            command=[sys.executable,str(Path(__file__).resolve()),'--worker','--target',target,'--model',model,'--phase',args.phase]
            with log.open('x') as handle:
                child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                    text=True,start_new_session=True)
                print(json.dumps(dict(event='REMAINING_CHILD_STARTED',target=target,model=model,phase=args.phase,
                    pid=child.pid,physical_gpu=gpu['physical_gpu'],stdout=str(log))),flush=True)
                for line in child.stdout:
                    handle.write(line);handle.flush()
                    if line.startswith('{'):
                        try:progress=json.loads(line)
                        except ValueError:continue
                        if progress.get('event') in ('SPORTS_VIDEO_COMPLETE','OFFICIAL_KITTI_VIDEO_COMPLETE'):
                            print(json.dumps(dict(event='REMAINING_VIDEO_COMPLETE',target=target,model=model,
                                video=progress.get('video'),compute_seconds=progress.get('compute_seconds'),
                                full_rows=progress.get('candidate_query_rows',progress.get('full_candidate_query_rows')))),flush=True)
                code=child.wait()
            record=dict(target=target,model=model,phase=args.phase,pid=child.pid,exit_code=code,
                wall_seconds=time.perf_counter()-begin,physical_gpu=gpu['physical_gpu'],stdout=str(log))
            records.append(record);atomic_json(directory/(target+'_'+model+'.json'),record)
            print(json.dumps(dict(event='REMAINING_CHILD_FINISHED',**record)),flush=True)
            if code:
                failed.set();print(log.read_text()[-10000:],flush=True);break
        return records
    with ThreadPoolExecutor(max_workers=n) as pool:
        futures=[pool.submit(slot,gpu) for gpu in selected]
        for future in as_completed(futures):
            results.extend(future.result())
            atomic_json(directory/'launcher_status.json',dict(status='FAILED' if failed.is_set() else 'RUNNING',
                jobs=results,wall_seconds=time.perf_counter()-start))
    result=dict(status='FAILED_PRESERVE_HEALTHY_UNITS' if failed.is_set() else 'COMPLETE',
        phase=args.phase,jobs=results,max_GPUs=2,wall_seconds=time.perf_counter()-start,
        GPU_process_hours=sum(x['wall_seconds'] for x in results)/3600,optimizer_updates=0,teacher_calls=0)
    atomic_json(directory/'launcher_status.json',result)
    atomic_json(PUBLIC/('remaining_'+args.phase+'_costs.json'),result)
    if failed.is_set():raise SystemExit(2)


if __name__=='__main__':main()
