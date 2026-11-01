"""At most two memory-admitted shared GPUs; a single blocking parent owns waits."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decompose_common import *
from flex_uda.data.manifests import read_jsonl
from scripts.launch_train2val_prep import child, TEACHER_PY
from scripts.launch_uda_workers import audit_gpus


def launch():
    require_plan();started=time.perf_counter();plan=json.loads((RUN/'probe/plan.json').read_text())
    path=RUN/'probe/shards.json'
    if not path.exists():
        counts={}
        for f in read_jsonl(plan['manifest_path']):
            key=f['target'],f['video_id'];counts[key]=counts.get(key,0)+len(f['queries'])*len(f['candidates'])
        shards=[[],[]];loads=[0,0]
        for key,n in sorted(counts.items(),key=lambda item:(-item[1],item[0])):
            j=min(range(2),key=lambda j:(loads[j],j));shards[j].append(list(key));loads[j]+=n
        atomic_json(path,dict(shards=shards,logical_pair_loads=loads,whole_video_shards=True))
    attempt=RUN/'probe/launchers'/str(time.time_ns());attempt.mkdir(parents=True)
    while True:
        try:
            selected,gpus,apps=audit_gpus(2,minimum_free_mib=24576);break
        except RuntimeError as e:
            print(json.dumps(dict(event='DECOMPOSE_RESOURCE_WAIT',reason=str(e))),flush=True);time.sleep(20)
    atomic_json(attempt/'admission.json',dict(selected=selected,all_gpus=gpus,processes=apps,
                sharing_authorized=True,max_physical_GPUs=2,allocator_budget_GiB=20,admission_headroom_GiB=4))
    def job(shard,gpu):
        records=[]
        for smoke in (True,False):
            done=RUN/'probe/workers'/str(shard)/('smoke_receipt.json' if smoke else 'receipt.json')
            if done.exists():
                r=json.loads(done.read_text())
                if not smoke and file_sha(r['outputs_path'])!=r['outputs_sha256']:
                    raise ValueError('Changed completed shard')
                continue
            command=[TEACHER_PY,ROOT/'scripts/decompose_teacher.py','--shard',str(shard)]
            if smoke:command+=['--smoke']
            tag=f'shard{shard}_'+('smoke' if smoke else 'full')
            r=child(command,gpu,attempt/(tag+'.log'),tag);records.append(r)
            atomic_json(attempt/f'worker_{shard}.json',records)
            if r['exit_code']:
                raise RuntimeError('Retain completed calls and inspect '+r['log'])
        return records
    records=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(job,i,g) for i,g in enumerate(selected)]):records+=f.result()
    receipts=[json.loads((RUN/'probe/workers'/str(i)/'receipt.json').read_text()) for i in range(2)]
    result=dict(status='ALL_PAIRED_PROBES_SEALED_BEFORE_GT',workers=receipts,worker_attempts=records,
                wall_seconds=time.perf_counter()-started,GPU_process_hours=sum(r['worker_seconds'] for r in records)/3600,
                max_physical_GPUs=2,GT_open_attempts=0,teacher_updates=0,student_updates=0)
    atomic_json(attempt/'completion.json',result);atomic_json(RUN/'probe/seal.json',result)
    print(json.dumps(dict(event='DECOMPOSE_ALL_TEACHER_PROBES_SEALED',GPU_process_hours=result['GPU_process_hours'],wall_seconds=result['wall_seconds'])),flush=True)


if __name__=='__main__':launch()
