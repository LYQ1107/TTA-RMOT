"""One admitted shared GPU, one blocking parent, all frozen probe variants."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.count_structure_common import RUN,PUBLIC,setup,require_plan
from scripts.launch_uda_workers import audit_gpus
from scripts.launch_train2val_prep import child,TEACHER_PY
from flex_tta.runtime.pf_receipts import atomic_json,file_sha

def run():
    setup();require_plan();plan=json.loads((RUN/'probe_plan.json').read_text());start=time.perf_counter()
    attempt=RUN/'probe_launchers'/str(time.time_ns());attempt.mkdir(parents=True)
    while True:
        try:chosen,gpus,apps=audit_gpus(1,minimum_free_mib=49152 if '8B' in plan['models'] else 24576);break
        except RuntimeError as e:print(json.dumps(dict(event='COUNT_PROBE_RESOURCE_WAIT',reason=str(e))),flush=True);time.sleep(20)
    gpu=chosen[0];atomic_json(attempt/'admission.json',dict(selected=gpu,all_gpus=gpus,processes=apps,maximum_task_GPUs=1,sharing_authorized=True))
    records=[]
    for model in plan['models']:
        for smoke in (True,False):
            done=RUN/'probe'/model/('smoke_receipt.json' if smoke else 'receipt.json')
            if done.exists():
                r=json.loads(done.read_text())
                if not smoke and file_sha(r['outputs_path'])!=r['outputs_sha256']:raise ValueError('changed sealed probe outputs')
                continue
            command=[TEACHER_PY,ROOT/'scripts/count_structure_probe.py','worker','--model',model]
            if smoke:command+=['--smoke']
            r=child(command,gpu,attempt/(model+('_smoke' if smoke else '_full')+'.log'),model+('_smoke' if smoke else '_full'))
            records.append(r);atomic_json(attempt/'workers.json',records)
            if r['exit_code']:raise RuntimeError('probe failed; completed calls retained in SQLite; inspect '+r['log'])
    receipts={m:json.loads((RUN/'probe'/m/'receipt.json').read_text()) for m in plan['models']}
    result=dict(status='ALL_AVAILABLE_MODELS_PROBE_SEALED_BEFORE_GT',models=receipts,worker_attempts=records,
        maximum_physical_GPUs=1,actual_wall_seconds=time.perf_counter()-start,
        GPU_process_hours=sum(r['worker_seconds'] for r in records)/3600,
        skipped_8B=not plan['local8B_available'],no_student_training=True)
    atomic_json(attempt/'completion.json',result);atomic_json(RUN/'probe_seal.json',result);atomic_json(PUBLIC/'probe_seal.json',result)
    print(json.dumps(dict(event='COUNT_PROBE_ALL_SEALED',GPU_process_hours=result['GPU_process_hours'],skipped_8B=result['skipped_8B'])),flush=True)

if __name__=='__main__':run()
