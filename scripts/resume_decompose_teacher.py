"""Move ONLY this task's unfinished cached shard to an admitted free GPU."""
from collections import Counter,defaultdict
import json
import os
from pathlib import Path
import signal
import sqlite3
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decompose_common import *
from scripts.decompose_teacher import digest
from scripts.launch_train2val_prep import child,TEACHER_PY
from scripts.launch_uda_workers import audit_gpus


def main():
    require_plan();old_pid=1407433
    old_launch=RUN/'probe/launchers/1793551889976369066'
    completed=json.loads((RUN/'probe/workers/0/receipt.json').read_text())
    if file_sha(completed['outputs_path'])!=completed['outputs_sha256']:raise ValueError('Preserve completed shard0')
    if (RUN/'probe/workers/1/receipt.json').exists():raise ValueError('Shard already completed; do not interrupt')
    cmd=Path(f'/proc/{old_pid}/cmdline').read_bytes().split(b'\0')
    arguments=[x.decode() for x in cmd if x]
    if arguments[1:]!=[str(ROOT/'scripts/decompose_teacher.py'),'--shard','1']:
        raise PermissionError('Exact owned worker PID/command verification failed; signal nobody')
    os.environ['UDA_GPU_IDS']='2'
    chosen,gpus,apps=audit_gpus(1,minimum_free_mib=24576)
    attempt=RUN/'probe/launchers'/str(time.time_ns());attempt.mkdir(parents=True)
    atomic_json(attempt/'admission.json',dict(selected=chosen,all_gpus=gpus,processes=apps,
        sharing_authorized=True,max_physical_GPUs=2,allocator_budget_GiB=20,
        resume_reason='GPU7 now3 processes; GPU2 idle. Only exact owned cached worker is interrupted.',
        old_pid=old_pid,verified_owned_command=arguments,no_other_process_signalled=True))
    atomic_json(attempt/'resume_intent.json',dict(old_pid=old_pid,old_gpu=7,new_gpu=2,
        preserve_completed_shard0=True,preserve_SQLite=True,no_model_or_input_or_batch_parameter_change=True))
    # SIGINT runs the existing worker's finally/access audit. SQLite commits are
    # per batch; the last uncommitted batch is rolled back, never fabricated.
    os.kill(old_pid,signal.SIGINT)
    print(json.dumps(dict(event='DECOMPOSE_OWN_SHARD_SIGINT_FOR_RESOURCE_RESUME',pid=old_pid,old_gpu=7,new_gpu=2)),flush=True)
    while True:
        r=json.loads((old_launch/'worker_1.json').read_text())
        original=next((x for x in r if x['tag']=='shard1_full' and x['pid']==old_pid),None)
        if original is not None:break
        time.sleep(1)
    access=json.loads((RUN/'probe/workers/1/access.json').read_text())
    if access['GT_open_attempts']:raise ValueError('Interrupted teacher GT audit failed')
    cache=RUN/'probe/workers/1/calls.sqlite'
    db=sqlite3.connect('file:'+str(cache)+'?mode=ro',uri=True)
    if db.execute('PRAGMA quick_check').fetchone()[0]!='ok':raise ValueError('Cached calls not recoverable')
    batches=defaultdict(dict);counts=Counter();peak=Counter();cache_rows=0
    code_sha=file_sha(ROOT/'flex_uda/teacher/decompose_inputs.py')
    for key,payload in db.execute('SELECT key,payload FROM calls'):
        row=json.loads(payload);cache_rows+=1;method=row['method']
        expected=digest(dict(revision=REVISION,weight=row['frozen_weight_sha256'],code=code_sha,method=method,
            target=row['target'],video=row['video_id'],frame=row['frame_id'],track=row['track_id'],query=row['query_id'],
            image=row['image_sha256'],visual=row['visual_input_sha256'],prompt=row['prompt']))
        if expected!=key or row['model_revision']!=REVISION or not all(np.isfinite(row[k]) for k in ('yes_logit','no_logit','yes_probability')):
            raise ValueError('Changed cached request/model/input or invalid logits')
        counts[method]+=1;batches[method][row['batch_id']]=row['batch_forward_seconds']
        peak[method]=max(peak[method],row['allocated_peak_bytes'])
    db.close()
    smoke=json.loads((RUN/'probe/workers/1/smoke_receipt.json').read_text())
    budgets={}
    for method in ('CROP','CLIP','FRAME_BOX'):
        prior=smoke['budgets'][method]
        budgets[method]=dict(new_logical_calls=counts[method]-prior.get('new_logical_calls',0),
            forward_batches=len(batches[method])-prior.get('forward_batches',0),
            forward_seconds=sum(batches[method].values())-prior.get('forward_seconds',0),
            peak_allocated_bytes=peak[method],timing_status='COMMITTED_BATCH_FORWARD_SECONDS_LOWER_BOUND')
    partial=dict(status='OWN_RESOURCE_RESCHEDULE_PARTIAL_NOT_A_COMPLETE_PREDICTION',budgets=budgets,
        worker_seconds=original['worker_seconds'],GT_open_attempts=access['GT_open_attempts'],
        cached_rows_verified=cache_rows,cache_request_keys_verified=True,cache_path=str(cache),
        possible_discarded_inflight_batches_upper_bound=1,possible_discarded_inflight_judgments_upper_bound=8,
        unknown_inflight_forward_seconds_upper_bound=max(0.,original['worker_seconds']-sum(b['forward_seconds'] for b in budgets.values())),
        original_worker_record=original,preserved_completed_calls=True,parameters_not_updated=True,
        note='Full GPU-process elapsed is measured exactly; per-view committed forward time is a lower bound, not a fabricated exact in-flight time.')
    partial_path=attempt/'interrupted_receipt.json';atomic_json(partial_path,partial)
    atomic_json(RUN/'probe/resume_registry.json',dict(receipt_overrides={original['log']:str(partial_path)},
        partial_attempts=[str(partial_path)],migration_gpu_from=7,migration_gpu_to=2,
        original_backend_unchanged=True,original_crop_unchanged=True,cache_rows_verified=cache_rows))
    print(json.dumps(dict(event='DECOMPOSE_CACHE_RESUME_CHECK_PASS',committed_rows=cache_rows,original_gpu=7,new_gpu=2,
        exact_request_keys=True,model_and_input_unchanged=True)),flush=True)
    r=child([TEACHER_PY,ROOT/'scripts/decompose_teacher.py','--shard','1'],chosen[0],attempt/'shard1_full.log','shard1_full')
    atomic_json(attempt/'worker_1.json',[r])
    if r['exit_code']:raise RuntimeError('Resume failed; preserve cache and inspect '+r['log'])
    receipts=[json.loads((RUN/'probe/workers'/str(i)/'receipt.json').read_text()) for i in range(2)]
    records=[]
    for p in (RUN/'probe/launchers').glob('*/worker_*.json'):records+=json.loads(p.read_text())
    first=min(int(p.parent.name) for p in (RUN/'probe/launchers').glob('*/admission.json'))
    result=dict(status='ALL_PAIRED_PROBES_SEALED_BEFORE_GT',workers=receipts,worker_attempts=records,
        wall_seconds=(time.time_ns()-first)/1e9,wall_clock_basis='elapsed since first original admission attempt timestamp',
        GPU_process_hours=sum(x['worker_seconds'] for x in records)/3600,max_physical_GPUs=2,
        GT_open_attempts=0,teacher_updates=0,student_updates=0,resource_resume_registry=str(RUN/'probe/resume_registry.json'))
    atomic_json(attempt/'completion.json',result);atomic_json(RUN/'probe/seal.json',result)
    print(json.dumps(dict(event='DECOMPOSE_ALL_TEACHER_PROBES_SEALED_AFTER_RESUME',GPU_process_hours=result['GPU_process_hours'],
        wall_seconds=result['wall_seconds'],cached_completed_calls_not_repeated=True)),flush=True)


if __name__=='__main__':main()
