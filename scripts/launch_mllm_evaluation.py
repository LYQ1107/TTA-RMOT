#!/usr/bin/env python3
"""Whole-video read-only shards, one blocking parent wait, exact425 merge."""
from collections import Counter
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flex_uda.data.manifests import read_jsonl
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from scripts.launch_uda_workers import audit_gpus


def finish(root,method,rows,shards,checkpoint,started):
    folder=root/method;dest=folder/'results';dest.mkdir(parents=True,exist_ok=True)
    metadata=[];receipts=[];episodes=[];parities=[]
    for shard in shards:
        source=root/'shards'/f'shard{shard["shard"]:02d}'/method
        data=json.loads((source/'run_metadata.json').read_text())
        if data['status']!='SEALED' or data['checkpoint_sha256']!=file_sha(checkpoint) or data['target_GT_open_attempts'] or data['test_time_updates']:
            raise ValueError('invalid shard protocol/teacher/checkpoint')
        metadata.append(data);receipts.extend(data['episodes']);parities.extend(data['parities'])
        for video in shard['videos']:
            path=dest/video;target=(source/'results'/video).resolve()
            if not path.exists():path.symlink_to(target,target_is_directory=True)
            elif path.resolve()!=target:raise ValueError('different existing merged video')
    for receipt in receipts:
        for path,sha in [('prediction_path','prediction_sha256'),('margins_path','margins_sha256')]:
            if file_sha(Path(receipt[path]))!=receipt[sha]:raise ValueError('sealed shard file changed')
        if receipt['stats']['optimizer_steps']:raise ValueError('test-time updates forbidden')
        lines=Path(receipt['prediction_path']).read_text().splitlines();stats=receipt['stats']
        episodes.append(dict(video=receipt['video'],expression=receipt['expression_id'],frames=stats['frames'],
            candidates=stats['candidate_rows'],selected_rows=len(lines),selected_ids=len({s.split(',')[1] for s in lines})))
    expected={(r['video_id'],r['expression_id']) for r in rows}
    if len(receipts)!=425 or {(r['video'],r['expression_id']) for r in receipts}!=expected:raise ValueError('missing/duplicate full425 episodes')
    totals={k:sum(r[k] for r in episodes) for k in ('frames','candidates','selected_rows','selected_ids')}
    if totals['candidates']!=3588445:raise ValueError('incomplete immutable candidate universe')
    audit=dict(status='PASS',method=method,valid_episodes=425,unique_videos=25,missing_episodes=0,duplicated_episodes=0,
        target_gt_opened=False,tracker_geometry_changed=False,tracker_ids_changed=False,episodes=episodes,**totals)
    atomic_json(folder/'full_audit.json',audit)
    first=metadata[0];result={k:v for k,v in first.items() if k not in ('episodes','parities','videos','wall_seconds','cache_stats')}
    result.update(status='SEALED',valid_episodes=425,videos=sorted({r['video_id'] for r in rows}),episodes=receipts,parities=parities,
        wall_seconds=sum(m['wall_seconds'] for m in metadata),worker_seconds=sum(m['wall_seconds'] for m in metadata),
        actual_scheduler_wall_seconds=time.perf_counter()-started,
        peak_allocated_bytes=max(m['peak_allocated_bytes'] for m in metadata),source_shards=shards,
        timing_warning='worker seconds are summed; scheduler wall is separate; shared physical GPUs, no independent energy/utilization claim')
    atomic_json(folder/'run_metadata.json',result);return result


def launch(checkpoint,root,method='MLLM_DISTILL',workers=4):
    checkpoint,root=Path(checkpoint).resolve(),Path(root).resolve();root.mkdir(parents=True,exist_ok=True)
    complete=root/method/'run_metadata.json'
    if complete.exists():
        saved=json.loads(complete.read_text())
        if saved['status']=='SEALED' and saved['valid_episodes']==425 and saved['checkpoint_sha256']==file_sha(checkpoint):
            for r in saved['episodes']:
                if file_sha(Path(r['prediction_path']))!=r['prediction_sha256'] or file_sha(Path(r['margins_path']))!=r['margins_sha256']:raise ValueError('completed output changed')
            print(json.dumps(dict(status='PRESERVED_COMPLETE_EVALUATION',GPU_calls=0,original_wall_seconds=saved['actual_scheduler_wall_seconds'])),flush=True)
            return saved
    train=json.loads((checkpoint.parent/'training_receipt.json').read_text())
    if train['status'] not in ('COMPLETE_FIXED_COVERAGE_TRANSDUCTIVE_DIAGNOSTIC','COMPLETE_FIXED_COVERAGE_PSR_STUDY') or train['steps']!=8973 or train['target_GT_open_attempts']:
        raise ValueError('complete original8973 training and access audit required')
    _,gpus,apps=audit_gpus(1,minimum_free_mib=20480)
    chosen=sorted([g for g in gpus if g['free_mib']>=20480],key=lambda g:(g['processes'],-g['free_mib'],g['physical_gpu']))[:min(workers,4)]
    if not chosen:raise RuntimeError('no memory-fit GPU for read-only evaluation')
    rows=read_jsonl(ROOT/'manifests/dance_val_runtime.jsonl');videos=sorted({r['video_id'] for r in rows});weights={}
    for video in videos:
        path=Path(next(r for r in rows if r['video_id']==video)['predicted_tracks']);weights[video]=len(path.read_text().splitlines())
    shards=[dict(shard=i,videos=[],candidate_observations=0) for i in range(len(chosen))]
    for video in sorted(videos,key=lambda v:(-weights[v],v)):
        shard=min(shards,key=lambda s:(s['candidate_observations'],s['shard']))
        shard['videos'].append(video);shard['candidate_observations']+=weights[video]
    plan=root/'whole_video_shards.json'
    if plan.exists():
        old=json.loads(plan.read_text())
        shards=old
    else:atomic_json(plan,shards)
    started=time.perf_counter();attempt=root/'launchers'/str(time.time_ns());attempt.mkdir(parents=True)
    clock=root/'first_launch_epoch.json'
    if not clock.exists():atomic_json(clock,dict(epoch_seconds=time.time(),checkpoint_sha256=file_sha(checkpoint)))
    atomic_json(attempt/'gpu_admission.json',dict(all_gpus=gpus,chosen=chosen,processes=apps,sharing_authorized=True,
        allocator_budget_mib=16384,headroom_mib=4096,checkpoint_sha256=file_sha(checkpoint)))
    children=[];preserved=[];pending=[]
    for shard in shards:
        out=root/'shards'/f'shard{shard["shard"]:02d}';done=out/method/'run_metadata.json'
        if done.exists():
            data=json.loads(done.read_text())
            if data['status']=='SEALED' and data['checkpoint_sha256']==file_sha(checkpoint):
                for r in data['episodes']:
                    if file_sha(Path(r['prediction_path']))!=r['prediction_sha256'] or file_sha(Path(r['margins_path']))!=r['margins_sha256']:raise ValueError('preserved episode changed')
                preserved.append(shard['shard']);continue
        pending.append(shard)
    if len(pending)>len(chosen):raise RuntimeError('insufficient admitted GPUs for remaining shards; completed shards remain preserved')
    for shard,gpu in zip(pending,chosen):
        out=root/'shards'/f'shard{shard["shard"]:02d}'
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=gpu['uuid'],UDA_PHYSICAL_GPU=str(gpu['physical_gpu']),
            OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1')
        command=[sys.executable,str(ROOT/'scripts/evaluate_mllm_distill.py'),'--checkpoint',str(checkpoint),
            '--output-root',str(out),'--method',method,'--videos',*sorted(shard['videos'])]
        log=(root/f'shard{shard["shard"]:02d}_stdout.txt').open('a')
        proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        worker=dict(shard=shard['shard'],pid=proc.pid,started_epoch_seconds=time.time(),**gpu)
        child_started=time.perf_counter()
        def relay(process=proc,handle=log,record=worker,begin=child_started):
            for line in process.stdout:
                handle.write(line);handle.flush()
                if line.startswith('{"event": "video_sealed"'):print(line.strip(),flush=True)
            record['elapsed_seconds']=time.perf_counter()-begin
        thread=threading.Thread(target=relay);thread.start()
        children.append((proc,log,thread,worker))
    print(json.dumps(dict(event='full_evaluation_started',workers=[r[3] for r in children],preserved_shards=preserved)),flush=True)
    results=[]
    # Exactly one blocking parent completion wait; no agent-level process polling.
    for proc,log,thread,row in children:
        code=proc.wait();thread.join();log.close();results.append(dict(row,exit_code=code))
        print(json.dumps(dict(event='evaluation_worker_finished',**results[-1])),flush=True)
    status='COMPLETE' if all(r['exit_code']==0 for r in results) else 'FAILED_SHARDS_RETAINED'
    atomic_json(attempt/'launcher_status.json',dict(status=status,workers=results,preserved_shards=preserved,wall_seconds=time.perf_counter()-started))
    if status!='COMPLETE':raise RuntimeError('retain healthy completed shards; repair only failed episodes')
    result=finish(root,method,rows,shards,checkpoint,started)
    history=[json.loads(p.read_text()) for p in root.glob('launchers/*/launcher_status.json')]
    result['successful_shard_body_seconds']=result['worker_seconds']
    result['worker_seconds']=sum(r['elapsed_seconds'] for a in history for r in a['workers'])
    result['actual_scheduler_wall_seconds']=time.time()-json.loads(clock.read_text())['epoch_seconds']
    result['all_attempt_worker_costs_included']=True
    atomic_json(root/method/'run_metadata.json',result)
    print(json.dumps(dict(event='full_evaluation_sealed',method=method,episodes=result['valid_episodes'],
        scheduler_wall_seconds=result['actual_scheduler_wall_seconds'],worker_seconds=result['worker_seconds'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output-root',type=Path,required=True);p.add_argument('--workers',type=int,default=4);p.add_argument('--method',default='MLLM_DISTILL')
    a=p.parse_args();launch(a.checkpoint,a.output_root,a.method,a.workers)
