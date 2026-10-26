"""GT-blind whole-video shards for the fixed P1 probe, <=4 physical GPUs."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import time
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from scripts.audit_teacher_leakage import METHODS,validate_receipt

ORDER=('YES_FULL','YES_ENSEMBLE_BUDGET_MATCHED','STRUCTURED_EVIDENCE',
    'EVIDENCE_VERIFIED','YES_ENSEMBLE_B_PLUS','YESTRACK_TCP_TRP')


def whole_video_shards(rows,count=4):
    counts={video:sum(r['video_id']==video for r in rows) for video in sorted({r['video_id'] for r in rows})}
    videos=[[] for _ in range(count)];load=[0]*count
    for video in sorted(counts,key=lambda v:(-counts[v],v)):
        chosen=min(range(count),key=lambda i:(load[i],i));videos[chosen].append(video);load[chosen]+=counts[video]
    return [[r for r in rows if r['video_id'] in set(group)] for group in videos]


def merge_method(method,manifest,phase_root,shard_manifests):
    expected=[json.loads(s) for s in manifest.read_text().splitlines()];ids=[r['sample_id'] for r in expected]
    destination=phase_root/method;destination.mkdir(exist_ok=True)
    if (destination/'receipt.json').exists():
        receipt=json.loads((destination/'receipt.json').read_text())
        validate_receipt(receipt,method,file_sha(manifest),len(ids))
        if file_sha(destination/'teacher_outputs.jsonl')!=receipt['predictions_sha256']:raise ValueError('sealed merged prediction changed')
        return receipt
    receipts=[];by_id={};provenance=[]
    for number,shard_manifest in enumerate(shard_manifests):
        folder=phase_root/'shards'/f'shard{number:02d}'/method
        receipt_path=folder/'receipt.json';receipt=json.loads(receipt_path.read_text())
        count=sum(1 for _ in shard_manifest.open())
        validate_receipt(receipt,method,file_sha(shard_manifest),count)
        path=folder/'teacher_outputs.jsonl'
        if file_sha(path)!=receipt['predictions_sha256']:raise ValueError('shard prediction hash mismatch')
        with path.open() as handle:
            for line in handle:
                sid=json.loads(line)['sample_id']
                if sid in by_id:raise ValueError('episode candidate duplicated across shards')
                by_id[sid]=line
        receipts.append(receipt)
        provenance.append(dict(receipt=str(receipt_path),receipt_sha256=file_sha(receipt_path),
            manifest=str(shard_manifest),manifest_sha256=file_sha(shard_manifest),
            predictions=str(path),predictions_sha256=file_sha(path),samples=count,
            physical_gpu=receipt['physical_gpu'],worker_elapsed_seconds=receipt['worker_elapsed_seconds']))
    if set(by_id)!=set(ids):raise ValueError('whole-video merge does not cover fixed global sample set')
    output=destination/'teacher_outputs.jsonl'
    with output.open('w') as handle:
        for sid in ids:handle.write(by_id[sid])
    merged=dict(receipts[0]);budget={}
    for key in receipts[0]['budget']:
        values=[r['budget'][key] for r in receipts]
        budget[key]=max(values) if key=='peak_allocated_bytes' else sum(values)
    frozen=dict(receipts[0]['frozen'])
    for key in ['forward_batches','model_load_and_initial_hash_seconds']:
        frozen[key]=sum(r['frozen'][key] for r in receipts)
    merged.update(manifest_sha256=file_sha(manifest),samples=len(ids),planned_manifest_samples=len(ids),
        videos=len({r['video_id'] for r in expected}),video_queries=len({(r['video_id'],r['query_id']) for r in expected}),
        predictions_sha256=file_sha(output),budget=budget,frozen=frozen,
        worker_elapsed_seconds=sum(r['worker_elapsed_seconds'] for r in receipts),
        inference_and_final_hash_seconds=sum(r['inference_and_final_hash_seconds'] for r in receipts),
        model_load_seconds=sum(r['model_load_seconds'] for r in receipts),
        worker_time_interpretation='sum of four allocated worker elapsed times, NOT scheduler wall time',
        physical_gpu=[r['physical_gpu'] for r in receipts],gpu_uuid=[r['gpu_uuid'] for r in receipts],
        source_worker_receipts=provenance,whole_video_shards=True,
        GT_access_audit=dict(target_GT_open_attempts=[],target_GT_reads=0),
        isolation=dict(kernel_read_whitelist=True,kernel_stage_write_denied=True,
            landlock_abi=min(r['isolation']['landlock_abi'] for r in receipts),
            per_shard=[r['isolation'] for r in receipts]),
        persistent_cache_reopen_check=all(r['persistent_cache_reopen_check'] for r in receipts),
        source_commits=sorted({r['experiment_commit'] for r in receipts}))
    atomic_json(destination/'receipt.json',merged);return merged


def run_sharded(a,config,manifest,stage,resources,python,root):
    phase_root=a.run_root.resolve()/'full'/a.condition;phase_root.mkdir(parents=True,exist_ok=True)
    all_rows=[json.loads(s) for s in manifest.read_text().splitlines()];shards=whole_video_shards(all_rows)
    if len(all_rows)!=8381 or len({s['video_id'] for s in all_rows})!=25:raise ValueError('require fixed complete P1 sampling')
    manifests=[];plan=[]
    for number,rows in enumerate(shards):
        folder=phase_root/'shards'/f'shard{number:02d}';folder.mkdir(parents=True,exist_ok=True)
        path=folder/'manifest.jsonl';payload=''.join(json.dumps(row,sort_keys=True)+'\n' for row in rows)
        if path.exists() and path.read_text()!=payload:raise ValueError('preserve existing GT-blind shard assignment')
        if not path.exists():path.write_text(payload)
        manifests.append(path);plan.append(dict(shard=number,videos=sorted({s['video_id'] for s in rows}),
            pairs=len(rows),manifest_sha256=file_sha(path)))
    methods=[m for m in ORDER if not a.methods or m in a.methods]
    if set(methods)!=set(METHODS):raise ValueError('full sealed P1 requires all six original methods; use episode resume, not a changed method set')
    admission=resources(4);attempt=phase_root/'launcher_attempts'/str(time.time_ns());attempt.mkdir(parents=True)
    atomic_json(attempt/'gpu_admission.json',admission)
    atomic_json(phase_root/'whole_video_shard_plan.json',dict(protocol='SAME_FIXED8381PAIRS_WHOLE_VIDEO_ONLY',
        target_GT_used=False,shards=plan,methods=methods,global_manifest_sha256=file_sha(manifest),
        no_TCP_TRP_window_split=True,numeric_config_unchanged=True))
    print(json.dumps(dict(phase='full',mode='WHOLE_VIDEO_FOUR_SHARDS',shards=plan)),flush=True)
    started=time.perf_counter();records=[]
    def worker(number,gpu):
        items=[]
        for method in methods:
            out=phase_root/'shards'/f'shard{number:02d}'/method;receipt_path=out/'receipt.json'
            if receipt_path.exists():
                old=json.loads(receipt_path.read_text());validate_receipt(old,method,file_sha(manifests[number]),len(shards[number]))
                if old['config_sha256']!=file_sha(config) or old['predictions_sha256']!=file_sha(out/'teacher_outputs.jsonl'):
                    raise ValueError('existing shard belongs to a different protocol or changed predictions')
                items.append(dict(method=method,shard=number,status='PRESERVED_SEALED_RESULT',receipt=str(receipt_path)));continue
            command=[python,str(root/'scripts/run_teacher_probe.py'),'--config',str(config),'--manifest',str(manifests[number]),
                '--staging-root',str(stage),'--method',method,'--output-root',str(out),'--reliability-config',str(a.rules.resolve())]
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu['index']),PYTHONNOUSERSITE='1',PYTHONPATH=str(root),
                OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
            log_path=phase_root/f'shard{number:02d}_{method}_stdout.txt';begin=time.perf_counter();epoch=time.time()
            with log_path.open('a') as log:
                process=subprocess.Popen(command,cwd=root,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
                print(json.dumps(dict(started=method,shard=number,physical_gpu=gpu['index'],pid=process.pid)),flush=True)
                for line in process.stdout:
                    log.write(line);log.flush()
                    if '"completed":' in line or 'Traceback' in line or 'Error:' in line:
                        print(f'shard{number:02d} {method} '+line.rstrip(),flush=True)
                code=process.wait()
            item=dict(method=method,shard=number,physical_gpu=gpu['index'],gpu_uuid=gpu['uuid'],pid=process.pid,
                exit_code=code,elapsed_seconds=time.perf_counter()-begin,start_epoch_seconds=epoch,
                samples=len(shards[number]),log=str(log_path),receipt=str(receipt_path))
            items.append(item);print(json.dumps(dict(finished=method,shard=number,exit_code=code,elapsed_seconds=item['elapsed_seconds'])),flush=True)
            if code:break
        return items
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(worker,i,gpu) for i,gpu in enumerate(admission['selected'])]
        for future in futures:records.extend(future.result())
    failed=any(r.get('exit_code',0)!=0 for r in records)
    status=dict(status='FAILED_ATTEMPTS_RETAINED' if failed else 'COMPLETE_FIXED_PHASE',phase='full',
        protocol='DEVELOPMENT_TEACHER_PROBE',condition=a.condition,workers=records,
        actual_wall_seconds=time.perf_counter()-started,manifest_sha256=file_sha(manifest),
        gpu_admission=admission,no_other_process_stopped=True,max_physical_GPUs=4,whole_video_shards=plan)
    atomic_json(attempt/'launcher_status.json',status)
    if failed:raise SystemExit(1)
    for method in methods:merge_method(method,manifest,phase_root,manifests)
    status['actual_wall_seconds']=time.perf_counter()-started
    atomic_json(attempt/'launcher_status.json',status);atomic_json(phase_root/'phase_completion.json',status)
    print(json.dumps(status,indent=2),flush=True)
