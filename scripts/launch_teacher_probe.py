#!/usr/bin/env python3
"""One blocking launcher for audited <=4GPU independent teacher workers."""
from concurrent.futures import ThreadPoolExecutor
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import yaml
from flex_tta.runtime.pf_receipts import atomic_json,file_sha

PYTHON='/data2/user/TTA-RMOT-assets/teacher-p1/env/bin/python'
ASSETS=Path('/data2/user/TTA-RMOT-assets/teacher-p1')
GROUPS=[['YES_FULL','YESTRACK_TCP_TRP'],['YES_ENSEMBLE_BUDGET_MATCHED','YES_ENSEMBLE_B_PLUS'],
    ['STRUCTURED_EVIDENCE'],['EVIDENCE_VERIFIED']]


def resources(count):
    gpu_csv=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.total,memory.free',
        '--format=csv,noheader,nounits'],text=True)
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory',
        '--format=csv,noheader,nounits'],text=True)
    processes=Counter(line.split(',')[0].strip() for line in apps.splitlines() if line.strip())
    gpus=[]
    for line in gpu_csv.splitlines():
        index,uuid,name,total,free=map(str.strip,line.split(','))
        if float(free)>=20480:gpus.append(dict(index=int(index),uuid=uuid,name=name,total_mib=float(total),
            free_mib=float(free),processes=processes[uuid]))
    gpus.sort(key=lambda g:(g['processes'],-g['free_mib'],g['index']))
    if len(gpus)<count:raise RuntimeError(f'only{len(gpus)} memory-fit GPUs for{count} workers; do not stop other jobs')
    return dict(selected=gpus[:count],all_gpus_csv=gpu_csv,all_compute_processes_csv=apps,
        policy='>=16GiB allocation+4GiB free margin; fewest processes, then most free; shared allowed; no other task stopped')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=['functionality','calibrate','smoke','full'],required=True)
    p.add_argument('--condition',choices=['T0','T1'],default='T0');p.add_argument('--rules',type=Path)
    p.add_argument('--run-root',type=Path,default=ROOT/'results/runs/teacher_probe_p1_20261026_v1')
    p.add_argument('--methods',nargs='+');p.add_argument('--execute',action='store_true');a=p.parse_args()
    config=ROOT/'configs/teacher'/('base_qwen2b.yaml' if a.condition=='T0' else 'yestrack_v2_lora.yaml')
    cfg=yaml.safe_load(config.read_text())
    if a.condition=='T1' and not cfg.get('lora_path'):raise FileNotFoundError('T1 officialV2 adapter not available; user permitted honest missing condition, not substitution')
    groups=[['SOURCE_RELIABILITY']] if a.phase=='calibrate' else GROUPS
    if a.methods:groups=[[m for m in group if m in a.methods] for group in groups];groups=[g for g in groups if g]
    if not groups:raise ValueError('no requested methods')
    if a.phase in ('smoke','full') and (not a.rules or not a.rules.exists()):raise ValueError('freeze source-only reliability rules before complete target probe')
    if a.phase=='calibrate':
        manifest=ASSETS/'calibration/input/source_manifest.jsonl';stage=ASSETS/'calibration/input'
    else:
        full=a.phase=='full';manifest=ROOT/'results/teacher_probe'/('manifest.jsonl' if full else 'smoke_manifest.jsonl')
        stage=ASSETS/('staging_full' if full else 'staging_smoke')
    phase_root=a.run_root.resolve()/a.phase/a.condition
    print(json.dumps(dict(phase=a.phase,groups=groups,condition=a.condition,manifest=str(manifest),execute=a.execute)),flush=True)
    if a.phase=='full' and not a.execute:
        from scripts.launch_teacher_probe_sharded import whole_video_shards
        rows=[json.loads(s) for s in manifest.read_text().splitlines()]
        print(json.dumps(dict(mode='WHOLE_VIDEO_FOUR_SHARDS',shards=[dict(index=i,pairs=len(shard),
            videos=sorted({s['video_id'] for s in shard})) for i,shard in enumerate(whole_video_shards(rows))])),flush=True)
    if not a.execute:return
    completed=phase_root/'phase_completion.json'
    requested=[m for group in groups for m in group]
    if completed.exists() and all((phase_root/m/'receipt.json').exists() for m in requested):
        old_phase=json.loads(completed.read_text())
        if old_phase['status']!='COMPLETE_FIXED_PHASE' or old_phase['manifest_sha256']!=file_sha(manifest):
            raise ValueError('existing phase belongs to a different fixed protocol')
        for method in requested:
            old=json.loads((phase_root/method/'receipt.json').read_text())
            if old['manifest_sha256']!=file_sha(manifest) or old['config_sha256']!=file_sha(config):
                raise ValueError('existing method configuration changed')
            if old['predictions_sha256']!=file_sha(phase_root/method/'teacher_outputs.jsonl'):
                raise ValueError('sealed method predictions changed')
        print(json.dumps(dict(status='PRESERVED_COMPLETE_PHASE',phase=a.phase,
            original_wall_seconds=old_phase['actual_wall_seconds'],no_GPU_worker_started=True)),flush=True)
        return
    if a.phase=='full':
        for group in GROUPS:
            for method in group:
                prior=json.loads((a.run_root/'smoke'/a.condition/method/'receipt.json').read_text())
                if prior['status']!='SEALED_FIXED_PROBE' or prior['samples']!=1649 or prior['GT_access_audit']['target_GT_open_attempts']:
                    raise ValueError('all six complete GT-free smoke methods must pass before full probe')
        from scripts.launch_teacher_probe_sharded import run_sharded
        run_sharded(a,config,manifest,stage,resources,PYTHON,ROOT);return
    admission=resources(len(groups));phase_root.mkdir(parents=True,exist_ok=True)
    attempt=phase_root/'launcher_attempts'/str(time.time_ns());attempt.mkdir(parents=True)
    atomic_json(attempt/'gpu_admission.json',admission)
    started=time.perf_counter();records=[]
    def group_worker(group,gpu):
        items=[]
        for method in group:
            out=phase_root/method;receipt=out/'receipt.json'
            if receipt.exists():
                old=json.loads(receipt.read_text())
                if old['manifest_sha256']!=file_sha(manifest) or old['config_sha256']!=file_sha(config):raise ValueError('existing result belongs to a different fixed protocol')
                items.append(dict(method=method,status='PRESERVED_SEALED_RESULT',receipt=str(receipt)));continue
            command=[PYTHON,str(ROOT/'scripts/run_teacher_probe.py'),'--config',str(config),'--manifest',str(manifest),
                '--staging-root',str(stage),'--method',method,'--output-root',str(out)]
            if a.phase=='functionality':command+=['--limit','4']
            if a.rules:command+=['--reliability-config',str(a.rules.resolve())]
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu['index']),PYTHONNOUSERSITE='1',PYTHONPATH=str(ROOT),
                OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
            log_path=phase_root/(method+'_stdout.txt')
            begin=time.perf_counter()
            with log_path.open('a') as log:
                process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
                print(json.dumps(dict(started=method,physical_gpu=gpu['index'],gpu_uuid=gpu['uuid'],pid=process.pid)),flush=True)
                for line in process.stdout:
                    log.write(line);log.flush()
                    if '"completed":' in line or 'Traceback' in line or 'Error:' in line:print(method+' '+line.rstrip(),flush=True)
                code=process.wait()
            item=dict(method=method,physical_gpu=gpu['index'],gpu_uuid=gpu['uuid'],pid=process.pid,exit_code=code,
                elapsed_seconds=time.perf_counter()-begin,log=str(log_path),receipt=str(receipt))
            items.append(item);print(json.dumps(dict(finished=method,exit_code=code,elapsed_seconds=item['elapsed_seconds'])),flush=True)
            if code:break
        return items
    with ThreadPoolExecutor(max_workers=len(groups)) as pool:
        futures=[pool.submit(group_worker,group,gpu) for group,gpu in zip(groups,admission['selected'])]
        for future in futures:records.extend(future.result())
    failed=any(r.get('exit_code',0)!=0 for r in records)
    status=dict(status='FAILED_ATTEMPTS_RETAINED' if failed else 'COMPLETE_FIXED_PHASE',phase=a.phase,
        protocol='DEVELOPMENT_TEACHER_PROBE',condition=a.condition,workers=records,
        actual_wall_seconds=time.perf_counter()-started,manifest_sha256=file_sha(manifest),
        gpu_admission=admission,no_other_process_stopped=True,max_physical_GPUs=len(groups))
    atomic_json(attempt/'launcher_status.json',status)
    if not failed:atomic_json(phase_root/'phase_completion.json',status)
    print(json.dumps(status,indent=2),flush=True)
    if failed:raise SystemExit(1)


if __name__=='__main__':main()
