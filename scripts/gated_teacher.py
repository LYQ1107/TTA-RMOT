"""One frozen T1 path, resumable cache; gates use no referring annotations."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.gated_common import *
from flex_uda.data.manifests import read_jsonl,write_jsonl

def key(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def worker(smoke=False):
    import yaml
    import torch
    from PIL import Image
    from flex_uda.teacher.yestrack_backend import FrozenYesTrack
    from flex_uda.teacher.visual_input_builder import VisualInputBuilder
    from flex_uda.teacher.isolation import restrict_filesystem
    from flex_tta.runtime.dance_episode import Observation
    setup();require_plan();attempts=deny_annotations();started=time.perf_counter();plan=json.loads((RUN/'teacher_plan.json').read_text())
    if file_sha(plan['manifest_path'])!=plan['manifest_sha256']:raise ValueError('teacher manifest changed')
    if file_sha(ROOT/'flex_uda/teacher/yestrack_backend.py')!=plan['backend_sha256'] or file_sha(ROOT/'flex_uda/teacher/visual_input_builder.py')!=plan['builder_sha256']:raise ValueError('unmodified T1 code required')
    frames=read_jsonl(plan['manifest_path'])
    if smoke:
        frames=[dict(next(r for r in frames if r['dataset']==d),candidates=next(r for r in frames if r['dataset']==d)['candidates'][:1],queries=next(r for r in frames if r['dataset']==d)['queries'][:3]) for d in ('Dance','KITTI')]
    out=RUN/'teacher';out.mkdir(exist_ok=True)
    for name in ('hf','cuda','tmp'):(out/name).mkdir(exist_ok=True)
    os.environ.update(HF_HOME=str(out/'hf'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUDA_CACHE_PATH=str(out/'cuda'),
        PYTORCH_KERNEL_CACHE_PATH=str(out/'cuda'),TRITON_CACHE_DIR=str(out/'cuda'),TMPDIR=str(out/'tmp'))
    config=yaml.safe_load((ROOT/'configs/teacher/base_qwen2b.yaml').read_text());prompts=yaml.safe_load((ROOT/'configs/teacher/prompts_v1.yaml').read_text())
    isolation=restrict_filesystem([*[r['image_path'] for r in frames],plan['manifest_path'],config['model_path'],config['yestrack_source'],
        ROOT/'flex_uda',ROOT/'flex_tta',ROOT/'scripts',ROOT/'configs',Path(sys.prefix),Path(sys.base_prefix),
        '/home/szgx/anaconda3/envs/vllmedit','/usr','/lib','/lib64','/bin','/etc','/proc','/sys'],[out,'/dev','/tmp'])
    torch.manual_seed(0);torch.cuda.set_device(0);backend=FrozenYesTrack(config)
    class SourceViews(VisualInputBuilder):
        def frame(self,video,frame):
            k=video,frame
            if k not in self.frame_cache:
                with Image.open(SOURCE/'KITTI/training/image_02'/video/f'{frame:06d}.png') as im:self.frame_cache[k]=[im.convert('RGB'),None]
                while len(self.frame_cache)>self.frame_limit:self.frame_cache.popitem(last=False)
            return self.frame_cache[k]
    views={'Dance':VisualInputBuilder(v2.DATA,backend.upstream,320,.2),'KITTI':SourceViews(SOURCE,backend.upstream,320,.2)}
    db=sqlite3.connect(out/'progress.sqlite');db.execute('CREATE TABLE IF NOT EXISTS calls (key TEXT PRIMARY KEY,payload TEXT)')
    counts=[];outputs=[]
    try:
        for frame in frames:
            dataset,v,t=frame['dataset'],frame['video_id'],frame['frame_id'];view=views[dataset]
            if file_sha(frame['image_path'])!=frame['image_sha256']:raise ValueError('teacher image changed')
            full=view.frame(v,t)[0];width,height=full.size;pending=[];result=[]
            torch.cuda.reset_peak_memory_stats();stat=dict(dataset=dataset,video=v,gate_mode=frame['gate_mode'],frame=t,
                requested_calls=0,new_calls=0,cache_hits=0,forward_batches=0,forward_seconds=0.,peak_allocated_bytes=0)
            def flush():
                if not pending:return
                raw=backend.forward([[j['patch']] for j in pending],[j['prompt'] for j in pending]);seconds=raw[0]['batch_forward_seconds']
                batch=key([j['request_id'] for j in pending])
                for job,r in zip(pending,raw):
                    payload=dict(r,**{k:x for k,x in job.items() if k!='patch'},batch_id=batch,
                        dataset=dataset,video_id=v,frame_id=t,image_sha256=frame['image_sha256'],model_revision=config['revision'])
                    db.execute('INSERT INTO calls VALUES (?,?)',(job['request_id'],json.dumps(payload,sort_keys=True)));result.append(payload)
                db.commit();stat['new_calls']+=len(pending);stat['forward_batches']+=1;stat['forward_seconds']+=seconds;pending.clear()
            for c in frame['candidates']:
                x,y,w,h=c['bbox_xywh'];obs=Observation(t,c['track_id'],(x,y,x+w,y+h),c['MOT_row'])
                patch=view.patch(v,obs);visual=hashlib.sha256(patch.tobytes()).hexdigest();coord=f'{(x+w/2)/width:.3f} {(y+h/2)/height:.3f}'
                for q in frame['queries']:
                    prompt=prompts['official_single'].format(coord=coord,sentence=q['query_text'])
                    job=dict(track_id=c['track_id'],query_id=q['query_id'],bbox_xywh=c['bbox_xywh'],bbox_sha256=key(c['bbox_xywh']),
                        prompt=prompt,prompt_sha256=key(prompt),visual_input_sha256=visual,patch=patch)
                    request=key(dict(revision=config['revision'],weight=backend.weight_sha,dataset=dataset,video=v,frame=t,
                        track=c['track_id'],query=q['query_id'],prompt=job['prompt_sha256'],image=frame['image_sha256'],bbox=job['bbox_sha256'],visual=visual))
                    job['request_id']=request;stat['requested_calls']+=1;cached=db.execute('SELECT payload FROM calls WHERE key=?',(request,)).fetchone()
                    if cached:result.append(json.loads(cached[0]));stat['cache_hits']+=1
                    else:pending.append(job)
                    if len(pending)>=8:flush()
            flush();stat['peak_allocated_bytes']=torch.cuda.max_memory_allocated();counts.append(stat)
            outputs.extend(dict(r,gate_mode=frame['gate_mode']) for r in result)
            atomic_json(out/('smoke_progress.json' if smoke else 'progress.json'),dict(completed_frames=len(counts),last=stat))
            print(json.dumps(dict(event='GATED_T1_FRAME_COMPLETE',**stat)),flush=True)
        frozen=backend.seal_frozen();filename='smoke_outputs.jsonl' if smoke else 'outputs.jsonl';write_jsonl(out/filename,outputs)
        db_total=db.execute('SELECT count(*) FROM calls').fetchone()[0]
        receipt=dict(status='REAL_T1_SMOKE_PASS' if smoke else 'ALL_T1_OUTPUTS_SEALED',frames=len(frames),rows=len(outputs),
            outputs_path=str(out/filename),outputs_sha256=file_sha(out/filename),manifest_sha256=plan['manifest_sha256'],
            GT_open_attempts=len(attempts),isolation=isolation,frozen=frozen,counts=counts,cache_unique_calls=db_total,
            original_backend_sha256=plan['backend_sha256'],original_builder_sha256=plan['builder_sha256'],
            worker_seconds=time.perf_counter()-started,physical_gpu=os.environ['CUDA_VISIBLE_DEVICES'],student_updates=0,teacher_updates=0)
        if attempts:raise RuntimeError('teacher attempted GT access')
        atomic_json(out/('smoke_receipt.json' if smoke else 'receipt.json'),receipt)
        print(json.dumps(dict(event='GATED_T1_SEALED',smoke=smoke,rows=len(outputs),worker_seconds=receipt['worker_seconds'])),flush=True)
    finally:
        db.close();atomic_json(out/('smoke_access.json' if smoke else 'access.json'),dict(GT_open_attempts=len(attempts),attempts=attempts))

def launch():
    from scripts.launch_uda_workers import audit_gpus
    setup();require_plan()
    if json.loads((RUN/'reproduction.json').read_text())['status']!='PASS':raise ValueError('reproduce controls before teacher')
    start=time.perf_counter();folder=RUN/'teacher_launchers'/str(time.time_ns());folder.mkdir(parents=True);jobs=[]
    for smoke in (True,False):
        done=RUN/'teacher'/('smoke_receipt.json' if smoke else 'receipt.json')
        if done.exists():
            r=json.loads(done.read_text())
            if file_sha(r['outputs_path'])!=r['outputs_sha256']:raise ValueError('changed sealed teacher output')
            continue
        while True:
            try:chosen,gpus,apps=audit_gpus(1,minimum_free_mib=12288);break
            except RuntimeError as e:print(json.dumps(dict(event='GATED_RESOURCE_WAIT',reason=str(e))),flush=True);time.sleep(20)
        gpu=chosen[0];label='smoke' if smoke else 'full'
        atomic_json(folder/(label+'_admission.json'),dict(selected=gpu,all_gpus=gpus,processes=apps,maximum_task_GPUs=1,sharing_authorized=True))
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu['physical_gpu']),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
        command=[TEACHER_PY,str(ROOT/'scripts/gated_teacher.py'),'worker']+(['--smoke'] if smoke else [])
        begin=time.perf_counter();log=folder/(label+'.log')
        with log.open('w') as h:
            p=subprocess.Popen(command,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
            for line in p.stdout:h.write(line);h.flush();print(line,end='',flush=True)
            code=p.wait()
        r=dict(smoke=smoke,pid=p.pid,physical_gpu=gpu['physical_gpu'],exit_code=code,worker_seconds=time.perf_counter()-begin,log=str(log))
        jobs.append(r);atomic_json(folder/(label+'.json'),r)
        if code:raise RuntimeError('T1 failed, cache retained; inspect '+str(log))
    records=[]
    for path in sorted((RUN/'teacher_launchers').glob('*/*.json')):
        if path.name in ('smoke.json','full.json'):records.append(json.loads(path.read_text()))
    result=dict(status='ALL_FROZEN_TEACHER_WORK_COMPLETE',jobs=records,maximum_physical_GPUs=1,
        actual_this_launcher_wall_seconds=time.perf_counter()-start,GPU_process_hours=sum(j['worker_seconds'] for j in records)/3600)
    atomic_json(RUN/'teacher_launch_completion.json',result);atomic_json(PUBLIC/'teacher_launch_completion.json',result)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['worker','launch']);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    worker(a.smoke) if a.phase=='worker' else launch()
