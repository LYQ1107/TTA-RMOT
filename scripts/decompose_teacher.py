"""Frozen paired-view probe worker. Kernel whitelist; no evaluator/GT imports."""
from argparse import ArgumentParser
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.decompose_common import RUN, PUBLIC, REVISION, require_plan, guard, sandbox, atomic_json, file_sha
from flex_uda.data.manifests import read_jsonl, write_jsonl


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def worker(shard, smoke=False):
    import torch
    import yaml
    from flex_uda.teacher.yestrack_backend import FrozenYesTrack
    from flex_uda.teacher.decompose_inputs import PairedViews, images_sha, CLIP_PROMPT, FRAME_PROMPT
    require_plan(); access = guard(); started = time.perf_counter()
    plan = json.loads((RUN/'probe/plan.json').read_text())
    assignment = json.loads((RUN/'probe/shards.json').read_text())
    if file_sha(plan['manifest_path']) != plan['manifest_sha256'] or file_sha(plan['videos_path']) != plan['videos_sha256']:
        raise ValueError('Changed fixed input manifest')
    frames = [r for r in read_jsonl(plan['manifest_path']) if [r['target'],r['video_id']] in assignment['shards'][shard]]
    videos = json.loads(Path(plan['videos_path']).read_text())
    if smoke:
        # Fixed first two candidate/query pairs in the first selected frame of
        # each target. They are cached under the identical production key.
        seen = set(); sample = []
        for frame in frames:
            if frame['target'] not in seen:
                frame = dict(frame, candidates=frame['candidates'][:1], queries=frame['queries'][:2])
                sample.append(frame); seen.add(frame['target'])
        frames = sample
    reused = plan['dance_reused_CROP']
    if file_sha(reused['outputs_path']) != reused['outputs_sha256']:
        raise ValueError('Changed original Dance CROP outputs')
    old = {(r['video_id'],r['frame_id'],r['track_id'],r['query_id']): r
           for r in read_jsonl(reused['outputs_path']) if r['method']=='T1_CROP_YESNO'}
    config = yaml.safe_load((ROOT/'configs/teacher/base_qwen2b.yaml').read_text())
    prompts = yaml.safe_load((ROOT/'configs/teacher/prompts_v1.yaml').read_text())
    if config['revision'] != REVISION or config['lora_path'] is not None:
        raise ValueError('Only the fixed base Qwen teacher is authorized')
    for key,name in [('original_backend_sha256','yestrack_backend.py'),('original_visual_builder_sha256','visual_input_builder.py')]:
        if file_sha(ROOT/'flex_uda/teacher'/name)!=plan[key]:
            raise ValueError('Do not modify original T1 code')
    out = RUN/'probe/workers'/str(shard); out.mkdir(parents=True,exist_ok=True)
    for d in ('hf','cuda','tmp'):
        (out/d).mkdir(exist_ok=True)
    os.environ.update(HF_HOME=str(out/'hf'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        CUDA_CACHE_PATH=str(out/'cuda'),PYTORCH_KERNEL_CACHE_PATH=str(out/'cuda'),TRITON_CACHE_DIR=str(out/'cuda'),TMPDIR=str(out/'tmp'))
    isolation = sandbox([*[v['image_root'] for v in videos], config['model_path'], config['yestrack_source'],
                         '/home/szgx/anaconda3/envs/vllmedit', plan['manifest_path'], plan['videos_path']], [out])
    torch.cuda.set_device(0); torch.manual_seed(0)
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(.9,20*2**30/total))
    backend = FrozenYesTrack(config)
    views = PairedViews(videos,backend.upstream)
    db = sqlite3.connect(out/'calls.sqlite')
    db.execute('CREATE TABLE IF NOT EXISTS calls (key TEXT PRIMARY KEY,payload TEXT)')
    code_sha = file_sha(ROOT/'flex_uda/teacher/decompose_inputs.py')
    budgets = {m: Counter() for m in ('CROP','CLIP','FRAME_BOX')}
    results = []; checks = []
    try:
        for frame in frames:
            target,v,t = frame['target'],frame['video_id'],frame['frame_id']; video = target,v
            if file_sha(frame['image_path']) != frame['image_sha256']:
                raise ValueError('Changed current probe image')
            crops = {}
            for method in budgets:
                budget = budgets[method]; pending = []
                torch.cuda.reset_peak_memory_stats()
                def flush():
                    if not pending:
                        return
                    raw = backend.forward([p[1] for p in pending],[p[2] for p in pending])
                    budget['new_logical_calls'] += len(pending); budget['forward_batches'] += 1
                    budget['forward_seconds'] += raw[0]['batch_forward_seconds']
                    for (cache_key,images,prompt,info),r in zip(pending,raw):
                        record = dict(r,**info,method=method,target=target,video_id=v,frame_id=t,
                                      request_id=cache_key,prompt=prompt,prompt_sha256=digest(prompt),
                                      image_sha256=frame['image_sha256'],model_revision=REVISION,
                                      new_input_code_sha256=code_sha,batch_id=digest([p[0] for p in pending]))
                        db.execute('INSERT INTO calls VALUES (?,?)',(cache_key,json.dumps(record,sort_keys=True)))
                        results.append(record)
                        if method=='CROP':
                            crops[info['track_id'],info['query_id']] = record
                    db.commit(); pending.clear()
                for c in frame['candidates']:
                    tid = c['track_id']; ob = views.observations[video,t,tid]
                    crop = views.patch(video,ob)
                    full = views.frame(video,t)[0]
                    x1,y1,x2,y2 = ob.bbox_xyxy
                    coord = f'{(x1+x2)/2/full.width:.3f} {(y1+y2)/2/full.height:.3f}'
                    clip_images,clip_info = views.clip(video,t,tid) if method=='CLIP' else (None,None)
                    boxed = views.frame_box(video,t,tid) if method=='FRAME_BOX' else None
                    for q in frame['queries']:
                        qid,text = q['query_id'],q['query_text']; budget['planned_pairs'] += 1
                        if method=='CROP' and target=='Dance':
                            r = old[v,t,tid,qid]
                            h = hashlib.sha256(crop.tobytes()).hexdigest()
                            prompt = prompts['official_single'].format(coord=coord,sentence=text)
                            if h!=r['visual_input_sha256'] or prompt!=r['prompt'] or r['frozen_weight_sha256']!=backend.weight_sha:
                                raise ValueError('STOP: reused Dance CROP input/model not exact')
                            record = dict(r,method='CROP',target=target,reused_original_CROP=True)
                            results.append(record); crops[tid,qid]=record; budget['reused_original_calls']+=1
                            continue
                        if method=='CLIP' and clip_info['fallback']:
                            record = dict(crops[tid,qid],method='CLIP',clip=clip_info,clip_fallback=True,
                                          fallback_CROP_request=crops[tid,qid]['request_id'])
                            results.append(record); budget['fallback_to_CROP']+=1
                            continue
                        if method=='CROP':
                            images=[crop]; prompt=prompts['official_single'].format(coord=coord,sentence=text)
                        elif method=='CLIP':
                            images=clip_images; prompt=CLIP_PROMPT.format(expression=text)
                        else:
                            images=[boxed]; prompt=FRAME_PROMPT.format(expression=text)
                        visual = images_sha(images)
                        info = dict(track_id=tid,query_id=qid,bbox_xywh=c['bbox_xywh'],visual_input_sha256=visual,
                                    clip=clip_info if method=='CLIP' else None)
                        cache_key = digest(dict(revision=REVISION,weight=backend.weight_sha,code=code_sha,method=method,
                            target=target,video=v,frame=t,track=tid,query=qid,image=frame['image_sha256'],visual=visual,prompt=prompt))
                        cached = db.execute('SELECT payload FROM calls WHERE key=?',(cache_key,)).fetchone()
                        if cached:
                            record=json.loads(cached[0]); results.append(record); budget['cached_calls']+=1
                            if method=='CROP':
                                crops[tid,qid]=record
                        else:
                            pending.append((cache_key,images,prompt,info))
                        if len(pending)>=8:
                            flush()
                flush()
                budget['peak_allocated_bytes']=max(budget['peak_allocated_bytes'],torch.cuda.max_memory_allocated())
            checks.append(dict(target=target,video=v,frame=t,complete_methods=3,candidates=len(frame['candidates']),queries=len(frame['queries'])))
            if len(checks)%8==0 or smoke:
                print(json.dumps(dict(event='DECOMPOSE_TEACHER_PROGRESS',shard=shard,smoke=smoke,
                    completed_frames=len(checks),total_frames=len(frames),target=target,video=v,frame=t,
                    new_calls=sum(b['new_logical_calls'] for b in budgets.values()))),flush=True)
        frozen=backend.seal_frozen()
        if access:
            raise ValueError('Teacher attempted target GT/referring annotations')
        receipt=dict(status='REAL_PAIRED_VIEW_SMOKE_PASS' if smoke else 'ALL_ASSIGNED_PAIRED_PROBES_SEALED',
            shard=shard,smoke=smoke,budgets={m:dict(b) for m,b in budgets.items()},frames=checks,
            frozen=frozen,GT_open_attempts=access,isolation=isolation,worker_seconds=time.perf_counter()-started,
            physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),student_updates=0,teacher_updates=0,
            original_backend_sha256=plan['original_backend_sha256'],original_visual_builder_sha256=plan['original_visual_builder_sha256'],
            new_input_code_sha256=code_sha,manifest_sha256=plan['manifest_sha256'])
        if not smoke:
            p=out/'outputs.jsonl'; write_jsonl(p,results)
            receipt.update(outputs_path=str(p),outputs_sha256=file_sha(p),output_pairs=len(results))
        atomic_json(out/('smoke_receipt.json' if smoke else 'receipt.json'),receipt)
        print(json.dumps(dict(event='DECOMPOSE_TEACHER_SEALED',shard=shard,smoke=smoke,budgets=receipt['budgets'],worker_seconds=receipt['worker_seconds'])),flush=True)
    finally:
        atomic_json(out/('smoke_access.json' if smoke else 'access.json'),dict(GT_open_attempts=access))


if __name__=='__main__':
    p=ArgumentParser();p.add_argument('--shard',type=int,required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    worker(a.shard,a.smoke)
