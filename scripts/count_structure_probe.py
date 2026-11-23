"""GT-free fixed probe preparation/worker. No target GT evaluator imports."""
import argparse
from collections import defaultdict
import hashlib
import itertools
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.count_structure_common import RUN,PUBLIC,v2,setup,require_plan,video_banks
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from flex_uda.data.manifests import read_jsonl,write_jsonl
from flex_uda.teacher.structure_prompts import choice_prompt,som_prompt,draw_som,choice_forward,som_forward,FONT

def key(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def prepare():
    setup();require_plan();attempts=v2.deny_gt();meta,_,_,_=v2.load_bank();root=RUN/'probe';root.mkdir(exist_ok=True)
    videos=sorted({q['video_id'] for q in meta['episodes']})[::5]
    train_texts={' '.join(r['query_text'].lower().split()) for r in read_jsonl('/data2/user/TTA-RMOT-assets/train2val/20261031/runtime_train.jsonl')}
    rows=[]
    for video,queries,tracks,frames,ids,unique,inverse in video_banks(meta):
        if video not in videos:continue
        q=queries[0];start,stop=q['frame_indices']['start'],q['frame_indices']['stop']
        anchors=[int(np.floor(start+j*(stop-1-start)/9+.5)) for j in range(10)]
        for j,t in enumerate(anchors):
            image=Path(q['image_root'])/f'{t:06d}.png';candidates=[]
            for i,line in sorted(tracks.get(t,{}).items()):
                f=list(map(float,line.split(',')));candidates.append(dict(track_id=i,bbox_xywh=f[2:6],MOT_row=line))
            rows.append(dict(video_id=video,frame_id=t,probe_frame_index=j,image_path=str(image),image_sha256=file_sha(image),
                candidates=candidates,queries=[dict(query_id=x['expression_id'],query_text=x['query_text'],
                    seen_train_text=' '.join(x['query_text'].lower().split()) in train_texts) for x in queries]))
    write_jsonl(root/'manifest.jsonl',rows)
    searched=[p for p in ('/data2/user','/data3/user','/home/user/.cache/huggingface','/home/szgx/.cache/huggingface') if Path(p).exists()]
    scan=subprocess.run(['rg','--files','--hidden',*searched,'-g','config.json','-g','!**/node_modules/**','-g','!**/.git/**'],
        capture_output=True,text=True)
    available=[]
    for name in scan.stdout.splitlines():
        if 'qwen3-vl-8b' not in name.lower():continue
        path=Path(name).parent
        if (path/'adapter_config.json').exists():continue
        config=json.loads((path/'config.json').read_text())
        if config.get('quantization_config'):continue
        if (path/'model.safetensors.index.json').exists():
            index=json.loads((path/'model.safetensors.index.json').read_text())
            valid=all((path/f).exists() for f in set(index['weight_map'].values()))
        else:valid=(path/'model.safetensors').exists()
        if valid:available.append(str(path))
    models={'2B':dict(config=str(ROOT/'configs/teacher/base_qwen2b.yaml'),
        model_path='/data2/user/TTA-RMOT-assets/teacher-p1/models/Qwen3-VL-2B-Instruct',
        revision='89644892e4d85e24eaac8bacfd4f463576704203',methods=['T1_CROP_YESNO','T2_CROP_CHOICE','T3_FRAME_SOM'])}
    if available:
        path=sorted(available)[0];models['8B']=dict(model_path=path,revision='LOCAL_BASE_CONFIG_SHA256:'+file_sha(Path(path)/'config.json'),
            methods=['T2_CROP_CHOICE','T3_FRAME_SOM'])
    result=dict(status='FIXED_GT_FREE_PROBE_MANIFEST',videos=videos,frames=len(rows),
        manifest_path=str(root/'manifest.jsonl'),manifest_sha256=file_sha(root/'manifest.jsonl'),
        candidates=sum(len(r['candidates']) for r in rows),planned_logical_calls_2B=dict(
            T1_CROP_YESNO=sum(len(r['candidates'])*17 for r in rows),T2_CROP_CHOICE=sum(len(r['candidates']) for r in rows),T3_FRAME_SOM=len(rows)*17),
        models=models,local8B_available=bool(available),model_search_roots=searched,
        model_search_permission_notes=[line for line in scan.stderr.splitlines() if 'Permission denied' in line],
        no_download=True,target_GT_open_attempts=len(attempts),font=dict(path=FONT,size=24,sha256=file_sha(FONT)),
        scope='5video50frame probe only; unobserved frames UNKNOWN, not full RMOT',student_updates=0)
    if len(rows)!=50 or len(videos)!=5 or attempts:raise ValueError('fixed probe/audit failed')
    atomic_json(RUN/'probe_plan.json',result);atomic_json(PUBLIC/'probe_plan.json',result)
    print(json.dumps(dict(event='COUNT_PROBE_PREPARED',**{k:result[k] for k in ('videos','frames','candidates','local8B_available','planned_logical_calls_2B')})),flush=True)

def decisions(raw,method,frame):
    out={};candidates=frame['candidates'];queries=frame['queries']
    for r in raw:
        v,t=frame['video_id'],frame['frame_id']
        if method=='T1_CROP_YESNO':out[v,t,r['track_id'],r['query_id']]=r['yes_probability']
        elif method=='T2_CROP_CHOICE':
            for j,q in enumerate(queries):out[v,t,r['track_id'],q['query_id']]=r['option_probabilities'][j]
        else:
            for j,c in enumerate(candidates):out[v,t,c['track_id'],r['query_id']]=float(j+1 in r['numbers']) if r['valid'] else None
    return out

def agreements(records,frames,methods):
    all_scores={m:{} for m in methods};temporal={}
    for frame in frames:
        fkey=frame['video_id'],frame['frame_id']
        for m in methods:all_scores[m].update(decisions(records[m][fkey],m,frame))
    for m,score in all_scores.items():
        pairs=valid=same=0
        by=defaultdict(list)
        for f in frames:by[f['video_id']].append(f)
        for video,fs in by.items():
            for a,b in zip(fs,fs[1:]):
                ids={x['track_id'] for x in a['candidates']}&{x['track_id'] for x in b['candidates']}
                for i in ids:
                    for q in a['queries']:
                        x,y=score[video,a['frame_id'],i,q['query_id']],score[video,b['frame_id'],i,q['query_id']];pairs+=1
                        if x is not None and y is not None:valid+=1;same+=int((x>.5)==(y>.5))
        temporal[m]=dict(planned_pairs=pairs,valid_pairs=valid,agreement=same/valid if valid else None)
    pairwise={}
    for a,b in itertools.combinations(methods,2):
        keys=set(all_scores[a])&set(all_scores[b]);valid=[k for k in keys if all_scores[a][k] is not None and all_scores[b][k] is not None]
        pairwise[a+'__'+b]=dict(planned_pairs=len(keys),valid_pairs=len(valid),
            agreement=sum((all_scores[a][k]>.5)==(all_scores[b][k]>.5) for k in valid)/len(valid) if valid else None)
    return dict(binary_threshold='p>0.5; T3 hard scores',pairwise=pairwise,adjacent_probe_frame_same_track=temporal)

def worker(model_name,smoke=False):
    import yaml
    import torch
    from PIL import Image
    from flex_uda.teacher.yestrack_backend import FrozenYesTrack
    from flex_uda.teacher.visual_input_builder import VisualInputBuilder
    from flex_uda.teacher.isolation import restrict_filesystem
    from flex_tta.runtime.dance_episode import DanceEpisode
    setup();require_plan();access=v2.deny_gt();started=time.perf_counter()
    plan=json.loads((RUN/'probe_plan.json').read_text());spec=plan['models'][model_name];frames=read_jsonl(plan['manifest_path'])
    if file_sha(plan['manifest_path'])!=plan['manifest_sha256']:raise ValueError('probe manifest changed')
    if smoke:frames=frames[:1]
    config=yaml.safe_load((ROOT/'configs/teacher/base_qwen2b.yaml').read_text());config.update(model_path=spec['model_path'],revision=spec['revision'])
    prompts=yaml.safe_load((ROOT/'configs/teacher/prompts_v1.yaml').read_text())
    out=RUN/'probe'/model_name;out.mkdir(parents=True,exist_ok=True)
    for name in ('hf','cuda','tmp'): (out/name).mkdir(exist_ok=True)
    os.environ.update(HF_HOME=str(out/'hf'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        CUDA_CACHE_PATH=str(out/'cuda'),PYTORCH_KERNEL_CACHE_PATH=str(out/'cuda'),
        TRITON_CACHE_DIR=str(out/'cuda'),TMPDIR=str(out/'tmp'))
    read_roots=[Path(r['image_path']).parent for r in frames]
    isolation=restrict_filesystem([*read_roots,plan['manifest_path'],spec['model_path'],config['yestrack_source'],
        v2.TRACKS,ROOT/'flex_uda',ROOT/'flex_tta',ROOT/'scripts',ROOT/'configs',Path(sys.prefix),Path(sys.base_prefix),
        '/home/szgx/anaconda3/envs/vllmedit','/usr','/lib','/lib64','/bin','/etc','/proc','/sys'],[out,'/dev','/tmp'])
    torch.manual_seed(0);torch.cuda.set_device(0);backend=FrozenYesTrack(config)
    views=VisualInputBuilder(v2.DATA,backend.upstream,320,.2)
    db=sqlite3.connect(out/'progress.sqlite');db.execute('CREATE TABLE IF NOT EXISTS calls (key TEXT PRIMARY KEY,payload TEXT)')
    methods=spec['methods'];records={m:{} for m in methods};budgets={}
    try:
        for method in methods:
            torch.cuda.reset_peak_memory_stats();budget=dict(planned_logical_calls=0,new_logical_calls=0,cached_logical_calls=0,
                forward_batches_or_generations=0,model_forward_calls=0,forward_path_seconds=0.,parse_failures=0)
            for frame in frames:
                v,t=frame['video_id'],frame['frame_id'];queries=frame['queries'];jobs=[]
                if method=='T3_FRAME_SOM':
                    with Image.open(frame['image_path']) as image:som=draw_som(image.convert('RGB'),frame['candidates'])
                    visual_hash=hashlib.sha256(som.tobytes()).hexdigest()
                    for q in queries:jobs.append((dict(query_id=q['query_id']),som_prompt(q['query_text']),som,visual_hash))
                else:
                    if v not in views.episodes:views.episodes[v]=DanceEpisode(v2.DATA,v2.TRACKS,v)
                    ep=views.episodes[v]
                    for c in frame['candidates']:
                        obs=ep.observations_by_frame[t][c['track_id']];patch=views.patch(v,obs)
                        x,y,x2,y2=obs.bbox_xyxy;coord=f'{(x+x2)/2/ep.width:.3f} {(y+y2)/2/ep.height:.3f}'
                        h=hashlib.sha256(patch.tobytes()).hexdigest();base=dict(track_id=c['track_id'],bbox_xywh=c['bbox_xywh'])
                        if method=='T1_CROP_YESNO':
                            for q in queries:jobs.append((dict(base,query_id=q['query_id']),prompts['official_single'].format(coord=coord,sentence=q['query_text']),patch,h))
                        else:jobs.append((base,choice_prompt(queries,coord),patch,h))
                pending=[];output=[]
                def flush():
                    if not pending:return
                    before=backend.calls;images=[j[3] for j in pending];texts=[j[2] for j in pending]
                    if method=='T1_CROP_YESNO':
                        raw=backend.forward([[im] for im in images],texts);seconds=raw[0]['batch_forward_seconds']
                    elif method=='T2_CROP_CHOICE':raw,seconds=choice_forward(backend,images,texts)
                    else:
                        r,seconds=som_forward(backend,images[0],texts[0],len(frame['candidates']));raw=[r]
                    batch=key(dict(ids=[j[0] for j in pending],method=method))
                    for (cache_key,info,prompt,image,h),r in zip(pending,raw):
                        payload=dict(r,**info,method=method,model=model_name,video_id=v,frame_id=t,
                            image_sha256=frame['image_sha256'],visual_input_sha256=h,prompt=prompt,prompt_sha256=key(prompt),
                            request_id=cache_key,batch_id=batch,batch_forward_seconds=seconds,
                            model_revision=spec['revision'],frozen_weight_sha256=backend.weight_sha)
                        db.execute('INSERT INTO calls VALUES (?,?)',(cache_key,json.dumps(payload,sort_keys=True)));output.append(payload)
                    db.commit();budget['new_logical_calls']+=len(pending);budget['forward_batches_or_generations']+=1
                    budget['forward_path_seconds']+=seconds;budget['model_forward_calls']+=backend.calls-before;pending.clear()
                for info,prompt,image,h in jobs:
                    cache_key=key(dict(model=spec['revision'],weight=backend.weight_sha,method=method,video=v,frame=t,
                        input_image=frame['image_sha256'],visual=h,prompt=prompt,fields=info))
                    budget['planned_logical_calls']+=1
                    cached=db.execute('SELECT payload FROM calls WHERE key=?',(cache_key,)).fetchone()
                    if cached:output.append(json.loads(cached[0]));budget['cached_logical_calls']+=1
                    else:pending.append((cache_key,info,prompt,image,h))
                    if len(pending)>=(1 if method=='T3_FRAME_SOM' else 8):flush()
                flush();records[method][v,t]=output
                budget['parse_failures']+=sum(not r['valid'] for r in output) if method=='T3_FRAME_SOM' else 0
                print(json.dumps(dict(event='COUNT_PROBE_FRAME',model=model_name,method=method,video=v,frame=t,
                    logical_completed=budget['planned_logical_calls'],new_calls=budget['new_logical_calls'],parse_failures=budget['parse_failures'])),flush=True)
            budget['peak_allocated_bytes']=torch.cuda.max_memory_allocated();budgets[method]=budget
        frozen=backend.seal_frozen()
        if access:raise RuntimeError('teacher touched target annotations')
        if not smoke:
            outputs=out/'outputs.jsonl';write_jsonl(outputs,[r for m in methods for f in frames for r in records[m][f['video_id'],f['frame_id']]])
            stats=agreements(records,frames,methods);choices=[r for f in frames for r in records.get('T2_CROP_CHOICE',{}).get((f['video_id'],f['frame_id']),[])]
            stats['T2_none_argmax_fraction']=sum(r['argmax_letter']=='R' for r in choices)/len(choices) if choices else None
            atomic_json(out/'GT_free_agreements.json',stats)
        receipt=dict(status='REAL_FIXED_PROBE_SMOKE' if smoke else 'ALL_FIXED_PROBE_CALLS_SEALED',model=model_name,
            model_revision=spec['revision'],manifest_sha256=plan['manifest_sha256'],frames=len(frames),budgets=budgets,
            frozen=frozen,target_GT_open_attempts=access,isolation=isolation,student_updates=0,teacher_updates=0,
            worker_seconds=time.perf_counter()-started,physical_gpu=os.environ.get('UDA_PHYSICAL_GPU',os.environ.get('CUDA_VISIBLE_DEVICES')),
            prompt_code_sha256=file_sha(ROOT/'flex_uda/teacher/structure_prompts.py'),
            original_backend_sha256=file_sha(ROOT/'flex_uda/teacher/yestrack_backend.py'),
            original_visual_builder_sha256=file_sha(ROOT/'flex_uda/teacher/visual_input_builder.py'))
        if not smoke:receipt.update(outputs_path=str(outputs),outputs_sha256=file_sha(outputs),
            agreement_path=str(out/'GT_free_agreements.json'),agreement_sha256=file_sha(out/'GT_free_agreements.json'))
        atomic_json(out/('smoke_receipt.json' if smoke else 'receipt.json'),receipt)
        print(json.dumps(dict(event='COUNT_PROBE_MODEL_SEALED',model=model_name,smoke=smoke,budgets=budgets)),flush=True)
    finally:atomic_json(out/('smoke_access.json' if smoke else 'access.json'),dict(target_GT_open_attempts=access))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=('prepare','worker'));p.add_argument('--model',default='2B');p.add_argument('--smoke',action='store_true');a=p.parse_args()
    prepare() if a.phase=='prepare' else worker(a.model,a.smoke)
