"""Cold-process A100 anytime benchmark, with all unprocessed rows retained."""
import time
PROCESS_STARTED=time.perf_counter()
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from flex_tta.runtime.pf_receipts import atomic_json,append_json,file_sha
from flex_uda.a100.runtime import OUTPUT,metadata,load_student,teacher,source_head,video_spec

METHODS=('MINUS_v8P','YT_PRIOR','UNIFORM_K')
GRID=(0,60,120,180,240,300,450,600)


class BudgetExpired(Exception):pass


def slice_metadata(scope):
    m=metadata()
    if scope=='full':return m
    selected=sorted(m['videos'])[:2];banks={};episodes=[];offset=0
    for video in selected:
        b=m['videos'][video];frames=sorted({f for f,t in b['observations']})[:128];keep=set(frames)
        ids=[i for i,(f,t) in enumerate(b['observations']) if f in keep]
        banks[video]=dict(b,observations=[b['observations'][i] for i in ids],lines=[b['lines'][i] for i in ids],
            frame_count=len(frames),evaluation_last_frame=max(frames))
        for q in (q for q in m['episodes'] if q['video_id']==video):
            episodes.append(dict(q,start=offset,stop=offset+len(ids),frame_count=len(frames)));offset+=len(ids)
    return dict(videos=banks,episodes=episodes,candidate_query_rows=offset)


def prepare():
    m=slice_metadata('short');out=OUTPUT/'stage3';out.mkdir(parents=True,exist_ok=True)
    p=dict(version='A100_WALL_V1',methods=METHODS,budget_seconds=600,time_grid=GRID,seed=0,
        short_videos={v:dict(first_frame=min(f for f,t in b['observations']),last_frame=b['evaluation_last_frame'],
            observations=len(b['observations']),queries=sum(q['video_id']==v for q in m['episodes'])) for v,b in m['videos'].items()},
        selection='first two lexically sorted videos; first128 observed frames each; all their real query texts; no GT',
        resources='one A100 40GB per arm, student and teacher share the device; no cross-arm answer cache',
        rotation='three repeats, alternation of physical GPUs2 and3; GPUs0/1 dedicated to ordered causal streams',
        cold='new process and newly loaded required models; shared server OS page cache not flushed',
        budget_includes='process import, student/teacher load, initial parameter fingerprint, image IO, real features/teacher forwards, fitting, prediction snapshot IO',
        cutoff='only atomically committed snapshots completed by deadline count; late or unfinished work receives no prediction credit',
        unprocessed='empty predictions over remaining samples; all query-frame sequences remain in evaluation',
        short_call_cap='rho=1, one nominal question per eligible pair total for all methods; active distributes the same cap over four inherited rounds',
        full_call_cap='rho=.3 with the fixed metadata-derived per-video cap; all45 videos, actual total wall reported',
        MINUS='inherited information gain/routing/recheck and evidence-only fast fit; zero LoRA',
        YT_PRIOR='query-round-robin descending native mean score; first observation crop; YesTrack>=.4; unasked Native fallback',
        UNIFORM_K='seed0 uniform query-track pairs; midpoint routed view; Yes>=.5; unasked UNKNOWN omitted',
        required_preprocessing='MINUS and YT load live student; UNIFORM appearance-only slice can skip unused student; full motion gates require live presence features',
        scientific_limit='600-second score is measured, not preselected to show active strategy wins; short slice is not evidence of full-domain generalization',
        registration_before_GT=True)
    path=out/'preregistration.json'
    if path.exists():assert json.loads(path.read_text())==json.loads(json.dumps(p))
    else:atomic_json(path,p)
    print(json.dumps(dict(status='WALL_PROTOCOL_REGISTERED',path=str(path))),flush=True)


class Journal:
    def __init__(self,out,m,budget,start):
        self.out=out;self.m=m;self.budget=budget;self.start=start
        self.mask=np.zeros(m['candidate_query_rows'],dtype=bool);self.n=0
        self.phases={};self.calls=0
    def elapsed(self):return time.perf_counter()-self.start
    def check(self):
        if self.budget and self.elapsed()>=self.budget:raise BudgetExpired()
    def phase(self,name,event,**detail):
        append_json(self.out/'phase_events.jsonl',dict(phase=name,event=event,elapsed_seconds=self.elapsed(),**detail))
    def publish(self,event):
        self.check();path=self.out/'snapshots'/f'{self.n:05d}.npz';path.parent.mkdir(parents=True,exist_ok=True)
        tmp=path.with_suffix('.partial')
        with tmp.open('wb') as f:np.savez_compressed(f,mask=self.mask);f.flush();os.fsync(f.fileno())
        tmp.replace(path);elapsed=self.elapsed()
        if self.budget and elapsed>self.budget:path.unlink();raise BudgetExpired()
        append_json(self.out/'snapshots.jsonl',dict(index=self.n,elapsed_seconds=elapsed,path=str(path),event=event,
            actual_calls=self.calls,prediction_rows=int(self.mask.sum())))
        self.n+=1
        print(json.dumps(dict(event='ANYTIME_SNAPSHOT',index=self.n-1,method=self.out.name,phase=event,elapsed=self.elapsed(),calls=self.calls)),flush=True)
    def set_video(self,video,local):
        qs=[q for q in self.m['episodes'] if q['video_id']==video]
        for i,q in enumerate(qs):self.mask[q['start']:q['stop']]=local[i]


class MeteredTeacher:
    def __init__(self,client,journal):self.client,self.journal=client,journal
    def __getattr__(self,name):return getattr(self.client,name)
    def ask_many(self,requests):
        j=self.journal;before=dict(self.client.stats)
        j.phase('teacher_request','BEGIN',attempted_requests=len(requests))
        answers=self.client.ask_many(requests)
        j.phase('teacher_request','END',returned_requests=len(answers),
            costs={k:self.client.stats[k]-before[k] for k in ('logical_calls','new_calls','cache_hits','forward_seconds')})
        return answers


def start_teacher(specs,out,j):
    j.phase('teacher_load','BEGIN');t=time.perf_counter()
    client=teacher(specs,out/'teacher.sqlite');j.phases['teacher_init']=time.perf_counter()-t
    j.phase('teacher_load','END',wall_seconds=j.phases['teacher_init'],model_load_and_hash_seconds=client.backend.load_seconds)
    j.check();return MeteredTeacher(client,j)


def prediction(state,fc,queries,bank,temperature,bias):
    from scipy.special import expit
    from flex_uda.active.acquisition import expression_types,exclusive_group
    post=state.posterior;selected=post>.5;selected[state.output_absence()]=False
    if exclusive_group([q['query_text'] for q in queries],'SportsMOT'):
        selected &= np.arange(len(queries))[:,None]==post.argmax(axis=0)[None]
    tid={t:i for i,t in enumerate(fc.tracks)};mask=np.zeros((len(queries),len(bank['observations'])),dtype=bool)
    p=expit(fc.raw_presence/temperature+bias)
    for qi,q in enumerate(queries):
        motion='MOTION' in expression_types(q['query_text'])
        for i,(f,t) in enumerate(bank['observations']):mask[qi,i]=selected[qi,tid[t]] and (not motion or p[qi,fc.frame_index[f]]>=.5)
    return mask


def run_worker(method,scope,repeat,budget,start):
    from flex_uda.presence.runtime import runtime_guard,episode_from_bank,cache_for
    from flex_uda.active.v6_frames import FrameCacheV6
    from flex_uda.active.v8p import setup,belief_state,PRESUPPOSED,variant_questions
    from flex_uda.active.v8 import fast_step_v8
    from flex_uda.active.baselines import budget_coverage,video_rng
    from flex_uda.active.belief import normalize_track_scores
    from flex_uda.active.acquisition import route,expression_types
    from scripts.active_v8p_run import live_round
    access=runtime_guard();torch.set_num_threads(8);torch.manual_seed(0);np.random.seed(0)
    out=OUTPUT/'stage3'/scope/f'repeat_{repeat}'/method;out.mkdir(parents=True,exist_ok=True)
    m=slice_metadata(scope);j=Journal(out,m,budget,start);j.publish('EMPTY_BEFORE_MODELS')
    atomic_json(out/'runtime_metadata.json',dict(videos={v:dict(frame_count=b['frame_count'],last_frame=max(f for f,t in b['observations'])) for v,b in m['videos'].items()},
        episodes=[{k:q[k] for k in ('video_id','expression_id','query_text','start','stop')} for q in m['episodes']],
        candidate_query_rows=m['candidate_query_rows'],acquisition_version='fixed_pool_per_registered_round'))
    _,_,ds=setup(PRESUPPOSED);slope=1.;temperature=1.;bias=0.;model=None;client=None
    needs_student=(method!='UNIFORM_K' or any('MOTION' in expression_types(q['query_text']) for q in m['episodes']))
    status='RUNNING';last_video=None
    try:
        if needs_student:
            j.phase('student_load','BEGIN')
            model,tok,transform,loaded=load_student(out/'model_log',torch.device('cuda:0'),verify_weights=False)
            j.phases['student_load']=loaded['model_load_seconds'];j.phase('student_load','END',wall_seconds=loaded['model_load_seconds']);j.check()
        specs=[video_spec(v,b) for v,b in m['videos'].items()]
        for vi,video in enumerate(sorted(m['videos'])):
            j.check();last_video=video;vb=time.perf_counter();bank=m['videos'][video]
            queries=[q for q in m['episodes'] if q['video_id']==video];texts=[q['query_text'] for q in queries]
            tids=sorted({t for f,t in bank['observations']});positions={t:[i for i,(f,ti) in enumerate(bank['observations']) if ti==t] for t in tids}
            tracks=defaultdict(list)
            for o in video_spec(video,bank)['observations']:tracks[o['track_id']].append(o)
            lengths=[len(tracks[t]) for t in tids]
            cap=int(np.floor((1. if scope=='short' else .3)*sum(n>=3 for n in lengths)*len(queries)))
            fc=None
            video_needs_student=(method!='UNIFORM_K' or any('MOTION' in expression_types(text) for text in texts))
            if video_needs_student:
                j.phase('initial_features','BEGIN',video=video);t=time.perf_counter()
                ep=episode_from_bank(video,bank,'SportsMOT',transform);cache=cache_for(model,tok,ep,torch.device('cuda:0'))
                fc=FrameCacheV6(model,cache,bank,queries,sorted({f for f,t in bank['observations']}),source_head(),out/'features'/video,store_pcd=False)
                fc.score();phase_seconds=time.perf_counter()-t;j.phases['initial_features']=j.phases.get('initial_features',0.)+phase_seconds
                j.phase('initial_features','END',video=video,wall_seconds=phase_seconds);j.check()
                native=np.asarray(fc.margin)>0
            if method=='MINUS_v8P':
                fractions,p,_=fc.decode(temperature,bias)
                state=belief_state(PRESUPPOSED,fc.track_means(),fractions,p.max(1),track_lengths=lengths,slope=slope)
                state.recompute_presence(ds.values);asked=defaultdict(set);rng=video_rng(video);carry=0
                j.set_video(video,prediction(state,fc,queries,bank,temperature,bias));j.publish('INITIAL_STUDENT_DECODE')
                if client is None:
                    client=start_teacher(specs,out,j)
                from flex_uda.active.acquisition import round_budget
                for r in range(4):
                    allowance=round_budget(cap,r)+carry;used=set();got=0
                    possible=variant_questions(PRESUPPOSED,state,texts,lengths,ds.values,r,rng=rng)
                    while got<allowance:
                        j.check();remaining=[q for q in possible if (q.query,q.track) not in used]
                        t=time.perf_counter();answers=live_round(state,remaining,min(8,allowance-got),asked,tracks,fc,queries,'SportsMOT',video,bank,client)
                        j.phases['teacher_service']=j.phases.get('teacher_service',0.)+time.perf_counter()-t
                        if not answers:break
                        for a in answers:
                            qi=next(i for i,q in enumerate(queries) if q['expression_id']==a['query_id']);ti=tids.index(a['track_id']);used.add((qi,ti))
                            ds.add((video,a['query_id'],a['track_id']),a['effective_view'],a['yes_probability']>=.5)
                        got+=len(answers);j.calls+=len(answers)
                        append_json(out/'answer_batches.jsonl',dict(video=video,round=r+1,answers=answers))
                        j.set_video(video,prediction(state,fc,queries,bank,temperature,bias));j.publish('TEACHER_EVIDENCE')
                    carry=allowance-got;t=time.perf_counter();ds.fit();state.recompute_presence(ds.values)
                    j.phase('fast_fit','BEGIN',video=video,round=r+1)
                    fast=fast_step_v8(state,fc.raw_presence,temperature=temperature,bias=bias)
                    temperature,bias=fast['temperature'],fast['bias'];fc.refresh(state,temperature,bias)
                    j.phases['fast_fit_decode']=j.phases.get('fast_fit_decode',0.)+time.perf_counter()-t
                    j.phase('fast_fit','END',video=video,round=r+1,wall_seconds=time.perf_counter()-t)
                    append_json(out/'fast_fits.jsonl',dict(video=video,round=r+1,fast=fast))
                    j.set_video(video,prediction(state,fc,queries,bank,temperature,bias));j.publish('FAST_FIT')
                slope=state.slope
            else:
                local=native.copy() if method=='YT_PRIOR' else np.zeros((len(queries),len(bank['observations'])),dtype=bool)
                if method=='YT_PRIOR':
                    means=np.stack([np.asarray(fc.margin[:,positions[t]]).mean(1) for t in tids],1)
                    covered=budget_coverage(video,queries,tids,normalize_track_scores(means),cap,prior=True)
                else:
                    pairs=[(q,t) for q in range(len(queries)) for t,n in enumerate(lengths) if n>=3]
                    covered=[pairs[i] for i in video_rng(video).choice(len(pairs),cap,replace=False)] if cap else []
                j.set_video(video,local);j.publish('INITIAL_BASELINE')
                if client is None:
                    client=start_teacher(specs,out,j)
                for a in range(0,len(covered),8):
                    j.check();part=covered[a:a+8];requests=[]
                    for qi,ti in part:
                        history=tracks[tids[ti]];obs=history[0] if method=='YT_PRIOR' else history[(len(history)-1)//2]
                        requests.append(dict(target='SportsMOT',video_id=video,frame_id=obs['frame_id'],track_id=tids[ti],
                            query_id=queries[qi]['expression_id'],query_text=queries[qi]['query_text'],view='CROP' if method=='YT_PRIOR' else route(texts[qi])))
                    t=time.perf_counter();answers=client.ask_many(requests);j.phases['teacher_service']=j.phases.get('teacher_service',0.)+time.perf_counter()-t
                    for (qi,ti),answer in zip(part,answers):
                        rows=positions[tids[ti]];local[qi,rows]=answer['yes_probability']>=(.4 if method=='YT_PRIOR' else .5)
                        if method=='UNIFORM_K' and 'MOTION' in expression_types(texts[qi]):
                            from scipy.special import expit
                            local[qi,rows] &= [expit(fc.raw_presence[qi,fc.frame_index[bank['observations'][i][0]]])>=.5 for i in rows]
                    j.calls+=len(answers);append_json(out/'answer_batches.jsonl',dict(video=video,answers=answers))
                    j.set_video(video,local);j.publish('BASELINE_ANSWERS')
            atomic_json(out/'videos'/video/'receipt.json',dict(video=video,cap=cap,wall_seconds=time.perf_counter()-vb,
                feature_score_records=fc.score_records if fc else [],method=method,GT_open_attempts=len(access)))
            if fc is not None:fc.close();del fc
        status='COMPLETE';j.publish('TASK_COMPLETE')
    except BudgetExpired:
        status='BUDGET_EXHAUSTED'
    finally:
        if client is not None:
            # Version check is sufficient here: no teacher optimizer exists. A
            # full bytewise seal is performed when there is remaining budget.
            client.backend.check_versions()
            if not budget or j.elapsed()+10<budget:
                j.phase('teacher_final_seal','BEGIN');t=time.perf_counter();teacher_receipt=client.close();j.phases['teacher_final_seal']=time.perf_counter()-t
                j.phase('teacher_final_seal','END',wall_seconds=j.phases['teacher_final_seal'])
            else:teacher_receipt=dict(client.stats,final_parameter_versions_unchanged=True,full_final_hash_not_run_at_deadline=True)
        else:teacher_receipt=None
        atomic_json(out/'worker_receipt.json',dict(status=status,method=method,scope=scope,repeat=repeat,last_video=last_video,
            phases=j.phases,teacher=teacher_receipt,actual_calls=j.calls,GT_open_attempts=len(access),
            wall_seconds=j.elapsed(),physical_GPU=os.environ.get('CUDA_VISIBLE_DEVICES'),needs_student=needs_student,
            cold_definition='process/model cold; shared OS page cache retained'))
    print(json.dumps(dict(event='WALL_WORKER_DONE',method=method,scope=scope,status=status,seconds=j.elapsed())),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepare',action='store_true');p.add_argument('--method',choices=METHODS)
    p.add_argument('--scope',choices=['short','full'],default='short');p.add_argument('--repeat',type=int,default=0)
    p.add_argument('--budget',type=float,default=600);p.add_argument('--start',type=float,default=None);a=p.parse_args()
    if a.prepare:prepare()
    else:run_worker(a.method,a.scope,a.repeat,a.budget,a.start or PROCESS_STARTED)
