"""GT-free fixed-evidence controls and causal-time, teacher-free baselines."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.special import expit,logit
from flex_tta.runtime.pf_receipts import atomic_json,append_json,file_sha
from flex_uda.a100.mechanism import OUT,PAIRS,metadata,history,FrozenInputs,generation_guard
from flex_uda.presence.predictions import seal_mask
from flex_uda.active.v8p import setup,belief_state,classify_protocol,CHECKLIST,PRESUPPOSED
from flex_uda.active.v8 import fast_step_v8
from flex_uda.active.acquisition import expression_types,exclusive_group
from scripts.active_v8p_run import ingest


def output_mask(state,fc,temperature,bias):
    posterior=state.posterior;selected=posterior>.5;selected[state.output_absence()]=False
    if exclusive_group([q['query_text'] for q in fc.queries],fc.target):
        selected &= np.arange(len(fc.queries))[:,None]==posterior.argmax(0)[None]
    p=expit(fc.raw_presence/temperature+bias);tid={t:i for i,t in enumerate(fc.tracks)}
    mask=np.zeros(fc.margin.shape,dtype=bool)
    for qi,q in enumerate(fc.queries):
        motion='MOTION' in expression_types(q['query_text'])
        for j,(frame,track) in enumerate(fc.observations):
            mask[qi,j]=selected[qi,tid[track]] and (not motion or p[qi,fc.frame_index[frame]]>=.5)
    return mask


def replay(pair,fusion,protocol,bank_routing=None):
    opened=generation_guard();begin=time.perf_counter();model,target=PAIRS[pair];m=metadata(target)
    effective=(CHECKLIST if target=='Dance' else PRESUPPOSED) if protocol=='EXPLICIT_TASK' else CHECKLIST
    name=fusion+'__'+protocol+('__BANK_'+bank_routing if bank_routing else '');dest=OUT/'fixed_replay'/pair/name
    if (dest/'predictions_seal.json').is_file():return
    _,em,ds=setup(effective,ablation='MINUS_DS' if fusion in ('FIXED','MEAN') else None)
    masks=np.zeros(m['candidate_query_rows'],dtype=bool);slope=1.;temperature=1.;bias=0.;total=0;video_receipts=[]
    base=history(pair);replacements={};replacement_count=0
    if bank_routing:
        samples=json.loads((OUT/'evidence_manifest.json').read_text())['samples']
        responses={(r['sample_id'],r['requested_view']):r for r in
            map(json.loads,(OUT/'evidence/responses.jsonl').read_text().splitlines())}
        for s in samples:
            if s['pair']!=pair:continue
            view=s['original_view'] if bank_routing=='ORIGINAL' else 'CROP'
            replacements[s['video_id'],s['query_id'],s['track_id'],s['frame_id']]=responses[s['sample_id'],view]
    for video in sorted(m['videos']):
        fc=FrozenInputs(pair,video);fraction,p,_=fc.decode(temperature,bias)
        start=json.loads((base/'videos'/video/'em_video_start.json').read_text())
        # The explicit arm exactly inherits its registered protocol prior.
        # A forced checklist on a presupposed task gets the generic .5 prior,
        # not the answer that every query is guaranteed present.
        rho,pi=(start['rho_hat'],start['pi_strata']) if effective==(CHECKLIST if target=='Dance' else PRESUPPOSED) else (.5,[.5]*3)
        state=belief_state(effective,fc.track_means(),fraction,p.max(1),track_lengths=[len(fc.by_track[t]) for t in fc.tracks],
            rho_hat=rho,pi_strata=pi,slope=slope)
        state.recompute_presence(ds.values)
        answers=json.loads((base/'videos'/video/'questions.json').read_text())
        if bank_routing:
            for i,a in enumerate(answers):
                key=(video,a['query_id'],a['track_id'],a['frame_id'])
                if key in replacements:
                    answers[i]=dict(a,**{k:v for k,v in replacements[key].items() if k not in ('round','kind','information_gain')})
                    replacement_count+=1
        tid={t:i for i,t in enumerate(fc.tracks)};qid={q['expression_id']:i for i,q in enumerate(fc.queries)}
        probabilities=defaultdict(list);rounds=[]
        for r in range(1,5):
            state.recompute_presence(ds.values)
            for a in (a for a in answers if a['round']==r):
                qi,ti=qid[a['query_id']],tid[a['track_id']]
                ingest(state,qi,ti,a['kind'],a,a['information_gain'],replay=True)
                ds.add((video,a['query_id'],a['track_id']),a['effective_view'],a['yes_probability']>=.5)
                probabilities[qi,ti].append(float(a['yes_probability']))
            ds.fit();state.recompute_presence(ds.values)
            if fusion=='MEAN':
                for (qi,ti),values in probabilities.items():
                    state.evidence[qi,ti]=float(np.clip(logit(np.clip(np.mean(values),1e-6,1-1e-6)),-2,2))
            fast=fast_step_v8(state,fc.raw_presence,temperature=temperature,bias=bias)
            temperature,bias=fast['temperature'],fast['bias']
            fraction,p,_=fc.decode(temperature,bias);state.replace_fractions(fraction)
            state.refresh_presence(p.max(1));state.replace_prior(scores=fc.track_means())
            rounds.append(dict(round=r,fast=fast,ds=ds.receipt(),absent=int(state.output_absence().sum())))
        local=output_mask(state,fc,temperature,bias)
        for qi,q in enumerate(fc.queries):masks[q['start']:q['stop']]=local[qi]
        (dest/'videos'/video).mkdir(parents=True,exist_ok=True)
        np.save(dest/'videos'/video/'mask.npy',local)
        np.savez(dest/'videos'/video/'belief.npz',posterior=state.posterior,tracks=fc.tracks,
            queries=[q['expression_id'] for q in fc.queries],absent=state.output_absence())
        row=dict(video=video,questions=len(answers),effective_protocol=effective,rounds=rounds)
        video_receipts.append(row);append_json(dest/'rounds.jsonl',row)
        total+=len(answers);slope=state.slope
        print(json.dumps(dict(event='FIXED_REPLAY_VIDEO',pair=pair,fusion=fusion,protocol=protocol,video=video,rows=int(local.sum()))),flush=True)
    assert not opened
    if bank_routing:assert replacement_count==len(replacements)
    seal_mask(dest,m,masks,target=target,model=model,method=name,GT_open_attempts=0,
        extra=dict(fixed_question_replay=True,questions_replayed=total,new_teacher_calls=0,neural_optimizer_updates=0,
            effective_protocol=effective,fusion=fusion,protocol=protocol,process_wall_seconds=time.perf_counter()-begin,
            mean_definition='one clipped log-odds of mean raw teacher probability per query-track' if fusion=='MEAN' else None,
            fixed_presence_schedule=False,not_live_counterfactual_acquisition=True,
            bank_routing=bank_routing,bank_replaced_positions=replacement_count,
            bank_routing_scope='partial intervention at predeclared100 positions, other original answers identical' if bank_routing else None))
    atomic_json(dest/'generation.json',dict(status='PASS_GT_FREE_SEALED',videos=video_receipts,GT_open_attempts=0))


def temporal(pair):
    attempts=generation_guard();model,target=PAIRS[pair];m=metadata(target)
    masks={k:np.zeros(m['candidate_query_rows'],dtype=bool) for k in ('RAW_MARGIN','EMA_MARGIN','HYSTERESIS_MARGIN')}
    if all((OUT/'temporal'/pair/k/'predictions_seal.json').is_file() for k in masks):return
    for video in sorted(m['videos']):
        fc=FrozenInputs(pair,video);raw=expit(fc.margin);ema=np.empty_like(raw);hysteresis=np.empty_like(raw,dtype=bool)
        for track,ids in fc.by_track.items():
            last_frame=None;state=None;bits=None
            for i in ids:
                frame=int(fc.observations[i,0]);value=raw[:,i]
                if last_frame is None or frame-last_frame>2:
                    state=value.copy();bits=value>.5
                else:
                    state=.2*value+.8*state
                    bits=np.where(value>=.6,True,np.where(value<=.4,False,bits))
                ema[:,i]=state;hysteresis[:,i]=bits;last_frame=frame
        local={'RAW_MARGIN':raw>.5,'EMA_MARGIN':ema>.5,'HYSTERESIS_MARGIN':hysteresis}
        for qi,q in enumerate(fc.queries):
            for name,bits in local.items():masks[name][q['start']:q['stop']]=bits[qi]
        print(json.dumps(dict(event='TEMPORAL_VIDEO',pair=pair,video=video)),flush=True)
    for name,mask in masks.items():
        seal_mask(OUT/'temporal'/pair/name,m,mask,target=target,model=model,method=name,GT_open_attempts=len(attempts),
            extra=dict(teacher_calls=0,neural_optimizer_updates=0,causal_filter=True,alpha=.2,on_probability=.6,off_probability=.4,
                reset_after_frame_gap=2,comparison='raw student classification, not a new v8P memory module',parameters_not_selected_from_GT=True))
    assert not attempts


def protocol_audit():
    attempts=generation_guard();rows=[]
    for pair,(_,target) in PAIRS.items():
        m=metadata(target);queries={v:[q['query_text'] for q in m['episodes'] if q['video_id']==v] for v in sorted(m['videos'])}
        full=classify_protocol(queries)['protocol'];single={v:classify_protocol({v:q})['protocol'] for v,q in queries.items()}
        explicit=CHECKLIST if target=='Dance' else PRESUPPOSED
        grouped=[];keys=sorted(queries)
        for start in range(0,len(keys),2):
            batch={v:queries[v] for v in keys[start:start+2]}
            grouped.append(dict(videos=list(batch),protocol=classify_protocol(batch)['protocol']))
        rows.append(dict(pair=pair,target=target,full_batch=full,explicit_task=explicit,
            single_video=single,two_video_batches=grouped,
            same_video_protocol_flips_due_to_batching=sum(p!=full for p in single.values()),
            formal_warning='query-set layout is a protocol heuristic, not learned visual existence'))
    atomic_json(OUT/'protocol/batch_invariance.json',dict(status='COMPLETE',rows=rows,GT_open_attempts=len(attempts),new_teacher_calls=0))
    print(json.dumps(rows),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['replay','temporal','protocol'],required=True)
    p.add_argument('--pair',choices=list(PAIRS),default='C4');p.add_argument('--fusion',choices=['DS','FIXED','MEAN'],default='DS')
    p.add_argument('--protocol',choices=['EXPLICIT_TASK','UNIFIED_CHECKLIST'],default='EXPLICIT_TASK')
    p.add_argument('--bank-routing',choices=['ORIGINAL','CROP']);a=p.parse_args()
    assert (OUT/'preregistration.json').is_file()
    if a.phase=='temporal':temporal(a.pair)
    elif a.phase=='protocol':protocol_audit()
    else:replay(a.pair,a.fusion,a.protocol,a.bank_routing)
