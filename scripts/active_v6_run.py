"""v6 current-model frame-cache stream; invoked by the original isolated runner."""
from collections import defaultdict
import json
import time
import numpy as np
import torch
from scipy.special import expit
from flex_uda.active.v6 import BeliefStateV6,ReliabilityV6,questions_v6,fast_step_v6
from flex_uda.active.v6_frames import FrameCacheV6
from flex_uda.active.acquisition import keyframe,exclusive_group,expression_types,round_budget
from flex_uda.active.adapt import SlowAdapter
from flex_uda.active.baselines import video_rng
from flex_uda.active.teacher_client import TeacherClient
from flex_uda.presence.runtime import RUN,target_metadata,sealed_margins,runtime_guard,load_model,episode_from_bank,cache_for
from flex_uda.presence.predictions import seal_mask
from flex_tta.runtime.pf_receipts import atomic_json,append_json,file_sha


def run_v6(pair,rho,variant,*,smoke_videos=0,pilot_videos=0):
    from scripts.active_run import ROOT,PAIRS,video_spec
    from flex_uda.active.execution import authorize_active
    execution,publication=authorize_active()
    if variant not in ('FULL','MINUS_SLOW') or rho!=.3 or smoke_videos:
        raise PermissionError('v6 only fixed rho.3 main/slow critical jobs; use resumable first-video pilot')
    job=dict(pair=pair,method=variant,rho=rho)
    if job not in execution.manifest['allowed_registered_jobs']:
        raise PermissionError('v6 budget gate did not authorize this direction')
    opened=runtime_guard();begin=time.perf_counter();model_name,target=PAIRS[pair]
    torch.manual_seed(0);np.random.seed(0)
    device=torch.device('cuda:0')
    torch.cuda.set_per_process_memory_fraction(min(.9,(16<<30)/torch.cuda.get_device_properties(0).total_memory))
    meta=target_metadata(target);_,margin_seal=sealed_margins(target,model_name)
    out=execution.active/pair/variant/('rho_'+str(rho));out.mkdir(parents=True,exist_ok=True)
    if (out/'predictions_seal.json').exists():
        print(json.dumps(dict(event='ALREADY_SEALED_NO_RERUN',folder=str(out))),flush=True);return
    runtime=out/'teacher_runtime.json'
    atomic_json(runtime,[video_spec(target,v,b) for v,b in meta['videos'].items()])
    model,tokenizer,transform,load=load_model(target,model_name,out/'model_log',device)
    teacher=TeacherClient(runtime,RUN/'teacher/raw.sqlite',out/'teacher_server.log',ROOT)
    ordered=sorted(meta['videos']);selected=ordered[:pilot_videos] if pilot_videos else ordered
    ds=ReliabilityV6();w=1.;temperature=1.;bias=0.;slow=None;saved=None;start=0
    checkpoints=sorted(out.glob('stream_*.pt'))
    if checkpoints:
        saved=torch.load(checkpoints[-1],map_location='cpu',weights_only=False)
        if saved.get('version')!='v6':raise ValueError('Cannot initialize v6 from a v5 stream')
        ds=saved['ds'];w=saved['slope'];temperature=saved['temperature'];bias=saved['bias'];start=saved['next_video']
    try:
        for vi,video in enumerate(selected):
            if vi<start:continue
            bank=meta['videos'][video];queries=[q for q in meta['episodes'] if q['video_id']==video]
            texts=[q['query_text'] for q in queries]
            episode=episode_from_bank(video,bank,target,transform);cache=cache_for(model,tokenizer,episode,device)
            if variant=='FULL':
                if slow is None:
                    slow=SlowAdapter(model,cache)
                    if saved:
                        slow.adapter.load_state_dict(saved['adapter']);slow.optimizer.load_state_dict(saved['optimizer'])
                        slow.scaler.load_state_dict(saved['scaler']);slow.rng.set_state(saved['slow_rng'])
                        torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state(saved['cuda_rng'])
                else:slow.set_cache(cache)
            local=json.loads((RUN/'features'/target/model_name/'videos'/video/'metadata.json').read_text())
            directory=out/'videos'/video;directory.mkdir(parents=True,exist_ok=True)
            fc=FrameCacheV6(model,cache,bank,queries,local['frames'],
                execution.manifest['source_heads'][model_name]['checkpoint'],directory/'frame_cache')
            fc.score(model_step=slow.updates if slow else 0)
            fractions,prob,_=fc.decode(temperature,bias)
            lengths=[len(fc.by_track[t]) for t in fc.tracks]
            state=BeliefStateV6(fc.track_means(),fractions,prob.max(1),track_lengths=lengths,slope=w)
            tracks=defaultdict(list)
            for obs in video_spec(target,video,bank)['observations']:tracks[obs['track_id']].append(obs)
            group=exclusive_group(texts,target);budget=int(np.floor(rho*sum(n>=3 for n in lengths)*len(queries)))
            asked=defaultdict(set);answers_saved=[];rounds=[];ever=np.zeros_like(state.asked_mask)
            before_teacher=dict(teacher.stats);carry=0;had_sample=False
            for r in range(4):
                values=ds.values.copy();possible=questions_v6(state,texts,lengths,values,r)
                allowance=round_budget(budget,r)+carry;requests=[];chosen=[];used=set()
                for question in possible if allowance else []:
                    q,ti=question.query,question.track
                    if (q,ti) in used:continue
                    obs=keyframe(tracks[fc.tracks[ti]],asked[q,ti],local['width'],local['height'])
                    if obs is None:continue
                    used.add((q,ti));asked[q,ti].add(obs['frame_id']);chosen.append(question)
                    requests.append(dict(target=target,video_id=video,frame_id=obs['frame_id'],track_id=fc.tracks[ti],
                        query_id=queries[q]['expression_id'],query_text=texts[q],view=question.view))
                    if len(chosen)>=allowance:break
                before=state.pseudo_labels();answers=teacher.ask_many(requests) if requests else []
                for question,answer in zip(chosen,answers):
                    q,ti=question.query,question.track;view=answer['effective_view'];yes=answer['yes_probability']>=.5
                    probability_before=float(state.posterior[q,ti])
                    state.observe(q,ti,yes,view,values[view],presence_question=question.kind=='PRESENCE')
                    ds.add((video,queries[q]['expression_id'],fc.tracks[ti]),view,yes)
                    answers_saved.append(dict(answer,round=r+1,kind=question.kind,information_gain=question.information_gain,
                        identity_vote=None,identity_used_as_DS_annotator=False,
                        prior_stratum=int(state.strata[q,ti]),posterior_probability_before=probability_before,
                        posterior_probability_after=float(state.posterior[q,ti]),
                        evidence_delta=state.events[-1]['delta'],unclipped_evidence_delta=state.events[-1]['unclipped_delta']))
                carry=allowance-len(answers);ds.fit();after=state.pseudo_labels()
                fast=fast_step_v6(state,fc.raw_presence,temperature=temperature,bias=bias)
                temperature,bias=fast['temperature'],fast['bias']
                fc.refresh(state,temperature,bias,retain_sampled=had_sample)
                slow_log=dict(status='DISABLED',optimizer_updates=0,GPU_seconds=0.)
                if slow is not None:
                    slow_log=slow.slow_step(state,fc.tracks,texts,exclusive=group)
                    if slow_log['optimizer_updates']:
                        fc.score(sampled=True,model_step=slow.updates);had_sample=True
                        fc.refresh_sampled(state,temperature,bias)
                        after_slow=fast_step_v6(state,fc.raw_presence,temperature=temperature,bias=bias)
                        temperature,bias=after_slow['temperature'],after_slow['bias']
                        fc.refresh(state,temperature,bias,retain_sampled=True);fast['after_slow']=after_slow
                labels=state.pseudo_labels();new=int(((labels>=0)&~ever).sum());ever|=labels>=0
                assert not ((labels>=0)&~state.asked_mask).any()
                row=dict(version='v6',video=video,video_index=vi,round=r+1,allocated=allowance,
                    actual_questions=len(answers),presence_questions=sum(q.kind=='PRESENCE' for q in chosen),
                    identity_questions=sum(q.kind=='IDENTITY' for q in chosen),
                    new_confirmed_pairs_teacher=int(((before<0)&(after>=0)).sum()),
                    new_confirmed_pairs_final=int(((before<0)&(labels>=0)).sum()),new_distinct_confirmed_pairs=new,
                    initial_confirmed_pairs=0,additional_distinct_confirmed_pairs=int(ever.sum()),
                    confident_pairs_never_asked=0,confirmed_pairs=int((labels>=0).sum()),positive_pairs=int((labels==1).sum()),
                    absent_queries=int(state.absent.sum()),fast=fast,slow=slow_log,ds=ds.receipt())
                rounds.append(row);append_json(out/'rounds.jsonl',row)
                print(json.dumps({k:v for k,v in row.items() if k not in ('fast','slow','ds')}),flush=True)
            posterior=state.posterior;selected_tracks=posterior>.5
            if group:selected_tracks&=np.arange(len(texts))[:,None]==posterior.argmax(axis=0)[None]
            mask=np.zeros((len(queries),len(bank['observations'])),dtype=bool)
            calibrated_presence=expit(fc.raw_presence/temperature+bias)
            tid={t:i for i,t in enumerate(fc.tracks)}
            for q,text in enumerate(texts):
                motion='MOTION' in expression_types(text)
                for j,(frame,track) in enumerate(bank['observations']):
                    mask[q,j]=selected_tracks[q,tid[track]] and (not motion or calibrated_presence[q,fc.frame_index[frame]]>=.5)
            np.save(directory/'mask.npy',mask)
            np.savez(directory/'belief.npz',prior=state.prior,evidence=state.evidence,posterior=posterior,
                absent=state.absent,confirmed_present=state.confirmed_present,low_presence=state.low,
                expected_counts=state.expected,presence_max=state.presence_max,intercept=state.a,slope=state.slope,
                temperature=temperature,bias=bias,tracks=np.asarray(fc.tracks),queries=np.asarray([q['expression_id'] for q in queries]),
                fractions=state.fractions,base_logits=state.base_logits,pseudo_labels=state.pseudo_labels(),
                asked_mask=state.asked_mask,prior_strata=state.strata)
            atomic_json(directory/'questions.json',answers_saved);cache_record=fc.close()
            score_seconds=sum(x['GPU_seconds'] for x in fc.score_records)
            atomic_json(directory/'receipt.json',dict(version='v6',video=video,budget=budget,questions=len(answers_saved),
                unspent_budget=budget-len(answers_saved),exclusive_group=group,short_tracks=sum(n<3 for n in lengths),
                initial_confident_pairs=0,selected_rows=int(mask.sum()),mask_sha256=file_sha(directory/'mask.npy'),rounds=rounds,
                teacher_delta={k:teacher.stats[k]-before_teacher.get(k,0) for k in
                    ('logical_calls','new_calls','cache_hits','forward_seconds','equivalent_forward_seconds')},
                slow_GPU_seconds=sum(x['slow']['GPU_seconds'] for x in rounds),rescore_GPU_seconds=score_seconds,
                rescore_candidate_query_rows=sum(x['candidate_query_rows'] for x in fc.score_records),
                frame_cache_receipt=str(directory/'frame_cache/cache_receipt.json'),decode_CPU_seconds=fc.decode_seconds,
                fixed32_frames=fc.sample_frames))
            w=state.slope
            saved=dict(version='v6',next_video=vi+1,ds=ds,slope=w,temperature=temperature,bias=bias,
                adapter=slow.adapter.state_dict() if slow else None,optimizer=slow.optimizer.state_dict() if slow else None,
                scaler=slow.scaler.state_dict() if slow else None,slow_rng=slow.rng.get_state() if slow else None,
                torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state())
            path=out/f'stream_{vi:03d}.pt';temporary=path.with_suffix('.partial')
            torch.save(saved,temporary);temporary.replace(path)
            del fc,cache
        teacher_record=teacher.close();teacher=None
        receipts=[json.loads((out/'videos'/v/'receipt.json').read_text()) for v in selected]
        sessions=[json.loads(p.read_text()) for p in (out/'teacher_sessions').glob('*.json')]
        teacher_record.update(logical_calls=sum(v['questions'] for v in receipts),
            attempted_logical_calls=sum(s['logical_calls'] for s in sessions),
            **{k:sum(s[k] for s in sessions) for k in
               ('new_calls','cache_hits','forward_seconds','equivalent_forward_seconds')})
        if pilot_videos and len(selected)<len(ordered):
            atomic_json(out/'pilot_receipt.json',dict(status='V6_REAL_PILOT_RESUMABLE',videos=selected,
                teacher=teacher_record,GT_open_attempts=len(opened),version='v6'))
            return
        size=max(q['stop'] for q in meta['episodes']);mask=np.zeros(size,dtype=bool)
        for video in ordered:
            local_mask=np.load(out/'videos'/video/'mask.npy')
            queries=[q for q in meta['episodes'] if q['video_id']==video]
            for qi,q in enumerate(queries):mask[q['start']:q['stop']]=local_mask[qi]
        seal_mask(out,meta,mask,target=target,model=model_name,method=variant,GT_open_attempts=len(opened),
            extra=dict(version='v6',pair=pair,rho=rho,teacher=teacher_record,
                LoRA_GPU_seconds=sum(v['slow_GPU_seconds'] for v in receipts),
                rescore_GPU_seconds=sum(v['rescore_GPU_seconds'] for v in receipts),margin_input=margin_seal,
                video_transductive=True,strict_frame_causal=False,ordered_video_stream=True,
                wall_seconds=time.perf_counter()-begin,stage_A_publication=publication,
                execution_manifest_sha256=file_sha(execution.manifest_path)))
    finally:
        if teacher is not None and teacher.child.poll() is None:teacher.close()
