"""v8 isolated stream: real active questions or immutable MINUS_SLOW replay."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from scipy.special import expit
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json,append_json,file_sha
from flex_uda.presence.runtime import RUN,target_metadata,sealed_margins,runtime_guard
from flex_uda.active.v6 import ReliabilityV6
from flex_uda.active.v8 import BeliefStateV8,PresenceEM,questions_v8,fast_step_v8
from flex_uda.active.v7 import OnlinePresenceEB
from flex_uda.active.v7_frozen import FrozenFrameInputs
from flex_uda.active.acquisition import keyframe,exclusive_group,expression_types,round_budget
from scripts.active_run import PAIRS,video_spec


def live_round(state, possible, allowance, asked, tracks, fc, queries, target, video, local, teacher):
    result=[];used=set()
    # GLANCE -> RECHECK -> COVER -> ordinary, with at most
    # one question for a sentence is in flight, so absence really stops new calls.
    phases=[[q for q in possible if q.kind==kind] for kind in ('GLANCE','POSITIVE_RECHECK','PRESENCE_COVER')]
    phases.append([q for q in possible if q.kind not in ('GLANCE','POSITIVE_RECHECK','PRESENCE_COVER')])
    for pending in phases:
        while pending and len(result)<allowance:
            chosen=[];requests=[];deferred=[];inflight=set()
            for question in pending:
                q,ti=question.query,question.track
                if state.absent[q] or (q,ti) in used:continue
                if state.low_tentative[q] and question.kind=='IDENTITY':continue
                if q in inflight or len(chosen)>=min(8,allowance-len(result)):
                    deferred.append(question);continue
                obs=keyframe(tracks[fc.tracks[ti]],asked[q,ti],local['width'],local['height'])
                if obs is None:continue
                chosen.append(question);inflight.add(q);used.add((q,ti));asked[q,ti].add(obs['frame_id'])
                requests.append(dict(target=target,video_id=video,frame_id=obs['frame_id'],track_id=fc.tracks[ti],
                    query_id=queries[q]['expression_id'],query_text=queries[q]['query_text'],view=question.view))
            pending=deferred
            if not requests:break
            answers=teacher.ask_many(requests)
            for question,answer in zip(chosen,answers):
                result.append(ingest(state,question.query,question.track,question.kind,answer,question.information_gain))
    return result


def ingest(state,q,ti,kind,answer,gain,*,replay=False):
    view=answer['effective_view'];yes=answer['yes_probability']>=.5
    before=float(state.posterior[q,ti]);exist=float(state.presence_probability[q]);was_absent=bool(state.absent[q])
    tentative=bool(state.low_tentative[q])
    state.observe(q,ti,yes,view,state.presence_reliability[view],presence_question=kind in ('GLANCE','PRESENCE','PRESENCE_COVER'))
    return dict(answer,kind=kind,information_gain=gain,identity_vote=None,identity_used_as_DS_annotator=False,
        prior_stratum=int(state.strata[q,ti]),posterior_probability_before=before,
        posterior_probability_after=float(state.posterior[q,ti]),
        evidence_delta=state.events[-1]['delta'],unclipped_evidence_delta=state.events[-1]['unclipped_delta'],
        presence_probability_before=exist,presence_probability_after=float(state.presence_probability[q]),
        low_tentative_before=tentative,
        replay_fixed_answer=replay,replayed_after_counterfactual_absence=replay and was_absent)


def anchor_unupdated_replay(fc, frozen, directory):
    """A zero adapter uses the exact same frozen numerical inputs as MINUS."""
    margin_error=float(np.abs(np.asarray(fc.margin)-frozen.margin).max())
    presence_error=float(np.abs(fc.raw_presence-frozen.raw_presence).max())
    fc.margin[:]=frozen.margin;fc.margin.flush()
    fc.raw_presence=frozen.raw_presence.copy()
    fc.utilities=np.asarray(frozen.utilities,dtype=float).copy();fc.raw_k=np.asarray(frozen.raw_k).copy()
    atomic_json(directory/'zero_adapter_anchor.json',dict(
        status='FROZEN_NUMERICAL_INPUTS_ANCHORED_BEFORE_ANY_LORA_UPDATE',
        original_current_model_max_margin_error=margin_error,
        original_current_model_max_presence_logit_error=presence_error,
        same_canonical_inputs_as_MINUS=True,PCD_computation_retained=True,
        anchor_only_while_total_optimizer_updates_zero=True))


def run(pair,variant,pilot_videos=0):
    from flex_uda.active.execution import authorize_active
    from flex_uda.presence.predictions import seal_mask,verify_predictions
    execution,publication=authorize_active();job=dict(pair=pair,method=variant,rho=.3)
    assert execution.manifest['version']=='v8' and job in execution.manifest['allowed_registered_jobs']
    opened=runtime_guard();begin=time.perf_counter();model_name,target=PAIRS[pair]
    np.random.seed(0);torch.manual_seed(0)
    meta=target_metadata(target);margins,margin_seal=sealed_margins(target,model_name)
    out=execution.active/pair/variant/'rho_0.3';out.mkdir(parents=True,exist_ok=True)
    if (out/'predictions_seal.json').exists():
        print(json.dumps(dict(event='ALREADY_SEALED_NO_RERUN',path=str(out))),flush=True);return
    replay=variant=='SLOW_REPLAY';slow_enabled=variant!='MINUS_SLOW'
    dependency=execution.active/pair/'MINUS_SLOW/rho_0.3'
    replay_seal=verify_predictions(dependency) if replay else None
    runtime=out/'teacher_runtime.json';atomic_json(runtime,[video_spec(target,v,b) for v,b in meta['videos'].items()])
    teacher=None;model=tokenizer=transform=slow=None
    if slow_enabled:
        from flex_uda.presence.runtime import load_model,episode_from_bank,cache_for
        from flex_uda.active.v6_frames import FrameCacheV6
        from flex_uda.active.v8_slow import SlowAdapterV8
        device=torch.device('cuda:0')
        torch.cuda.set_per_process_memory_fraction(min(.9,(16<<30)/torch.cuda.get_device_properties(0).total_memory))
        model,tokenizer,transform,loaded=load_model(target,model_name,out/'model_log',device)
        atomic_json(out/'model_load.json',dict(loaded,version='v8',pair=pair,variant=variant))
    if not replay:
        from flex_uda.active.teacher_client import TeacherClient
        teacher=TeacherClient(runtime,RUN/'teacher/raw.sqlite',out/'teacher_server.log',ROOT)
    ordered=sorted(meta['videos']);selected=ordered[:pilot_videos] if pilot_videos else ordered
    ds=ReliabilityV6();em=PresenceEM();shadow=OnlinePresenceEB();w=1.;temperature=1.;bias=0.;saved=None;start=0
    checkpoints=sorted(out.glob('stream_*.pt'))
    if checkpoints:
        saved=torch.load(checkpoints[-1],map_location='cpu',weights_only=False)
        assert saved['version']=='v8'
        ds=saved['ds'];em=saved['em'];w=saved['slope'];temperature=saved['temperature'];bias=saved['bias'];start=saved['next_video']
        shadow=saved['shadow']
    try:
        for vi,video in enumerate(selected):
            if vi<start:continue
            bank=meta['videos'][video];queries=[q for q in meta['episodes'] if q['video_id']==video]
            texts=[q['query_text'] for q in queries];local=json.loads((RUN/'features'/target/model_name/'videos'/video/'metadata.json').read_text())
            directory=out/'videos'/video;directory.mkdir(parents=True,exist_ok=True)
            if replay:
                start_parameters=json.loads((dependency/'videos'/video/'em_video_start.json').read_text())
            else:
                start_parameters=em.receipt()
            atomic_json(directory/'em_video_start.json',start_parameters)
            if slow_enabled:
                episode=episode_from_bank(video,bank,target,transform);cache=cache_for(model,tokenizer,episode,device)
                if slow is None:
                    slow=SlowAdapterV8(model,cache)
                    if saved:
                        slow.adapter.load_state_dict(saved['adapter']);slow.optimizer.load_state_dict(saved['optimizer'])
                        slow.scaler.load_state_dict(saved['scaler']);slow.rng.set_state(saved['slow_rng'])
                        slow.updates=saved['slow_updates']
                        torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state(saved['cuda_rng'])
                else:slow.set_cache(cache)
                fc=FrameCacheV6(model,cache,bank,queries,local['frames'],execution.manifest['source_heads'][model_name]['checkpoint'],directory/'frame_cache')
                fc.score(model_step=slow.updates)
                if replay and slow.updates==0:
                    frozen=FrozenFrameInputs(execution.stage_a,target,model_name,video,bank,queries,margins)
                    anchor_unupdated_replay(fc,frozen,directory)
            else:fc=FrozenFrameInputs(execution.stage_a,target,model_name,video,bank,queries,margins)
            fractions,probability,_=fc.decode(temperature,bias);lengths=[len(fc.by_track[t]) for t in fc.tracks]
            state=BeliefStateV8(fc.track_means(),fractions,probability.max(1),track_lengths=lengths,
                rho_hat=start_parameters['rho_hat'],pi_strata=start_parameters['pi_strata'],slope=w)
            state.recompute_presence(ds.values)
            if replay:
                with np.load(dependency/'videos'/video/'presence_round0.npz') as snap:
                    state.freeze_presence_snapshot(snap)
            np.savez(directory/'presence_round0.npz',**state.presence_snapshot())
            tracks=defaultdict(list)
            for obs in video_spec(target,video,bank)['observations']:tracks[obs['track_id']].append(obs)
            budget=int(np.floor(.3*sum(n>=3 for n in lengths)*len(queries)))
            group=exclusive_group(texts,target);asked=defaultdict(set);answers_saved=[];rounds=[];carry=0;had_sample=False
            ever=np.zeros_like(state.asked_mask)
            before_teacher=dict(teacher.stats) if teacher else None
            source_answers=json.loads((dependency/'videos'/video/'questions.json').read_text()) if replay else None
            if replay:
                assert all(a['round'] in (1,2,3,4) for a in source_answers)
                replay_identity=dict(source_questions=str(dependency/'videos'/video/'questions.json'),
                    sha256=file_sha(dependency/'videos'/video/'questions.json'),source_prediction_seal_sha256=file_sha(dependency/'predictions_seal.json'))
                atomic_json(directory/'replay_source.json',replay_identity)
            tid={t:i for i,t in enumerate(fc.tracks)};qid={q['expression_id']:i for i,q in enumerate(queries)}
            for r in range(4):
                state.recompute_presence(ds.values);before=state.pseudo_labels()
                pool_histogram=np.histogram(state.posterior[:,~state.frozen_tracks],bins=np.linspace(0,1,21))[0].tolist()
                allowance=round_budget(budget,r)+carry
                if replay:
                    answers=[ingest(state,qid[a['query_id']],tid[a['track_id']],a['kind'],a,a['information_gain'],replay=True)
                             for a in source_answers if a['round']==r+1]
                else:
                    possible=questions_v8(state,texts,lengths,ds.values,r)
                    answers=live_round(state,possible,allowance,asked,tracks,fc,queries,target,video,local,teacher)
                for answer in answers:
                    q,ti=qid[answer['query_id']],tid[answer['track_id']]
                    ds.add((video,answer['query_id'],answer['track_id']),answer['effective_view'],answer['yes_probability']>=.5)
                    answers_saved.append(dict(answer,round=r+1))
                carry=allowance-len(answers);ds.fit();state.recompute_presence(ds.values)
                if replay:
                    with np.load(dependency/'videos'/video/f'presence_round{r+1}.npz') as snap:
                        state.freeze_presence_snapshot(snap)
                after_teacher=state.pseudo_labels().copy()
                fast=fast_step_v8(state,fc.raw_presence,temperature=temperature,bias=bias)
                temperature,bias=fast['temperature'],fast['bias'];fc.refresh(state,temperature,bias,retain_sampled=had_sample)
                slow_log=dict(status='DISABLED',optimizer_updates=0,GPU_seconds=0.,actual_optimized_positive_samples=0,
                    actual_optimized_negative_samples=0,actual_optimizer_steps=0,binary_positive_weight=1.,none_class_weight=1.)
                if slow:
                    slow_log=slow.slow_step(state,fc.tracks,texts,exclusive=group)
                    if slow_log['optimizer_updates']:
                        fc.score(sampled=True,model_step=slow.updates);had_sample=True;fc.refresh_sampled(state,temperature,bias)
                        after=fast_step_v8(state,fc.raw_presence,temperature=temperature,bias=bias)
                        temperature,bias=after['temperature'],after['bias'];fc.refresh(state,temperature,bias,retain_sampled=True)
                        fast['after_slow']=after
                labels=state.pseudo_labels();assert (labels[~state.asked_mask]==-1).all()
                if replay:
                    with np.load(dependency/'videos'/video/f'presence_round{r+1}.npz') as snap:
                        assert all(state.presence_snapshot()[k].tobytes()==snap[k].tobytes() for k in state.snapshot_fields)
                np.savez(directory/f'presence_round{r+1}.npz',**state.presence_snapshot())
                new_distinct=int(((labels>=0)&~ever).sum());ever|=labels>=0
                row=dict(version='v8',video=video,video_index=vi,round=r+1,allocated=allowance,actual_questions=len(answers),
                    presence_questions=sum(a['kind']=='PRESENCE' for a in answers),
                    identity_questions=sum(a['kind']=='IDENTITY' for a in answers),
                    positive_rechecks=sum(a['kind']=='POSITIVE_RECHECK' for a in answers),
                    glance_questions=sum(a['kind']=='GLANCE' for a in answers),
                    presence_cover_questions=sum(a['kind']=='PRESENCE_COVER' for a in answers),
                    question_counts={kind:sum(a['kind']==kind for a in answers) for kind in ('GLANCE','PRESENCE','PRESENCE_COVER','IDENTITY','POSITIVE_RECHECK')},
                    positive_pairs=int((labels==1).sum()),confirmed_pairs=int((labels>=0).sum()),
                    new_confirmed_pairs_teacher=int(((before<0)&(after_teacher>=0)).sum()),
                    new_confirmed_pairs_final=int(((before<0)&(labels>=0)).sum()),
                    new_distinct_confirmed_pairs=new_distinct,additional_distinct_confirmed_pairs=int(ever.sum()),
                    initial_confirmed_pairs=0,confident_pairs_never_asked=0,
                    absent_queries=int(state.absent.sum()),fast=fast,slow=slow_log,ds=ds.receipt())
                row['all_pair_posterior_before_round_histogram']=pool_histogram
                row['posterior_histogram_edges']=np.linspace(0,1,21).tolist()
                rounds.append(row);append_json(out/'rounds.jsonl',row)
                print(json.dumps({k:v for k,v in row.items() if k not in ('fast','slow','ds')}),flush=True)
            posterior=state.posterior;selected_tracks=posterior>.5
            selected_tracks[state.output_absence()]=False
            if group:selected_tracks &= np.arange(len(texts))[:,None]==posterior.argmax(axis=0)[None]
            mask=np.zeros((len(queries),len(bank['observations'])),dtype=bool)
            calibrated=expit(fc.raw_presence/temperature+bias)
            for q,text in enumerate(texts):
                motion='MOTION' in expression_types(text)
                for j,(frame,track) in enumerate(bank['observations']):
                    mask[q,j]=selected_tracks[q,tid[track]] and (not motion or calibrated[q,fc.frame_index[frame]]>=.5)
            np.save(directory/'mask.npy',mask)
            np.savez(directory/'belief.npz',prior=state.prior,evidence=state.evidence,posterior=posterior,
                absent=state.absent,confirmed_present=state.confirmed_present,low_presence=state.low,
                presence_max=state.presence_max,intercept=state.a,slope=state.slope,temperature=temperature,bias=bias,
                expected_counts=state.expected,
                tracks=np.asarray(fc.tracks),queries=np.asarray([q['expression_id'] for q in queries]),
                fractions=state.fractions,base_logits=state.base_logits,pseudo_labels=state.pseudo_labels(),
                asked_mask=state.asked_mask,prior_strata=state.strata,presence_probability=state.presence_probability,
                presence_probability_recomputed=state.presence_probability_recomputed,
                weak_presence_logits=state.weak_presence_logits,stop_probability=state.stop_probability,
                presence_logLR_unclipped_sum=state.presence_logLR_unclipped_sum,
                presence_logLR_clipped_sum=state.presence_logLR_clipped_sum,presence_logLR_clip_count=state.presence_logLR_clip_count,
                low_tentative=state.low_tentative,covered_strata=state.covered_strata,needed_strata=state.needed_strata,
                output_absent=state.output_absence(),presence_likelihood_evaluations=state.presence_likelihood_evaluations)
            before_em=em.receipt()
            if replay:
                eb_record=dict(before=start_parameters,after=start_parameters,replay_frozen=True)
            else:
                em.add_video(video,state)
                eb_record=dict(before=before_em,after=em.fit(ds.values),replay_frozen=False)
                if not em.converged:
                    atomic_json(directory/'em_nonconvergence.json',eb_record)
                    raise RuntimeError('Registered stop: v8 EM did not converge; preserve completed videos')
            # Shadow estimator is diagnostic only; its values never enter v8.
            state.pi_start=shadow.pi
            eb_record['v7_shadow']=shadow.finish_video(state,ds.values)
            eb_record['v7_shadow_definition']='Original v7 outcome-filtered update formula on current v8 decisions and stratified answer channels; not a rerun of the historical v7 schedule'
            per_query=[dict(query=q['expression_id'],decision_probability=float(state.presence_probability[i]),
                recomputed_probability=float(state.presence_probability_recomputed[i]),absent=bool(state.absent[i]),
                present=bool(state.confirmed_present[i]),weak_rank_logit=float(state.weak_presence_logits[i]),
                logLR_unclipped_sum=float(state.presence_logLR_unclipped_sum[i]),
                logLR_clipped_sum=float(state.presence_logLR_clipped_sum[i]),
                clipped_answers=int(state.presence_logLR_clip_count[i]),
                presence_answers=len(state.presence_by_query[i]),low_tentative=bool(state.low_tentative[i]),
                decision_source='EVIDENCE' if state.absent[i] else ('PRIOR_BUDGET_EXHAUSTED' if state.output_absence()[i] else 'NOT_REJECTED'),
                covered_strata=np.flatnonzero(state.covered_strata[i]).tolist()) for i,q in enumerate(queries)]
            atomic_json(directory/'presence.json',dict(eb_record,queries=per_query))
            append_json(out/'eb_stream.jsonl',dict(video=video,video_index=vi,**eb_record))
            atomic_json(directory/'questions.json',answers_saved)
            if slow_enabled:fc.close()
            atomic_json(directory/'receipt.json',dict(version='v8',video=video,budget=budget,questions=len(answers_saved),
                unspent_budget=budget-len(answers_saved),exclusive_group=group,selected_rows=int(mask.sum()),
                mask_sha256=file_sha(directory/'mask.npy'),rounds=rounds,
                teacher_delta={k:teacher.stats[k]-before_teacher[k] for k in before_teacher} if teacher else None,
                slow_GPU_seconds=sum(rr['slow']['GPU_seconds'] for rr in rounds),
                rescore_GPU_seconds=sum(x['GPU_seconds'] for x in fc.score_records),
                fixed32_frames=fc.sample_frames if slow_enabled else [],decode_CPU_seconds=fc.decode_seconds,
                glance_unanswered_queries=int((~state.asked_mask.any(1)).sum()),
                GLANCE_BUDGET_SHORTFALL=budget<len(queries) and bool((~state.asked_mask.any(1)).any()),
                replay_presence_byte_exact=bool(replay),
                EB=eb_record,replay_fixed_answers=replay,source_margin_sha256=margin_seal['sha256']))
            w=state.slope
            saved=dict(version='v8',next_video=vi+1,ds=ds,em=em,shadow=shadow,slope=w,temperature=temperature,bias=bias,
                adapter=slow.adapter.state_dict() if slow else None,optimizer=slow.optimizer.state_dict() if slow else None,
                scaler=slow.scaler.state_dict() if slow else None,slow_rng=slow.rng.get_state() if slow else None,
                torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state() if slow else None)
            saved['slow_updates']=slow.updates if slow else 0
            path=out/f'stream_{vi:03d}.pt';tmp=path.with_suffix('.partial');torch.save(saved,tmp);tmp.replace(path)
            del fc
        if teacher:teacher_record=teacher.close();teacher=None
        else:teacher_record=dict(replay_no_teacher_invocation=True,source=str(dependency),
                source_seal_sha256=file_sha(dependency/'predictions_seal.json'),**replay_seal['teacher'])
        receipts=[json.loads((out/'videos'/v/'receipt.json').read_text()) for v in selected]
        if not replay:
            sessions=[json.loads(p.read_text()) for p in (out/'teacher_sessions').glob('*.json')]
            teacher_record.update(logical_calls=sum(v['questions'] for v in receipts),
                **{k:sum(s[k] for s in sessions) for k in ('new_calls','cache_hits','forward_seconds','equivalent_forward_seconds')})
        else:
            teacher_record.update(inherited_teacher_calls=replay_seal['teacher']['logical_calls'],
                logical_calls=sum(v['questions'] for v in receipts),new_calls=0,cache_hits=0,forward_seconds=0.,
                replayed_answers=sum(v['questions'] for v in receipts))
        assert not opened
        if pilot_videos and len(selected)<len(ordered):
            atomic_json(out/'pilot_receipt.json',dict(status='V8_FIRST_VIDEOS_RESUMABLE',videos=selected,teacher=teacher_record,
                GT_open_attempts=0,pair=pair,variant=variant,next_video=len(selected),
                em=em.receipt(),snapshot_count=len(selected)*5,
                GLANCE_BUDGET_SHORTFALL_videos=sum(v['GLANCE_BUDGET_SHORTFALL'] for v in receipts),
                unasked_queries=sum(v['glance_unanswered_queries'] for v in receipts)))
            return
        mask=np.zeros(len(margins),dtype=bool)
        for video in ordered:
            local_mask=np.load(out/'videos'/video/'mask.npy');qrows=[q for q in meta['episodes'] if q['video_id']==video]
            for qi,q in enumerate(qrows):mask[q['start']:q['stop']]=local_mask[qi]
        seal_mask(out,meta,mask,target=target,model=model_name,method=variant,GT_open_attempts=0,
            extra=dict(version='v8',pair=pair,rho=.3,teacher=teacher_record,margin_input=margin_seal,
                LoRA_GPU_seconds=sum(v['slow_GPU_seconds'] for v in receipts),
                rescore_GPU_seconds=sum(v['rescore_GPU_seconds'] for v in receipts),
                video_transductive=True,strict_frame_causal=False,ordered_video_stream=True,
                wall_seconds=time.perf_counter()-begin,stage_A_publication=publication,
                execution_manifest_sha256=file_sha(execution.manifest_path),LoRA_enabled=slow_enabled))
    finally:
        if teacher is not None and teacher.child.poll() is None:teacher.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--pair',choices=list(PAIRS),required=True)
    p.add_argument('--variant',choices=['FULL','MINUS_SLOW','SLOW_REPLAY'],required=True)
    p.add_argument('--ablation',choices=['MINUS_EM','MINUS_GLANCE','V7_ESTIMATOR','RANDOM_ACQ','MINUS_DS','MINUS_ROUTE','MINUS_PRES'])
    p.add_argument('--pilot-videos',type=int,default=0);a=p.parse_args()
    from flex_uda.active.execution import active_paths
    from flex_uda.active.receipts import record_attempt
    if a.ablation:
        raise PermissionError('Stage4 implementation/dispatch requires the completed registered main gate; no ablation launched here')
    folder=active_paths().active/a.pair/a.variant/'rho_0.3'
    with record_attempt(folder,phase='v8_full_stream'):run(a.pair,a.variant,a.pilot_videos)
