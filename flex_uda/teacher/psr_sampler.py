"""GT-free paired temporal sampling with a fixed, auditable teacher budget."""
from collections import Counter
import hashlib
import random
from .mllm_pseudo_dataset import digest

ROLES=('SOURCE_HIGH','SOURCE_UNCERTAIN','SOURCE_LOW_VISIBLE','TRAJECTORY_COVERAGE','UNIFORM_EXPLORATION')
MODES=('NAIVE','UNIFORM','CONFIDENCE','PSR')


def temporal_kind(text):
    words=set(text.lower().replace('-',' ').split())
    return 'DYNAMIC_OR_MIXED' if words&{'dancing','dacning','standing','walking','running','jumping','sitting','moving'} else 'STATIC_OR_UNVERIFIED'


def choose_tracks(ids,margins,coverage,mode,roles,rng):
    if mode not in MODES:raise ValueError('unregistered discovery policy')
    remaining=set(ids);chosen=[]
    for slot,role in enumerate(roles):
        if not remaining:break
        order=sorted(remaining)
        if mode=='UNIFORM' or mode=='NAIVE' and slot==1 or mode=='PSR' and role=='UNIFORM_EXPLORATION':
            track=rng.choice(order);reason='UNIFORM_EXPLORATION'
        elif mode in ('NAIVE','CONFIDENCE') or role=='SOURCE_HIGH':
            track=min(order,key=lambda i:(-margins[i],i));reason='SOURCE_HIGH'
        elif role=='SOURCE_UNCERTAIN':track=min(order,key=lambda i:(abs(margins[i]),i));reason=role
        elif role=='SOURCE_LOW_VISIBLE':track=min(order,key=lambda i:(margins[i],i));reason=role
        else:track=min(order,key=lambda i:(coverage[i],abs(margins[i]),i));reason='TRAJECTORY_COVERAGE'
        remaining.remove(track);chosen.append((track,reason,role));coverage[track]+=1
    return chosen


def sample_episode(runtime,observations,source_scores,mode='PSR',seed=17,quantiles=(.1,.3,.5,.7,.9)):
    """observations[frame][predictedID] contains xyxy and unmodified MOT line.

    source_scores[(frame,ID)] are frozen V2 margins; no membership/GT input.
    Return five training groups even when no eligible candidate exists.
    """
    video=runtime['video_id'];query=runtime['expression_id'];frames=sorted(observations)
    start=runtime['frame_indices']['start'];last=runtime['frame_indices']['stop']-1
    anchors=sorted({start+round((last-start)*q) for q in quantiles})
    rng=random.Random(int(hashlib.sha256(f'{seed}/{video}/{query}'.encode()).hexdigest()[:16],16))
    coverage=Counter();samples=[];windows=[];selection=[]
    for index,anchor in enumerate(anchors):
        times=(anchor-1,anchor,anchor+1)
        eligible=set.intersection(*(set(observations.get(t,{})) for t in times))
        eligible={i for i in eligible if all(observations[t][i]['bbox_xyxy'][2]>observations[t][i]['bbox_xyxy'][0] and
            observations[t][i]['bbox_xyxy'][3]>observations[t][i]['bbox_xyxy'][1] for t in times)}
        margins={i:float(source_scores[anchor,i]) for i in eligible}
        roles=[ROLES[(2*index+j)%5] for j in range(2)]
        chosen=choose_tracks(eligible,margins,coverage,mode,roles,rng)
        window_id=digest((video,query,anchor))[:24]
        for track,reason,role in chosen:
            for t in (anchor-1,anchor+1):
                obs=observations[t][track]
                sample=dict(video_id=video,query_id=query,query_text=runtime['query_text'],frame_id=t,track_id=track,
                    bbox_xyxy=list(obs['bbox_xyxy']),raw_mot_line=obs['raw_mot_line'],frozen_margin=float(source_scores[t,track]),
                    anchor_frame=anchor,window_id=window_id,selection_reasons=[reason],budget_role=role,
                    expression_type=temporal_kind(runtime['query_text']),history=[],attributes=[],parser_notes=[])
                sample['sample_id']=digest((video,query,t,track))[:24];samples.append(sample)
        windows.append(dict(video_id=video,query_id=query,frame_id=anchor,window_id=window_id,sampled_candidates=len(chosen)*2))
        selection.append(dict(video_id=video,query_id=query,anchor=anchor,eligible_tracks=len(eligible),
            selected_tracks=[i for i,_,_ in chosen],logical_teacher_calls=2*len(chosen),empty_reason=None if chosen else 'NO_CONTIGUOUS_PREDICTED_CANDIDATE_NOT_A_NEGATIVE'))
    if len({s['sample_id'] for s in samples})!=len(samples):raise ValueError('overlapping anchors require deduplicated budget registration')
    return samples,windows,selection
