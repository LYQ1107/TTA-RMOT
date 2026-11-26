"""Nominal temporal input support, not semantic accuracy or encoded content.

Frozen PCD already receives four track observations; the newer input also
has trajectory velocities. Qwen CLIP uses four requested stride offsets.
Audit their exact metadata support before proposing a temporal module.
No images, GT, teacher answers, neural forwards, training or random seeds.
"""
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from scripts.a100_fullscope_visual_six import PLAN,registration
from scripts.a100_dino_appearance_collect import ANCHORS


def physical(target,video,bank):
    from scripts.active_run import video_spec
    obs=np.asarray(bank['observations'],dtype=int)
    assert len({tuple(r) for r in obs})==len(obs)
    delta=video_spec(target,video,bank)['delta'];assert delta>0
    with np.load(ANCHORS/target/video/'anchor_histograms.npz') as a:
        tracks=a['tracks'].copy();anchors=a['observations'].copy();present=a['present'].copy()
    index={int(t):i for i,t in enumerate(tracks)}
    arrays={k:np.zeros(len(obs),dtype=int) for k in ['teacher_usable_crops','teacher_unique_frames',
        'teacher_frames_missing_from_current_PCD','teacher_frames_missing_from_current_and_anchor_PCD',
        'current_PCD_span','teacher_clip_span','student_nominal_visual_has_future_anchor']}
    for track in np.unique(obs[:,1]):
        rows=np.flatnonzero(obs[:,1]==track);times=obs[rows,0]
        assert np.all(np.diff(times)>0)
        positions=np.arange(len(rows))
        current_history=times[np.maximum(positions[:,None]-np.asarray([3,2,1,0]),0)]
        anchor_rows=anchors[index[int(track)]][present[index[int(track)]]]
        anchor_positions=np.searchsorted(rows,anchor_rows)
        assert np.array_equal(rows[anchor_positions],anchor_rows) and (obs[anchor_rows,1]==track).all()
        anchor_history=np.concatenate([times[max(0,int(p)-3):int(p)+1] for p in anchor_positions])
        nominal_support=np.column_stack((current_history,np.broadcast_to(anchor_history,(len(rows),len(anchor_history)))))
        selected=[]
        for offset in (3*delta,2*delta,delta,0):
            previous=np.searchsorted(times,times-offset,side='right')-1
            selected.append(np.where(previous>=0,times[np.maximum(previous,0)],-1))
        clip=np.column_stack(selected);valid=clip>=0
        unique=valid.copy();unique[:,1:]&=(clip[:,1:]!=clip[:,:-1])
        in_current=(clip[:,:,None]==current_history[:,None,:]).any(-1)
        in_nominal=(clip[:,:,None]==nominal_support[:,None,:]).any(-1)
        arrays['teacher_usable_crops'][rows]=valid.sum(1)
        arrays['teacher_unique_frames'][rows]=unique.sum(1)
        arrays['teacher_frames_missing_from_current_PCD'][rows]=(unique&~in_current).sum(1)
        arrays['teacher_frames_missing_from_current_and_anchor_PCD'][rows]=(unique&~in_nominal).sum(1)
        arrays['current_PCD_span'][rows]=times-current_history[:,0]
        arrays['teacher_clip_span'][rows]=times-np.where(valid,clip,np.iinfo(np.int64).max).min(1)
        arrays['student_nominal_visual_has_future_anchor'][rows]=(nominal_support>times[:,None]).any(1)
    usable=arrays['teacher_usable_crops']>=2
    assert (arrays['teacher_unique_frames']>=1).all() and np.array_equal(usable,arrays['teacher_unique_frames']>=2)
    report=dict(target=target,video=video,physical_observations=len(obs),teacher_delta_frames=int(delta),
        potential_CLIP_usable_observations=int(usable.sum()),potential_CLIP_CROP_fallback_observations=int((~usable).sum()),
        usable_CLIP_all_frames_in_current_PCD=int((usable&(arrays['teacher_frames_missing_from_current_PCD']==0)).sum()),
        usable_CLIP_all_frames_in_current_and_anchor_PCD=int((usable&(arrays['teacher_frames_missing_from_current_and_anchor_PCD']==0)).sum()),
        usable_CLIP_unique_frame_instances=int(arrays['teacher_unique_frames'][usable].sum()),
        usable_CLIP_frame_instances_missing_from_current_PCD=int(arrays['teacher_frames_missing_from_current_PCD'][usable].sum()),
        usable_CLIP_frame_instances_missing_from_current_and_anchor_PCD=int(arrays['teacher_frames_missing_from_current_and_anchor_PCD'][usable].sum()),
        current_PCD_span_quantiles=np.quantile(arrays['current_PCD_span'],[0,.5,.9,1]).tolist(),
        teacher_CLIP_span_quantiles=np.quantile(arrays['teacher_clip_span'],[0,.5,.9,1]).tolist(),
        student_with_nominal_future_anchors_observations=int(arrays['student_nominal_visual_has_future_anchor'].sum()),
        source_PCD_temporal_input_available=True,source_PCD_history_is_last4observations_with_start_repetition=True,
        current_input_also_has_stride_trajectory_velocities=True,not_a_measure_of_encoded_semantic_information=True)
    return report


def main():
    from flex_uda.a100.mechanism import generation_guard,metadata
    started=time.perf_counter();guard=generation_guard();plan=registration()
    meta={target:metadata(target) for target in {c['target'] for c in plan['cases']}}
    cache={};cases=[]
    for case in plan['cases']:
        key=case['target'],case['evaluated_video']
        if key not in cache:
            cache[key]=physical(*key,meta[key[0]]['videos'][key[1]])
        p=cache[key];obs=np.load(case['observations'])
        assert np.array_equal(obs,np.asarray(meta[key[0]]['videos'][key[1]]['observations']))
        document=json.loads(Path(case['temporal_queries']).read_text())
        assert file_sha(Path(case['temporal_queries']))==case['temporal_queries_sha256']
        clip=[q for q in document['queries'] if q['view']=='CLIP'];n=len(clip)
        row=dict(pair=case['pair'],video=key[1],queries=len(document['queries']),CLIP_queries=n,
            planned_logical_budget_on_CLIP_queries=sum(q['logical_budget'] for q in clip),
            full_possible_requested_CLIP_actions=n*p['physical_observations'])
        for field in ['potential_CLIP_usable_observations','potential_CLIP_CROP_fallback_observations',
            'usable_CLIP_all_frames_in_current_PCD','usable_CLIP_all_frames_in_current_and_anchor_PCD',
            'usable_CLIP_unique_frame_instances','usable_CLIP_frame_instances_missing_from_current_PCD',
            'usable_CLIP_frame_instances_missing_from_current_and_anchor_PCD']:
            row['query_action_weighted_'+field]=n*p[field]
        cases.append(row)
    assert len(cache)==74 and len(cases)==148 and not guard
    settings=[]
    for pair in sorted({r['pair'] for r in cases}):
        selected=[r for r in cases if r['pair']==pair]
        row=dict(pair=pair,videos=len(selected))
        for key in selected[0]:
            if key not in ('pair','video'):row[key]=sum(r[key] for r in selected)
        denominator=row['query_action_weighted_potential_CLIP_usable_observations']
        row['usable_CLIP_action_fraction_fully_in_current_PCD']=None if not denominator else row['query_action_weighted_usable_CLIP_all_frames_in_current_PCD']/denominator
        row['usable_CLIP_action_fraction_fully_in_current_and_anchor_PCD']=None if not denominator else row['query_action_weighted_usable_CLIP_all_frames_in_current_and_anchor_PCD']/denominator
        settings.append(row)
    receipt=dict(status='COMPLETE74PHYSICAL148SETTING_TEMPORAL_INPUT_SUPPORT_AUDIT',
        physical_videos=74,setting_video_units=148,physical_observations=sum(p['physical_observations'] for p in cache.values()),
        physical=list(cache.values()),cases=cases,settings=settings,
        source_sha256=file_sha(__file__),parent_registration_sha256=file_sha(PLAN),
        source_code=dict(PCD_export='scripts/presence_export_target.py:export_video -> episode.current_candidates/history -> prepared_inputs -> margins_and_features',
            PCD_history='flex_tta/runtime/dance_episode.py:history length4, contiguous observed track rows, start padded',
            teacher_CLIP='flex_uda/teacher/decompose_inputs.py:PairedViews.clip t-3delta,t-2delta,t-delta,t via last available observation',
            added_motion='flex_uda/active/joint_observation_inputs.py:trajectory_features, stride displacement and frame-relative residuals',
            static_anchors='Three track-wide anchor PCD/DINO inputs may occur after current t in this permitted full-video transductive study.'),
        interpretation='Nominal input support only. Distinct timestamps do not prove needed semantics are missing or retained in PCD. These counts concern the full possible action universe, not actual paid question rates or GT error rates. Geometry velocities and query-conditioned learned temporal features remain available. No new temporal model is justified by these counts alone.',
        GT_open_attempts=0,new_teacher_calls=0,image_decodes=0,neural_forwards=0,optimizer_steps=0,
        original42sources_unchanged=True,wall_seconds=time.perf_counter()-started,goal_status='ACTIVE')
    destination=ROOT/'a100_runs/visual_classical_20261121/supervisor/temporal_observation_support_audit_20261126.json'
    atomic_json(destination,receipt)
    print(json.dumps(dict(status=receipt['status'],physical_observations=receipt['physical_observations'],settings=settings,
        GT_open_attempts=0,neural_forwards=0)),flush=True)


if __name__=='__main__':main()
