"""No-new-QA graph specificity controls using the identical actual prefix KD.

Permuting node/feature correspondence preserves graph weights up to a node
permutation. The complete graph preserves total edge precision. Teacher anchors,
head outputs and missing-part masks stay fixed. No target GT is an input.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.special import expit
from flex_tta.runtime.pf_receipts import atomic_json
from flex_uda.a100.mechanism import metadata,PAIRS,FrozenInputs,generation_guard
from flex_uda.active.color_cohort import augmented_features,part_mask
from flex_uda.active.visual_classical import visual_graph,refine_distilled_prior
from flex_uda.active.acquisition import expression_types,exclusive_group
from flex_uda.presence.predictions import seal_mask
from scripts.a100_visual_classical_prepare import OUT
from scripts.a100_visual_classical_strict_prefix import plain,immediate_predictions,read_guard

METHODS=('PERMUTED_GRAPH','COMPLETE_GRAPH','PERMUTED_AT_FULL_COUNT','COMPLETE_AT_FULL_COUNT')
EVIDENCE_METHODS=('VISUAL_GUARD','BMA','MAP','EQUAL','BMA_GUARD','MAP_GUARD','EQUAL_GUARD',
                  'BMA_GUARD_AT_FULL_COUNT','MAP_GUARD_AT_FULL_COUNT','EQUAL_GUARD_AT_FULL_COUNT',
                  'BMA_GUARD_AT_KD_COUNT')


def fixed_counts(score,reference,fc,exclusive):
    result=np.zeros_like(reference,dtype=bool)
    for frame in np.unique(fc.observations[:,0]):
        ids=np.flatnonzero(fc.observations[:,0]==frame);counts=reference[:,ids].sum(1).astype(int)
        if exclusive:
            assert np.all(reference[:,ids].sum(0)<=1)
            slots=np.repeat(np.arange(len(fc.queries)),counts)
            if len(slots):
                rows,columns=linear_sum_assignment(-score[slots][:,ids].astype(float))
                result[slots[rows],ids[columns]]=True
        else:
            for qi,count in enumerate(counts):
                order=np.lexsort((fc.observations[ids,1],-fc.margin[qi,ids],-score[qi,ids]))
                result[qi,ids[order[:count]]]=True
        assert np.array_equal(result[:,ids].sum(1),counts)
    return result


def run(pair,seed,bounded,adaptive=False):
    gt=generation_guard();visible=set();forbidden,reads=read_guard(pair,visible)
    group='strong_cover_seed'+str(seed)+('_clipped' if bounded else '')
    source=OUT/group/pair;root=OUT/('graph_evidence_controls' if adaptive else 'geometry_controls')/group/pair
    methods_to_write=EVIDENCE_METHODS if adaptive else METHODS
    if (root/'completion.json').is_file():return
    assert (source/'completion.json').is_file()
    started=time.perf_counter();model,target=PAIRS[pair];m=metadata(target);rows=[];probe=[]
    contract=json.loads((OUT/'preregistration_v1.json').read_text())['protocol'][pair]
    policy='COVER' if seed==0 else 'COVER_S'+str(seed)
    def prevent_future(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        path=Path(os.fsdecode(args[0]));flags=args[2] if len(args)>2 else 0
        if flags&(os.O_WRONLY|os.O_RDWR):return
        try:parts=path.relative_to(source).parts
        except ValueError:return
        if 'videos' in parts:
            index=parts.index('videos')+1
            if index<len(parts) and parts[index] not in visible:raise PermissionError('Future source-head output: '+str(path))
        elif path.suffix in ('.pt','.pth'):raise PermissionError('Aggregate future trained head: '+str(path))
    sys.addaudithook(prevent_future)
    for video in sorted(m['videos']):
        visible.add(video);fc=FrozenInputs(pair,video)
        with np.load(OUT/'inputs'/pair/video/'inputs.npz') as values:original=values['features']
        with np.load(OUT/'color_features'/pair/video/'features.npz') as values:colour=values['colours']
        geometry,_,_=augmented_features(original,colour,[q['query_text'] for q in fc.queries])
        saved=np.load(source/'SOFT_KD/videos'/video/'scores.npy',mmap_mode='r')
        kd=np.stack([saved[:,fc.by_track[t][0]] for t in fc.tracks],axis=1)
        teacher=np.full(kd.shape,np.nan)
        for answer in json.loads((OUT/'causal_teacher'/pair/policy/(video+'.json')).read_text())['answers']:
            teacher[answer['local_query'],answer['local_track']]=answer['yes_probability']
        values={name:kd.copy() for name in (('IDENTITY',)+EVIDENCE_METHODS[:7] if adaptive else ('IDENTITY','PERMUTED_GRAPH','COMPLETE_GRAPH'))};audit=[]
        for qi,q in enumerate(fc.queries):
            required=part_mask(q['query_text'])>0
            if not required.any():continue
            valid=np.flatnonzero(np.all(colour[:,required].sum(2)>0,axis=1));local,_=visual_graph(geometry[qi,valid])
            if adaptive:
                from flex_uda.active.graph_evidence import graph_models,model_weights,leave_one_out
                visible_parts=np.zeros(len(fc.tracks),bool);visible_parts[valid]=True
                matrices,kernels=graph_models(geometry[qi],visible_parts);mix,record=model_weights(kernels,teacher[qi])
                predictions=np.stack([refine_distilled_prior(kd[qi],geometry[qi],teacher[qi],source_weight=.1,weights_override=w) for w in matrices])
                guard=contract=='PRESUPPOSED' and not np.any(teacher[qi,np.isfinite(teacher[qi])]>=.5)
                visual,bma,map_value,equal=predictions[0],mix@predictions,predictions[int(mix.argmax())],predictions.mean(0)
                for name,value in [('IDENTITY',visual),('VISUAL_GUARD',predictions[2] if guard else visual),
                    ('BMA',bma),('MAP',map_value),('EQUAL',equal),('BMA_GUARD',predictions[2] if guard else bma),
                    ('MAP_GUARD',predictions[2] if guard else map_value),('EQUAL_GUARD',predictions[2] if guard else equal)]:values[name][qi]=value
                audit.append(dict(query=qi,weights=mix.tolist(),contract=contract,guard_active=bool(guard),**record))
                if seed==0 and not bounded:
                    for item in leave_one_out(kernels,teacher[qi]):probe.append(dict(video=video,query=qi,**item))
                continue
            rng=np.random.default_rng(seed+int(hashlib.sha256((video+str(qi)).encode()).hexdigest()[:8],16))
            permutation=rng.permutation(len(valid));permuted=local[np.ix_(permutation,permutation)]
            complete=np.zeros_like(local)
            if len(valid)>1:
                complete[:]=local.sum()/(len(valid)*(len(valid)-1));np.fill_diagonal(complete,0)
            for method,weights in [('IDENTITY',local),('PERMUTED_GRAPH',permuted),('COMPLETE_GRAPH',complete)]:
                matrix=np.zeros((len(fc.tracks),len(fc.tracks)));matrix[np.ix_(valid,valid)]=weights
                values[method][qi]=refine_distilled_prior(kd[qi],geometry[qi],teacher[qi],source_weight=.1,weights_override=matrix)
        lookup={t:i for i,t in enumerate(fc.tracks)};columns=np.asarray([lookup[int(t)] for _,t in fc.observations])
        exclusive=exclusive_group([q['query_text'] for q in fc.queries],target)
        reference=np.load(source/'COHORT/videos'/video/'mask.npy');scores={};masks={}
        for method,value in values.items():
            scores[method]=value[:,columns];masks[method]=plain(scores[method],exclusive)
            for qi,q in enumerate(fc.queries):
                if 'MOTION' in expression_types(q['query_text']):
                    masks[method][qi]&=np.asarray([expit(fc.raw_presence[qi,fc.frame_index[int(f)]])>=.5 for f,_ in fc.observations])
        assert np.array_equal(masks['IDENTITY'],reference),'Graph override must reproduce every original COHORT decision'
        assert np.array_equal(fixed_counts(scores['IDENTITY'],reference,fc,exclusive),reference),'Common CountSelf projection changed source predictions'
        if adaptive:
            for method in ('BMA_GUARD','MAP_GUARD','EQUAL_GUARD'):
                masks[method+'_AT_FULL_COUNT']=fixed_counts(scores[method],reference,fc,exclusive)
            kd_reference=np.load(source/'SOFT_KD/videos'/video/'mask.npy')
            assert np.array_equal(fixed_counts(kd[:,columns],kd_reference,fc,exclusive),kd_reference),'KD CountSelf changed source predictions'
            masks['BMA_GUARD_AT_KD_COUNT']=fixed_counts(scores['BMA_GUARD'],kd_reference,fc,exclusive)
        else:
            for kind in ('PERMUTED','COMPLETE'):
                masks[kind+'_AT_FULL_COUNT']=fixed_counts(scores[kind+'_GRAPH'],reference,fc,exclusive)
        for method in methods_to_write:
            folder=root/method;local=folder/'videos'/video;local.mkdir(parents=True,exist_ok=True)
            np.save(local/'mask.npy',masks[method])
            immediate_predictions(folder,fc,masks[method])
        rows.append(dict(video=video,GT_open_attempts=len(gt),same_visual_graph_regression_changed_rows=0,
                         CountSelf_common_projection_changed_rows=0,
                         original_full_rows=int(reference.sum()),model_evidence=audit))
        print(json.dumps(dict(event='GRAPH_EVIDENCE_VIDEO' if adaptive else 'GRAPH_SPECIFICITY_VIDEO',pair=pair,seed=seed,bounded=bounded,video=video)),flush=True)
    methods=[]
    for method in methods_to_write:
        folder=root/method;mask=np.zeros(m['candidate_query_rows'],dtype=bool)
        for video in sorted(m['videos']):
            value=np.load(folder/'videos'/video/'mask.npy')
            for qi,q in enumerate([q for q in m['episodes'] if q['video_id']==video]):mask[q['start']:q['stop']]=value[qi]
        seal=seal_mask(folder,m,mask,target=target,model=model,method=method,GT_open_attempts=len(gt),extra=dict(
            exact_same_actual_prefix_KD=True,exact_same_teacher_answers=True,no_new_teacher_calls=True,
            same_missing_part_nodes=True,permutation_preserves_edge_weight_multiset=not adaptive,
            teacher_only_graph_model_comparison=adaptive,
            complete_graph_preserves_total_edge_precision=True,counts_from_GT=False,
            fixed_count_task_exclusivity_preserved=exclusive_group([q['query_text'] for q in fc.queries],target)))
        methods.append(dict(method=method,rows=seal['prediction_rows']))
    assert not gt and not forbidden
    atomic_json(root/'completion.json',dict(status='SEALED_GRAPH_SPECIFICITY_CONTROLS',pair=pair,seed=seed,
        bounded_soft_targets=bounded,source_group=group,methods=methods,GT_open_attempts=0,
        same_graph_regression_changed_rows=0,new_teacher_calls=0,teacher_head_training_updates=0,
        videos=rows,wall_seconds=time.perf_counter()-started,prepaid_quality_control_not_cold_runtime=True))
    if probe:
        names=('VISUAL','GLOBAL','INDEPENDENT','BMA','MAP','EQUAL');summary={}
        for name in names:
            truth=np.asarray([r['target'] for r in probe]);probability=np.clip([r[name] for r in probe],1e-6,1-1e-6)
            summary[name]=dict(teacher_soft_CE=float(np.mean(-truth*np.log(probability)-(1-truth)*np.log(1-probability))),teacher_Brier=float(np.mean((truth-probability)**2)))
        atomic_json(root/'teacher_only_probe.json',dict(held_teacher_answers=len(probe),models=summary,
            held_answer_excluded_from_model_weight_and_prediction=True,no_KD_predictions_in_model_comparison=True,
            teacher_consistency_not_RMOT_accuracy=True,target_GT_used=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pair',choices=['C4','C1'],required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--bounded',action='store_true');p.add_argument('--adaptive',action='store_true');a=p.parse_args();run(a.pair,a.seed,a.bounded,a.adaptive)
