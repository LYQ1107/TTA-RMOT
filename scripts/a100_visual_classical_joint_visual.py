"""Sealed-head factorial: visual graph, appearance calibration and base rates."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.special import expit,logit
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from flex_uda.a100.mechanism import PAIRS,metadata,generation_guard
from flex_uda.active.acquisition import exclusive_group
from flex_uda.active.grounded_graph import refine_queries
from flex_uda.active.semantic_colour import appearance_cues,shuffled_cues,crossfit_predictions,correct_belief,append_history,ODDS_CAP
from flex_uda.presence.predictions import seal_mask
from scripts.a100_visual_classical_prepare import OUT
from scripts.a100_visual_classical_rgb_parity import PARITY,source_spec
from scripts.a100_visual_classical_semantic_colour import SEMANTIC,empty_history
from scripts.a100_visual_classical_strict_prefix import plain,immediate_predictions

JOINT=OUT/'joint_visual'
METHODS=('CAL','JOINT','SHUFFLED','ROW_SHIFT','GLOBAL_SHUFFLED','QUERY_TEACHER','ANCHOR_CAL',
    'EARLY_GRAPH','EARLY_CAL','EARLY_JOINT')


def global_shuffle(cue,available,families,video):
    result=cue.copy()
    for family in (0,1):
        selected=available&(families[:,None]==family)
        seed=int(hashlib.sha256((video+':'+str(family)+':global-cue-placebo').encode()).hexdigest()[:8],16)
        result[selected]=np.random.default_rng(seed).permutation(cue[selected])
    return result


def row_shift(unary,probability,prior,active):
    result=np.asarray(unary,dtype=float).copy()
    for q in range(len(unary)):
        selected=active[q]
        if not selected.any():continue
        shift=float(np.clip(logit(probability[q,selected])-logit(prior[q,selected]),-ODDS_CAP,ODDS_CAP).mean())
        result[q]=expit(logit(np.clip(result[q],np.finfo(float).tiny,np.nextafter(1.,0.)))+shift)
    return result


def query_teacher_maps(available,families,answers,tracks,video,history,text_history,texts):
    """Empirical-Bayes control without colour compatibility values."""
    probability=np.full(available.shape,.5);prior=probability.copy();active=np.zeros(available.shape,bool)
    folds=np.asarray([int(hashlib.sha256((video+':'+str(int(t))+':semantic-fold').encode()).hexdigest()[:8],16)%5 for t in tracks])
    known=np.isfinite(answers)&available
    for fold in sorted(set(folds)):
        for family in (0,1):
            train=known&(families[:,None]==family)&(folds[None,:]!=fold)
            global_y=np.r_[history[family]['y'],np.clip(answers[train],.02,.98)]
            if len(global_y)<8:continue
            mean=float(np.clip(global_y.mean(),.02,.98))
            for q in np.flatnonzero(families==family):
                selected=available[q]&(folds==fold)
                key=' '.join(texts[q].lower().split())
                y=np.r_[text_history[family].get(key,np.empty(0)),np.clip(answers[q,train[q]],.02,.98)]
                if not len(y) or not selected.any():continue
                probability[q,selected]=(float(y.sum())+2*mean)/(len(y)+2)
                prior[q,selected]=mean;active[q,selected]=True
    return probability,prior,active


def append_text_history(history,available,families,answers,texts):
    for q in np.flatnonzero(families>=0):
        key=' '.join(texts[q].lower().split());family=int(families[q])
        y=np.clip(answers[q,np.isfinite(answers[q])&available[q]],.02,.98)
        history[family][key]=np.r_[history[family].get(key,np.empty(0)),y]


def run(pair,seed):
    guard=generation_guard();begin=time.perf_counter();root=JOINT/pair/f'seed{seed}'
    if (root/'completion.json').is_file():return
    root.mkdir(parents=True,exist_ok=True)
    implementation={name:file_sha(ROOT/path) for name,path in dict(worker='scripts/a100_visual_classical_joint_visual.py',
        calibration='flex_uda/active/semantic_colour.py',graph='flex_uda/active/grounded_graph.py').items()}
    model,target=PAIRS[pair];m=metadata(target);source=source_spec(pair)[0]
    old=PARITY/pair/f'seed{seed}';semantic=SEMANTIC/pair/f'seed{seed}'
    history=empty_history();permuted_history=empty_history();global_history=empty_history();text_history={0:{},1:{}}
    receipts=[];calls=0
    for vi,(video,bank) in enumerate(sorted(m['videos'].items())):
        stamp=time.perf_counter();local=root/'videos'/video;local.mkdir(parents=True,exist_ok=True)
        queries=[e for e in m['episodes'] if e['video_id']==video];texts=[e['query_text'] for e in queries]
        with np.load(source/'source'/video/'sparse_inputs.npz') as a:
            pcd,colours,prior,tracks=a['pcd'],a['colours'],a['prior'],a['tracks']
        obs=np.asarray(bank['observations']);first=np.asarray([np.flatnonzero(obs[:,1]==t)[0] for t in tracks])
        lookup={int(t):i for i,t in enumerate(tracks)};columns=np.asarray([lookup[int(t)] for _,t in obs])
        arrays={}
        for name,folder in [('KD','RGB_KD_MEAN'),('GRAPH','RGB_GRAPH_MEAN')]:
            expanded=np.load(old/folder/'videos'/video/'scores.npy',mmap_mode='r')
            arrays[name]=np.asarray(expanded[:,first]);assert np.array_equal(arrays[name][:,columns],expanded)
        lengths=np.asarray([(obs[:,1]==t).sum() for t in tracks]);cue,available,families,_=appearance_cues(colours,texts,lengths)
        with np.load(semantic/'videos'/video/'coarse_scores.npz') as a:
            assert np.array_equal(tracks,a['tracks']) and np.array_equal(cue,a['cue'])
            assert np.array_equal(available,a['available']) and np.array_equal(families,a['families'])
            early=a['SEM400'];previous_map=a['MONO_ONLY']
        teacher=np.full(cue.shape,np.nan)
        for answer in json.loads((source/'source'/video/'answers.json').read_text())['answers']:
            teacher[int(answer['local_query']),int(answer['local_track'])]=answer['yes_probability']
        calls+=int(np.isfinite(teacher).sum());map_begin=time.perf_counter()
        probability,base,active,_=crossfit_predictions(cue,available,families,teacher,tracks,video,history)
        map_seconds=time.perf_counter()-map_begin
        map_error=float(np.max(np.abs(probability[active].astype('float32')-previous_map[active]))) if active.any() else 0.
        assert map_error==0,(pair,video,'map replay',map_error)
        graph_begin=time.perf_counter();graph=refine_queries(arrays['KD'],pcd,colours,texts,teacher)
        graph_seconds=time.perf_counter()-graph_begin
        graph_error=float(np.max(np.abs(graph-arrays['GRAPH'])));assert graph_error==0,(pair,video,'graph replay',graph_error)
        apply_begin=time.perf_counter();joint=correct_belief(graph,probability,base,active);apply_seconds=time.perf_counter()-apply_begin
        permuted=shuffled_cues(cue,available,video)
        sp,sb,sa,_=crossfit_predictions(permuted,available,families,teacher,tracks,video,permuted_history)
        global_cue=global_shuffle(cue,available,families,video)
        gp,gb,ga,_=crossfit_predictions(global_cue,available,families,teacher,tracks,video,global_history)
        qp,qb,qa=query_teacher_maps(available,families,teacher,tracks,video,history,text_history,texts)
        anchors=refine_queries(arrays['KD'],pcd,colours,texts,teacher,graph=False)
        early_graph=refine_queries(early,pcd,colours,texts,teacher)
        arrays.update(CAL=correct_belief(arrays['KD'],probability,base,active),JOINT=joint,
            SHUFFLED=correct_belief(graph,sp,sb,sa),ROW_SHIFT=row_shift(graph,probability,base,active),
            GLOBAL_SHUFFLED=correct_belief(graph,gp,gb,ga),QUERY_TEACHER=correct_belief(graph,qp,qb,qa),
            ANCHOR_CAL=correct_belief(anchors,probability,base,active),EARLY_GRAPH=early_graph,
            EARLY_CAL=correct_belief(early,probability,base,active),EARLY_JOINT=correct_belief(early_graph,probability,base,active))
        changed_order=sum(not np.array_equal(np.lexsort((tracks,-prior[q],-graph[q])),
            np.lexsort((tracks,-prior[q],-arrays['ROW_SHIFT'][q]))) for q in range(len(graph)))
        assert changed_order==0,(pair,video,'row-shift within-query order',changed_order)
        exclusive=exclusive_group(texts,target)
        for name,expected in [('KD','RGB_KD_MEAN'),('GRAPH','RGB_GRAPH_MEAN')]:
            assert np.array_equal(plain(arrays[name][:,columns],exclusive),np.load(old/expected/'videos'/video/'mask.npy'))
        assert np.array_equal(plain(early[:,columns],exclusive),np.load(semantic/'SEM400/videos'/video/'mask.npy'))
        post_seconds=time.perf_counter()-map_begin
        if (local/'receipt.json').is_file():
            receipt=json.loads((local/'receipt.json').read_text());assert receipt['implementation']==implementation
        else:
            for name in METHODS:
                mask=plain(arrays[name][:,columns],exclusive);destination=root/name;output=destination/'videos'/video
                output.mkdir(parents=True,exist_ok=True);np.save(output/'mask.npy',mask)
                immediate_predictions(destination,type('Context',(),dict(bank=bank,queries=queries))(),mask)
            np.savez_compressed(local/'coarse_scores.npz',tracks=tracks,**arrays)
            receipt=dict(video=video,index=vi,implementation=implementation,current_teacher_examples=int(np.isfinite(teacher).sum()),
                graph_max_probability_error=graph_error,previous_map_max_probability_error=map_error,
                row_shift_changed_within_query_order=changed_order,active_map_nodes=int(active.sum()),
                main_graph_cpu_seconds=graph_seconds,main_map_cpu_seconds=map_seconds,main_apply_cpu_seconds=apply_seconds,
                all_controls_postprocess_seconds=post_seconds,GT_open_attempts=len(guard),wall_seconds=time.perf_counter()-stamp)
            atomic_json(local/'receipt.json',receipt)
        receipts.append(receipt)
        append_history(history,cue,available,families,teacher)
        append_history(permuted_history,permuted,available,families,teacher)
        append_history(global_history,global_cue,available,families,teacher)
        append_text_history(text_history,available,families,teacher,texts)
        print(json.dumps(dict(event='JOINT_VISUAL_VIDEO',pair=pair,seed=seed,completed=vi+1,total=len(m['videos']))),flush=True)
    methods=[]
    for name in METHODS:
        mask=np.zeros(m['candidate_query_rows'],bool)
        for video in sorted(m['videos']):
            local=np.load(root/name/'videos'/video/'mask.npy')
            for q,e in enumerate(e for e in m['episodes'] if e['video_id']==video):mask[e['start']:e['stop']]=local[q]
        seal=seal_mask(root/name,m,mask,target=target,model=model,method='JOINT_VISUAL_'+name,GT_open_attempts=len(guard),extra=dict(
            seed=seed,new_teacher_calls=0,new_encoder_calls=0,new_head_fits=0,implementation=implementation,
            same_sealed_RGB1080_head_except_declared_SEM1081_early_controls=True,no_future_teacher=True,quality_replay_not_cold_cost=True))
        methods.append(dict(method=name,rows=seal['prediction_rows']))
    assert not guard and calls==(1160 if pair=='C4' else 1103)
    atomic_json(root/'completion.json',dict(status='SEALED_JOINT_VISUAL',pair=pair,seed=seed,methods=methods,receipts=receipts,
        GT_open_attempts=0,new_teacher_calls=0,new_encoder_calls=0,new_head_fits=0,logical_teacher_calls=calls,
        implementation=implementation,wall_seconds=time.perf_counter()-begin,goal_status='ACTIVE'))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--pair',choices=['C4','C1'],required=True);parser.add_argument('--seed',type=int,required=True)
    args=parser.parse_args();run(args.pair,args.seed)
