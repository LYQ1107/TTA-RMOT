"""Frozen native-direct head: isolate visual edges and appearance alignment."""
import argparse
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from flex_uda.a100.mechanism import PAIRS,metadata,generation_guard
from flex_uda.active.acquisition import exclusive_group
from flex_uda.active.grounded_graph import refine_queries,query_geometry
from flex_uda.active.visual_classical import visual_graph,refine_distilled_prior
from flex_uda.active.semantic_colour import appearance_cues,shuffled_cues,crossfit_predictions,correct_belief,append_history
from flex_uda.presence.predictions import seal_mask
from scripts.a100_visual_classical_prepare import OUT
from scripts.a100_visual_classical_rgb_parity import source_spec
from scripts.a100_visual_classical_direct_teacher_pcd import DIRECT
from scripts.a100_visual_classical_semantic_colour import empty_history
from scripts.a100_visual_classical_joint_visual import global_shuffle,row_shift,query_teacher_maps,append_text_history
from scripts.a100_visual_classical_strict_prefix import plain,immediate_predictions

FACTORIAL=OUT/'native_direct_factorial'
METHODS=('ANCHOR_CAL','ROW_SHIFT','SHUFFLED','GLOBAL_SHUFFLED','QUERY_TEACHER','COMPLETE')


def complete_graph(unary,pcd,colours,texts,teacher):
    parts,geometry=query_geometry(pcd,colours,texts)
    result=np.array(unary,copy=True);errors=[]
    for q,mask in enumerate(parts):
        if not mask.any():continue
        ids=np.flatnonzero(np.all(colours[:,mask>0].sum(2)>0,axis=1))
        local,_=visual_graph(geometry[q,ids]);complete=np.ones_like(local)-np.eye(len(ids))
        if complete.sum():complete*=local.sum()/complete.sum()
        errors.append(abs(float(complete.sum()-local.sum())))
        weights=np.zeros((len(unary[q]),len(unary[q])));weights[np.ix_(ids,ids)]=complete
        result[q]=refine_distilled_prior(unary[q],geometry[q],teacher[q],source_weight=.1,weights_override=weights)
    return result,max(errors,default=0.)


def run(pair,seed):
    guard=generation_guard();begin=time.perf_counter();root=FACTORIAL/pair/f'seed{seed}'
    if (root/'completion.json').is_file():return
    root.mkdir(parents=True,exist_ok=True)
    implementation={name:file_sha(ROOT/path) for name,path in dict(worker='scripts/a100_visual_classical_native_factorial.py',
        graph='flex_uda/active/grounded_graph.py',calibration='flex_uda/active/semantic_colour.py').items()}
    model,target=PAIRS[pair];m=metadata(target);source=source_spec(pair)[0];previous=DIRECT/pair/f'seed{seed}'
    history=empty_history();permuted_history=empty_history();global_history=empty_history();text_history={0:{},1:{}}
    receipts=[];calls=0
    for vi,(video,bank) in enumerate(sorted(m['videos'].items())):
        stamp=time.perf_counter();local=root/'videos'/video;local.mkdir(parents=True,exist_ok=True)
        queries=[q for q in m['episodes'] if q['video_id']==video];texts=[q['query_text'] for q in queries]
        with np.load(source/'source'/video/'sparse_inputs.npz') as a:pcd,colours,prior,tracks=a['pcd'],a['colours'],a['prior'],a['tracks']
        with np.load(previous/'videos'/video/'coarse_scores.npz') as a:
            assert np.array_equal(tracks,a['tracks'])
            values={name:a[original] for name,original in [('KD','KD_MEAN'),('GRAPH','GRAPH'),('CAL','CAL'),('JOINT','JOINT')]}
        obs=np.asarray(bank['observations']);lookup={int(t):i for i,t in enumerate(tracks)}
        columns=np.asarray([lookup[int(t)] for _,t in obs]);lengths=np.asarray([(obs[:,1]==t).sum() for t in tracks])
        cue,available,families,_=appearance_cues(colours,texts,lengths);teacher=np.full(cue.shape,np.nan)
        for answer in json.loads((source/'source'/video/'answers.json').read_text())['answers']:
            teacher[int(answer['local_query']),int(answer['local_track'])]=answer['yes_probability']
        calls+=int(np.isfinite(teacher).sum())
        probability,base,active,_=crossfit_predictions(cue,available,families,teacher,tracks,video,history)
        graph=refine_queries(values['KD'],pcd,colours,texts,teacher)
        errors={name:float(np.max(np.abs(computed-values[name]))) for name,computed in [
            ('GRAPH',graph),('CAL',correct_belief(values['KD'],probability,base,active)),('JOINT',correct_belief(graph,probability,base,active))]}
        assert not any(errors.values()),(pair,seed,video,errors)
        permuted=shuffled_cues(cue,available,video);sp,sb,sa,_=crossfit_predictions(permuted,available,families,teacher,tracks,video,permuted_history)
        global_cue=global_shuffle(cue,available,families,video);gp,gb,ga,_=crossfit_predictions(global_cue,available,families,teacher,tracks,video,global_history)
        qp,qb,qa=query_teacher_maps(available,families,teacher,tracks,video,history,text_history,texts)
        anchors=refine_queries(values['KD'],pcd,colours,texts,teacher,graph=False)
        complete,weight_error=complete_graph(values['KD'],pcd,colours,texts,teacher);assert weight_error<1e-10
        values.update(ANCHOR_CAL=correct_belief(anchors,probability,base,active),ROW_SHIFT=row_shift(graph,probability,base,active),
            SHUFFLED=correct_belief(graph,sp,sb,sa),GLOBAL_SHUFFLED=correct_belief(graph,gp,gb,ga),
            QUERY_TEACHER=correct_belief(graph,qp,qb,qa),COMPLETE=correct_belief(complete,probability,base,active))
        order_changes=sum(not np.array_equal(np.lexsort((tracks,-prior[q],-graph[q])),
            np.lexsort((tracks,-prior[q],-values['ROW_SHIFT'][q]))) for q in range(len(graph)))
        assert order_changes==0
        exclusive=exclusive_group(texts,target)
        for name,original in [('KD','KD_MEAN'),('GRAPH','GRAPH'),('CAL','CAL'),('JOINT','JOINT')]:
            assert np.array_equal(plain(values[name][:,columns],exclusive),np.load(previous/original/'videos'/video/'mask.npy'))
        if (local/'receipt.json').is_file():
            receipt=json.loads((local/'receipt.json').read_text());assert receipt['implementation']==implementation
        else:
            for name in METHODS:
                mask=plain(values[name][:,columns],exclusive);folder=root/name/'videos'/video;folder.mkdir(parents=True,exist_ok=True)
                np.save(folder/'mask.npy',mask);immediate_predictions(root/name,type('Context',(),dict(bank=bank,queries=queries))(),mask)
            np.savez_compressed(local/'coarse_scores.npz',tracks=tracks,**values)
            receipt=dict(video=video,index=vi,implementation=implementation,current_teacher_examples=int(np.isfinite(teacher).sum()),
                frozen_prediction_max_errors=errors,complete_graph_total_edge_weight_error=weight_error,
                row_shift_changed_within_query_order=order_changes,GT_open_attempts=len(guard),wall_seconds=time.perf_counter()-stamp)
            atomic_json(local/'receipt.json',receipt)
        receipts.append(receipt)
        append_history(history,cue,available,families,teacher);append_history(permuted_history,permuted,available,families,teacher)
        append_history(global_history,global_cue,available,families,teacher);append_text_history(text_history,available,families,teacher,texts)
        print(json.dumps(dict(event='NATIVE_FACTORIAL_VIDEO',pair=pair,seed=seed,completed=vi+1,total=len(m['videos']))),flush=True)
    methods=[]
    for name in METHODS:
        mask=np.zeros(m['candidate_query_rows'],bool)
        for video in sorted(m['videos']):
            local=np.load(root/name/'videos'/video/'mask.npy')
            for q,e in enumerate(e for e in m['episodes'] if e['video_id']==video):mask[e['start']:e['stop']]=local[q]
        seal=seal_mask(root/name,m,mask,target=target,model=model,method='NATIVE_FACTORIAL_'+name,GT_open_attempts=len(guard),extra=dict(
            seed=seed,implementation=implementation,neural_dimension=768,source_logit_zero=True,new_head_fits=0,
            new_teacher_calls=0,new_encoder_calls=0,current_video_transductive=True,no_future_teacher=True,quality_replay_not_cold_time=True))
        methods.append(dict(method=name,rows=seal['prediction_rows']))
    assert not guard and calls==(1160 if pair=='C4' else 1103)
    atomic_json(root/'completion.json',dict(status='SEALED_NATIVE_DIRECT_FACTORIAL',pair=pair,seed=seed,methods=methods,receipts=receipts,
        logical_teacher_calls=calls,GT_open_attempts=0,new_teacher_calls=0,new_encoder_calls=0,new_head_fits=0,
        implementation=implementation,wall_seconds=time.perf_counter()-begin,goal_status='ACTIVE'))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--pair',choices=['C4','C1'],required=True);parser.add_argument('--seed',type=int,required=True)
    args=parser.parse_args();run(args.pair,args.seed)
