"""Leak-free teacher-predictive local/global choice on the sealed native head."""
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
from scripts.a100_visual_classical_native_factorial import FACTORIAL
from flex_uda.active.topology_selection import apply_graph
from flex_uda.active.contract_graph import contract_graphs
from scripts.a100_visual_classical_semantic_colour import empty_history
from scripts.a100_visual_classical_joint_visual import global_shuffle,row_shift,query_teacher_maps,append_text_history
from scripts.a100_visual_classical_strict_prefix import plain,immediate_predictions

PREDICTIVE=OUT/'weighted_classical'
METHODS=('GRAPH','CAL','JOINT')


def run(pair,seed):
    guard=generation_guard();begin=time.perf_counter();root=PREDICTIVE/pair/f'seed{seed}'
    if (root/'completion.json').is_file():return
    root.mkdir(parents=True,exist_ok=True)
    implementation={name:file_sha(ROOT/path) for name,path in dict(worker='scripts/a100_visual_classical_weighted_classical.py',selector='flex_uda/active/grounded_graph.py',frozen_strict_selector='flex_uda/active/topology_selection.py',
        graph='flex_uda/active/grounded_graph.py',calibration='flex_uda/active/semantic_colour.py').items()}
    model,target=PAIRS[pair];m=metadata(target);source=source_spec(pair)[0];previous=OUT/'confidence_teacher'/pair/f'seed{seed}';native=DIRECT/pair/f'seed{seed}'
    history=empty_history();protocol=json.loads((OUT/'preregistration_v1.json').read_text())['protocol'][pair]
    receipts=[];calls=0
    for vi,(video,bank) in enumerate(sorted(m['videos'].items())):
        stamp=time.perf_counter();local=root/'videos'/video;local.mkdir(parents=True,exist_ok=True)
        queries=[q for q in m['episodes'] if q['video_id']==video];texts=[q['query_text'] for q in queries]
        with np.load(source/'source'/video/'sparse_inputs.npz') as a:pcd,colours,prior,tracks=a['pcd'],a['colours'],a['prior'],a['tracks']
        with np.load(previous/'videos'/video/'coarse_scores.npz') as a:
            assert np.array_equal(tracks,a['tracks'])
            values=dict(KD=a['KD_MEAN'])
        with np.load(native/'videos'/video/'coarse_scores.npz') as a:original_native={name:a[name] for name in ('KD_MEAN','GRAPH','CAL','JOINT')}
        obs=np.asarray(bank['observations']);lookup={int(t):i for i,t in enumerate(tracks)}
        columns=np.asarray([lookup[int(t)] for _,t in obs]);lengths=np.asarray([(obs[:,1]==t).sum() for t in tracks])
        cue,available,families,_=appearance_cues(colours,texts,lengths);teacher=np.full(cue.shape,np.nan)
        for answer in json.loads((source/'source'/video/'answers.json').read_text())['answers']:
            teacher[int(answer['local_query']),int(answer['local_track'])]=answer['yes_probability']
        calls+=int(np.isfinite(teacher).sum())
        probability,base,active,_=crossfit_predictions(cue,available,families,teacher,tracks,video,history)
        selection_start=time.perf_counter()
        values['GRAPH']=refine_queries(values['KD'],pcd,colours,texts,teacher)
        values['CAL']=correct_belief(values['KD'],probability,base,active)
        values['JOINT']=correct_belief(values['GRAPH'],probability,base,active)
        full_native=refine_queries(original_native['KD_MEAN'],pcd,colours,texts,teacher)
        errors=dict(GRAPH=float(np.max(np.abs(full_native-original_native['GRAPH']))),
            CAL=float(np.max(np.abs(correct_belief(original_native['KD_MEAN'],probability,base,active)-original_native['CAL']))),
            JOINT=float(np.max(np.abs(correct_belief(full_native,probability,base,active)-original_native['JOINT']))))
        assert not any(errors.values()),(pair,seed,video,errors)
        selection_seconds=time.perf_counter()-selection_start
        exclusive=exclusive_group(texts,target)
        assert np.array_equal(plain(values['KD'][:,columns],exclusive),np.load(previous/'KD_MEAN/videos'/video/'mask.npy'))
        if (local/'receipt.json').is_file():
            receipt=json.loads((local/'receipt.json').read_text());assert receipt['implementation']==implementation
        else:
            for name in METHODS:
                mask=plain(values[name][:,columns],exclusive);folder=root/name/'videos'/video;folder.mkdir(parents=True,exist_ok=True)
                np.save(folder/'mask.npy',mask);immediate_predictions(root/name,type('Context',(),dict(bank=bank,queries=queries))(),mask)
            np.savez_compressed(local/'coarse_scores.npz',tracks=tracks,**values)
            receipt=dict(video=video,index=vi,implementation=implementation,current_teacher_examples=int(np.isfinite(teacher).sum()),
                frozen_prediction_max_errors=errors,same_weighted_neural_head_as_ordinary=True,graph_and_calibration_CPU_seconds=selection_seconds,
                GT_open_attempts=len(guard),wall_seconds=time.perf_counter()-stamp)
            atomic_json(local/'receipt.json',receipt)
        receipts.append(receipt)
        append_history(history,cue,available,families,teacher)
        print(json.dumps(dict(event='WEIGHTED_CLASSICAL_VIDEO',pair=pair,seed=seed,completed=vi+1,total=len(m['videos']))),flush=True)
    methods=[]
    for name in METHODS:
        mask=np.zeros(m['candidate_query_rows'],bool)
        for video in sorted(m['videos']):
            local=np.load(root/name/'videos'/video/'mask.npy')
            for q,e in enumerate(e for e in m['episodes'] if e['video_id']==video):mask[e['start']:e['stop']]=local[q]
        seal=seal_mask(root/name,m,mask,target=target,model=model,method='WEIGHTED_CLASSICAL_'+name,GT_open_attempts=len(guard),extra=dict(
            seed=seed,implementation=implementation,neural_dimension=768,source_logit_zero=True,new_head_fits=0,
            new_teacher_calls=0,new_encoder_calls=0,current_video_transductive=True,no_future_teacher=True,quality_replay_not_cold_time=True))
        methods.append(dict(method=name,rows=seal['prediction_rows']))
    assert not guard and calls==(1160 if pair=='C4' else 1103)
    atomic_json(root/'completion.json',dict(status='SEALED_WEIGHTED_CLASSICAL',pair=pair,seed=seed,methods=methods,receipts=receipts,
        logical_teacher_calls=calls,GT_open_attempts=0,new_teacher_calls=0,new_encoder_calls=0,new_head_fits=0,
        implementation=implementation,wall_seconds=time.perf_counter()-begin,goal_status='ACTIVE'))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--pair',choices=['C4','C1'],required=True);parser.add_argument('--seed',type=int,required=True)
    args=parser.parse_args();run(args.pair,args.seed)
