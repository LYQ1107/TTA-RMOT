"""Equal-budget coverage-search evidence; strong ordinary heads and frozen classical controls."""
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
from flex_uda.active.context_cohort import distillation_features
from flex_uda.active.grounded_graph import refine_queries
from flex_uda.active.appearance_runtime import empty_appearance_history,regularize_appearance,commit_appearance_history
from flex_uda.active.acquisition import exclusive_group
from flex_uda.presence.predictions import seal_mask
from scripts.a100_visual_classical_prepare import OUT
from scripts.a100_visual_classical_rgb_parity import source_spec
from scripts.a100_visual_classical_strict_prefix import plain,immediate_predictions

from scripts.a100_visual_classical_search_collect import SEARCH,EVIDENCE
from flex_uda.active.topology_selection import graph_matrices,apply_graph
DIRECT=SEARCH/'quality'
METHODS=('KD400','KD_FINAL','KD_MEAN','HARD400','HARD_FINAL','GRAPH','CAL','JOINT','COMPLETE')


def run(pair,seed):
    import torch
    from flex_uda.active.confidence_teacher import fit_head
    guard=generation_guard();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    root=DIRECT/pair/f'seed{seed}'
    if (root/'completion.json').is_file():return
    root.mkdir(parents=True,exist_ok=True);begin=time.perf_counter()
    implementation={name:file_sha(ROOT/path) for name,path in dict(worker='scripts/a100_visual_classical_coverage_search.py',
        fitting='flex_uda/active/confidence_teacher.py',selector='flex_uda/active/coverage_search.py',graph='flex_uda/active/grounded_graph.py',calibration='flex_uda/active/semantic_colour.py').items()}
    assert pair=='C4' and (EVIDENCE/pair/'completion.json').is_file()
    model,target=PAIRS[pair];m=metadata(target);source=source_spec(pair)[0];videos=sorted(m['videos'])
    total=np.zeros(768);square=np.zeros(768);rows=0;bx=[];by=[];receipts=[];history=empty_appearance_history();start=0
    stream=root/'stream.pt'
    if stream.is_file():
        saved=torch.load(stream,map_location='cpu',weights_only=False);assert saved['implementation']==implementation
        total,square,rows=saved['total'],saved['square'],saved['rows'];bx,by=saved['bx'],saved['by']
        receipts,history,start=saved['receipts'],saved['history'],saved['next_video']
    for vi,video in enumerate(videos):
        if vi<start:continue
        stamp=time.perf_counter();bank=m['videos'][video];queries=[e for e in m['episodes'] if e['video_id']==video];texts=[e['query_text'] for e in queries]
        with np.load(source/'source'/video/'sparse_inputs.npz') as a:pcd,colours,prior,tracks=a['pcd'],a['colours'],a['prior'],a['tracks']
        raw=np.asarray(pcd,dtype='float32');current=raw.reshape(-1,768)
        total+=current.sum(0,dtype=float);square+=np.square(current.astype(float)).sum(0);rows+=len(current)
        mean=total/rows;std=np.maximum(np.sqrt(np.maximum(square/rows-mean**2,0)),1e-3)
        teacher=np.full(prior.shape,np.nan)
        for answer in json.loads((EVIDENCE/pair/'source'/video/'answers.json').read_text())['answers']:
            teacher[int(answer['local_query']),int(answer['local_track'])]=answer['yes_probability']
        known=np.isfinite(teacher);bx.append(raw[known]);by.append(teacher[known]);x=np.concatenate(bx);y=np.clip(np.concatenate(by),.02,.98)
        weight=1.+3.*np.abs(2.*np.concatenate(by)-1.)
        # Both training and current prediction remove the fixed source term.
        z=np.zeros(len(y));current_z=np.zeros(prior.size);heads={};fits={}
        for label,soft in [('SOFT',True),('HARD',False)]:
            values,fit,checkpoint=fit_head(x,z,y,current,current_z,mean,std,sample_weight=weight,soft=soft,seed=seed)
            fit.update(effective_model='direct teacher classifier; fixed source residual identically zero',source_logit_residual_disabled=True)
            fits[label]=fit;heads[label]={name:value.reshape(prior.shape) for name,value in values.items()}
            torch.save(checkpoint,root/(label+'_latest_head.pt'))
        scores=dict(KD400=heads['SOFT']['400'],KD_FINAL=heads['SOFT']['CONVERGED'],
            KD_MEAN=.5*(heads['SOFT']['400']+heads['SOFT']['CONVERGED']),HARD400=heads['HARD']['400'],HARD_FINAL=heads['HARD']['CONVERGED'])
        obs=np.asarray(bank['observations']);lengths=np.asarray([(obs[:,1]==t).sum() for t in tracks])
        scores['GRAPH']=refine_queries(scores['KD_MEAN'],pcd,colours,texts,teacher)
        scores['CAL'],inputs,cal_receipt=regularize_appearance(scores['KD_MEAN'],colours,texts,lengths,teacher,tracks,video,history)
        scores['JOINT'],_,joint_receipt=regularize_appearance(scores['GRAPH'],colours,texts,lengths,teacher,tracks,video,history)
        geometry,local,complete,valid,supported=graph_matrices(pcd,colours,texts)
        full=apply_graph(scores['KD_MEAN'],teacher,geometry,local,supported)
        assert np.array_equal(full,scores['GRAPH'])
        global_graph=apply_graph(scores['KD_MEAN'],teacher,geometry,complete,supported)
        scores['COMPLETE'],_,_=regularize_appearance(global_graph,colours,texts,lengths,teacher,tracks,video,history)
        commit_appearance_history(history,inputs,teacher)
        with np.load(OUT/'confidence_teacher'/pair/f'seed{seed}'/'videos'/video/'coarse_scores.npz') as a:scores['COVER_KD']=a['KD_MEAN']
        with np.load(OUT/'weighted_classical'/pair/f'seed{seed}'/'videos'/video/'coarse_scores.npz') as a:scores['COVER_JOINT']=a['JOINT']
        lookup={int(t):i for i,t in enumerate(tracks)};columns=np.asarray([lookup[int(t)] for _,t in obs]);exclusive=exclusive_group(texts,target)
        for name in METHODS:
            mask=plain(scores[name][:,columns],exclusive);output=root/name/'videos'/video;output.mkdir(parents=True,exist_ok=True)
            np.save(output/'mask.npy',mask);immediate_predictions(root/name,type('Context',(),dict(bank=bank,queries=queries))(),mask)
        local=root/'videos'/video;local.mkdir(parents=True,exist_ok=True);np.savez_compressed(local/'coarse_scores.npz',tracks=tracks,**scores)
        receipt=dict(video=video,index=vi,fits=fits,teacher_examples=len(y),current_teacher_examples=int(known.sum()),
            GT_open_attempts=len(guard),confidence_weight_rule="1+3*abs(2*raw_teacher_p-1)",calibration=cal_receipt,joint_calibration=joint_receipt,
            no_source_logit_in_training_or_prediction=True,wall_seconds=time.perf_counter()-stamp)
        receipts.append(receipt);atomic_json(local/'receipt.json',receipt)
        saved=dict(total=total,square=square,rows=rows,bx=bx,by=by,receipts=receipts,history=history,next_video=vi+1,implementation=implementation)
        temp=stream.with_suffix('.tmp');torch.save(saved,temp);temp.replace(stream)
        print(json.dumps(dict(event='COVERAGE_SEARCH_VIDEO',pair=pair,seed=seed,completed=vi+1,total=len(videos))),flush=True)
    methods=[]
    for name in METHODS:
        mask=np.zeros(m['candidate_query_rows'],bool)
        for video in videos:
            local=np.load(root/name/'videos'/video/'mask.npy')
            for q,e in enumerate(e for e in m['episodes'] if e['video_id']==video):mask[e['start']:e['stop']]=local[q]
        seal=seal_mask(root/name,m,mask,target=target,model=model,method='COVERAGE_SEARCH_'+name,GT_open_attempts=len(guard),extra=dict(
            seed=seed,source_logit_residual_disabled=True,neural_dimension=768,no_new_teacher_or_encoder_calls=True,
            current_video_transductive=True,no_future_teacher=True,implementation=implementation,quality_replay_not_cold_time=True))
        methods.append(dict(method=name,rows=seal['prediction_rows']))
    assert not guard
    atomic_json(root/'completion.json',dict(status='SEALED_COVERAGE_SEARCH_NO_RESIDUAL',pair=pair,seed=seed,methods=methods,receipts=receipts,
        GT_open_attempts=0,new_teacher_calls=0,new_encoder_calls=0,logical_teacher_calls=sum(map(len,by)),
        source_pool=str(source),teacher_evidence=str(EVIDENCE/pair),actual_head_fits=2*len(receipts),
        source_logit_residual_disabled=True,implementation=implementation,wall_seconds=time.perf_counter()-begin,
        physical_GPU=__import__('os').environ.get('CUDA_VISIBLE_DEVICES'),goal_status='ACTIVE'))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--pair',choices=['C4'],required=True);parser.add_argument('--seed',type=int,required=True)
    args=parser.parse_args();run(args.pair,args.seed)
