"""V2 follows registered development diagnosis; no GT is a runtime input."""
import argparse
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from flex_uda.a100.mechanism import PAIRS,metadata,FrozenInputs,generation_guard
from flex_uda.active.visual_classical import refine_distilled_prior
from flex_uda.presence.predictions import seal_mask
from scripts.a100_visual_classical_prepare import OUT
from scripts.a100_visual_classical_generate import ACQUISITIONS,decode

MODES=('HIDDEN_GRAPH','RAW_GRAPH','NO_GRAPH')


def run(pair):
    attempts=generation_guard();begin=time.perf_counter();model,target=PAIRS[pair];m=metadata(target)
    import torch
    from torch import nn
    torch.set_num_threads(2)
    root=OUT/'refinement_v2'/pair
    if (root/'completion.json').is_file():return
    assert (OUT/'convergence'/pair/'completion.json').is_file()
    plan=json.loads((OUT/'evidence'/pair/'acquisition_plan.json').read_text())['positions']
    responses={(r['video_id'],r['local_query'],r['local_track']):r for r in map(json.loads,(OUT/'evidence'/pair/'responses.jsonl').read_text().splitlines())}
    records=[]
    for policy in ACQUISITIONS:
        base=OUT/'convergence'/pair/(policy+'__KD_SOFT_4000')
        saved=torch.load(base/'head.pt',map_location='cpu',weights_only=False)
        projection=nn.Sequential(nn.Linear(768,64),nn.GELU())
        projection[0].load_state_dict(dict(weight=saved['model']['0.weight'],bias=saved['model']['0.bias']))
        projection.eval()
        masks={mode:np.zeros(m['candidate_query_rows'],dtype=bool) for mode in MODES};timings=[]
        for video in sorted(m['videos']):
            fc=FrozenInputs(pair,video)
            with np.load(OUT/'inputs'/pair/video/'inputs.npz') as a:features=a['features']
            with np.load(base/'videos'/video/'belief.npz') as a:prior=a['score']
            teacher=np.full(prior.shape,np.nan)
            for q,t in plan[video][policy]:teacher[q,t]=responses[video,q,t]['yes_probability']
            with torch.no_grad():
                x=torch.from_numpy((features.reshape(-1,768)-saved['mean'])/saved['std'])
                hidden=projection(x).numpy().reshape(features.shape[:2]+(64,))
            for mode in MODES:
                start=time.perf_counter();geometry=hidden if mode=='HIDDEN_GRAPH' else features
                score=np.stack([refine_distilled_prior(prior[q],geometry[q],teacher[q],graph=mode!='NO_GRAPH') for q in range(len(prior))])
                local=decode(score,fc);timings.append(dict(video=video,mode=mode,wall_seconds=time.perf_counter()-start))
                folder=root/(policy+'__'+mode)/'videos'/video;folder.mkdir(parents=True,exist_ok=True)
                np.save(folder/'mask.npy',local);np.savez_compressed(folder/'belief.npz',score=score,seen=np.isfinite(teacher))
                for q,query in enumerate(fc.queries):masks[mode][query['start']:query['stop']]=local[q]
            print(json.dumps(dict(event='V2_REFINEMENT_VIDEO',pair=pair,policy=policy,video=video)),flush=True)
        for mode,mask in masks.items():
            name=policy+'__'+mode;calls=sum(len(p[policy]) for p in plan.values())
            seal=seal_mask(root/name,m,mask,target=target,model=model,method='V2_'+name,GT_open_attempts=len(attempts),
                extra=dict(acquisition=policy,seed=0,logical_teacher_calls=calls,new_teacher_calls=0,new_optimizer_steps=0,
                    shared_distilled_head=str(base/'head.pt'),shared_head_sha256=file_sha(base/'head.pt'),
                    source_precision_unasked=1.,source_precision_asked=0.,teacher_precision='1+3*abs(2*p-1)',
                    logit_cap=4.,graph_neighbours=3,graph_strength=1.,geometry=mode,
                    primary_variant=mode=='HIDDEN_GRAPH',development_revision_after_V1=True,
                    inference_phase_timings=[r for r in timings if r['mode']==mode]))
            records.append(dict(method=name,rows=seal['prediction_rows'],calls=calls))
    assert not attempts
    atomic_json(root/'completion.json',dict(status='SEALED_V2_ALL_VARIANTS',pair=pair,methods=records,
        GT_open_attempts=0,wall_seconds=time.perf_counter()-begin))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pair',choices=['C4','C1'],required=True);a=p.parse_args();run(a.pair)
