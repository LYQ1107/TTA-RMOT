"""GT-free replay of canonical heads and the corresponding deployment glue."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.special import logit
from flex_tta.runtime.pf_receipts import atomic_json
from flex_uda.a100.mechanism import generation_guard,metadata,PAIRS
from flex_uda.active.context_cohort import distillation_features
from flex_uda.active.grounded_graph import query_geometry,positive_agreement_precision
from flex_uda.active.visual_classical import refine_distilled_prior
from flex_uda.active.acquisition import exclusive_group
from scripts.a100_visual_classical_canonical_pcd import CANON
from scripts.a100_visual_classical_canonical_pcd_launch import conditions
from scripts.a100_visual_classical_rgb_parity import source_spec
from scripts.a100_visual_classical_strict_prefix import plain


def deployment_graph(unary,pcd,colours,texts,answers,kind,precision,hidden=None,graph=True):
    # Mirrors cold_full_c4's explicit flags and precision routing. The study
    # uses refine_queries instead, so this checks the integration boundary.
    parts,geometry=query_geometry(pcd,colours,texts,kind,hidden)
    score=unary.copy()
    for qi,part in enumerate(parts):
        selected=part>0
        if selected.any():
            score[qi]=refine_distilled_prior(score[qi],geometry[qi],answers[qi],graph=graph,
                source_weight=precision[qi],node_mask=np.all(colours[:,selected].sum(2)>0,axis=1))
    return score


def main():
    records=conditions()
    assert all((Path(r['folder'])/'completion.json').is_file() for r in records)
    import torch
    from torch import nn
    from torch.nn import functional as F
    guard=generation_guard();torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    checks=[];graph_checks=0;graph_changed=0;probability_error=0.;head_error=0.
    for record in records:
        pair,seed=record['pair'],record['seed'];root=CANON/pair/f'seed{seed}'
        target=PAIRS[pair][1];m=metadata(target);videos=sorted(m['videos']);source=source_spec(pair)[0]
        protocol='PRESUPPOSED' if pair=='C4' else 'CHECKLIST'
        for video in videos:
            bank=m['videos'][video];obs=np.asarray(bank['observations'])
            texts=[e['query_text'] for e in m['episodes'] if e['video_id']==video]
            with np.load(source/'source'/video/'sparse_inputs.npz') as a:
                pcd,colours,prior,tracks=a['pcd'],a['colours'],a['prior'],a['tracks']
            raw=distillation_features(pcd,colours,texts,'pcd');assert np.array_equal(raw,pcd)
            lookup={int(t):i for i,t in enumerate(tracks)}
            columns=np.asarray([lookup[int(t)] for _,t in obs])
            first=np.asarray([np.flatnonzero(columns==i)[0] for i in range(len(tracks))])
            saved={name:np.load(root/name/'videos'/video/'scores.npy')[:,first]
                for name in ('KD400','KD_FINAL','KD_MEAN')}
            answers=np.full(prior.shape,np.nan)
            for answer in json.loads((source/'source'/video/'answers.json').read_text())['answers']:
                answers[int(answer['local_query']),int(answer['local_track'])]=answer['yes_probability']
            hidden=None
            if video==videos[-1]:
                for label,name in [('SOFT','KD_FINAL'),('HARD','HARD_FINAL')]:
                    checkpoint=torch.load(root/(label+'_latest_head.pt'),map_location='cpu',weights_only=False)
                    model=nn.Sequential(nn.Linear(768,64),nn.GELU(),nn.Linear(64,1))
                    model.load_state_dict(checkpoint['model']);model.to('cuda:0').eval()
                    values=torch.tensor((raw.reshape(-1,768)-checkpoint['mean'])/checkpoint['std'],dtype=torch.float32,device='cuda:0')
                    z=torch.tensor(np.clip(logit(np.clip(prior,.05,.95)),-2,2).ravel(),dtype=torch.float32,device='cuda:0')
                    with torch.no_grad():
                        probability=torch.sigmoid(z+model(values).squeeze(-1)).cpu().numpy().reshape(prior.shape)
                        if label=='SOFT':hidden=F.gelu(model[0](values)).cpu().numpy().reshape(prior.shape+(64,))
                    expected=np.load(root/name/'videos'/video/'scores.npy')[:,first]
                    error=float(np.max(np.abs(probability-expected)));head_error=max(head_error,error)
                    changed=int(np.count_nonzero(plain(probability[:,columns],exclusive_group(texts,target))!=np.load(root/name/'videos'/video/'mask.npy')))
                    assert error<1e-6 and changed==0,(pair,seed,label,error,changed)
                    checks.append(dict(pair=pair,seed=seed,label=label,video=video,maximum_probability_error=error,changed_candidate_rows=changed))
                    del model,values,z
            precision=positive_agreement_precision(saved['KD400'],saved['KD_FINAL'],protocol)
            for name,kind,use_guard,use_graph in [('GRAPH_PCD','pcd',False,True),('GRAPH_RGB','combined',False,True),
                    ('GRAPH_COLOUR','colour',False,True),('ANCHORS','pcd',False,False),('GRAPH_GUARD','pcd',True,True),
                    *([('GRAPH_HIDDEN','hidden',False,True)] if hidden is not None else [])]:
                weight=precision if use_guard else np.full(prior.shape,.1)
                value=deployment_graph(saved['KD_MEAN'],pcd,colours,texts,answers,kind,weight,hidden,use_graph)
                expected=np.load(root/name/'videos'/video/'scores.npy')[:,first]
                probability_error=max(probability_error,float(np.max(np.abs(value-expected))))
                changed=int(np.count_nonzero(plain(value[:,columns],exclusive_group(texts,target))!=np.load(root/name/'videos'/video/'mask.npy')))
                assert np.array_equal(value,expected) and changed==0,(pair,seed,video,name,changed)
                graph_changed+=changed;graph_checks+=1
    assert not guard
    result=dict(status='PASS_CANONICAL_DEPLOYMENT_NUMERICAL_REPLAY',saved_head_checks=checks,
        graph_video_method_checks=graph_checks,graph_changed_candidate_rows=graph_changed,
        maximum_graph_probability_error=probability_error,maximum_head_probability_error=head_error,
        GT_open_attempts=0,new_teacher_or_source_calls=0,new_head_updates=0,
        actual_cold_worker_executed=False,not_a_new_quality_or_cold_cost_measurement=True)
    atomic_json(ROOT/'docs/a100_visual_classical_20261121/canonical_deployment_replay.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
