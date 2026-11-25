"""Teacher-fit convergence control, registered without viewing target metrics."""
import argparse
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from flex_uda.a100.mechanism import FrozenInputs,metadata,PAIRS,generation_guard
from flex_uda.presence.predictions import seal_mask
from scripts.a100_visual_classical_prepare import OUT
from scripts.a100_visual_classical_generate import distill,decode,ACQUISITIONS


def run(pair):
    attempts=generation_guard();begin=time.perf_counter();model,target=PAIRS[pair];m=metadata(target)
    root=OUT/'convergence'/pair
    if (root/'completion.json').is_file():return
    evidence=OUT/'evidence'/pair;assert (evidence/'completion.json').is_file()
    plan=json.loads((evidence/'acquisition_plan.json').read_text())['positions']
    responses={(r['video_id'],r['local_query'],r['local_track']):r for r in map(json.loads,(evidence/'responses.jsonl').read_text().splitlines())}
    inputs={}
    for video in sorted(m['videos']):
        with np.load(OUT/'inputs'/pair/video/'inputs.npz') as a:inputs[video]={k:a[k].copy() for k in a.files}
    records=[]
    for policy in ACQUISITIONS:
        answers={v:np.full(x['prior'].shape,np.nan) for v,x in inputs.items()}
        for v in inputs:
            for q,t in plan[v][policy]:answers[v][q,t]=responses[v,q,t]['yes_probability']
        for label in ('HARD','SOFT'):
            name=policy+'__KD_'+label+'_4000';dest=root/name
            scores,training=distill(inputs,answers,dest,soft=label=='SOFT',steps=4000)
            mask=np.zeros(m['candidate_query_rows'],dtype=bool)
            for video,score in scores.items():
                fc=FrozenInputs(pair,video);local=decode(score,fc)
                folder=dest/'videos'/video;folder.mkdir(parents=True,exist_ok=True)
                np.save(folder/'mask.npy',local);np.savez_compressed(folder/'belief.npz',score=score)
                for q,query in enumerate(fc.queries):mask[query['start']:query['stop']]=local[q]
            calls=sum(len(p[policy]) for p in plan.values())
            seal=seal_mask(dest,m,mask,target=target,model=model,method=name,GT_open_attempts=len(attempts),
                extra=dict(logical_teacher_calls=calls,training=training,acquisition=policy,
                    strengthening_control_not_a_new_proposed_method=True,
                    registration='preregistration_baseline_convergence.json',
                    teacher_evidence_sha256=file_sha(evidence/'responses.jsonl')))
            records.append(dict(method=name,rows=seal['prediction_rows'],calls=calls))
            print(json.dumps(dict(event='LONG_KD_SEALED',pair=pair,method=name,teacher_loss=training['final_full_teacher_loss'])),flush=True)
    assert not attempts
    atomic_json(root/'completion.json',dict(status='SEALED_CONVERGENCE_BASELINES',pair=pair,methods=records,
        GT_open_attempts=0,wall_seconds=time.perf_counter()-begin))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pair',choices=['C4','C1'],required=True);a=p.parse_args();run(a.pair)
