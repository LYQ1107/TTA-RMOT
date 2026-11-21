"""GT-free temporal controls on already trained teacher heads, full videos."""
import argparse
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.special import expit
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from flex_uda.a100.mechanism import PAIRS,metadata,FrozenInputs,generation_guard
from flex_uda.active.semantic_memory import blend_memory,categorical_smooth,binary_smooth,ema
from flex_uda.active.acquisition import exclusive_group,expression_types
from flex_uda.presence.predictions import seal_mask
from scripts.a100_visual_classical_prepare import OUT

MODES=('MEAN_ENSEMBLE','FRAME_EMA','FRAME_HMM','MEMORY_HMM','RESET_MEMORY_HMM','RESET_NO_HMM')


def plain_decode(score,exclusive):
    mask=score>.5
    if exclusive:mask &= np.arange(len(score))[:,None]==score.argmax(0)[None]
    return mask


def run(pair):
    opened=generation_guard();begin=time.perf_counter();model,target=PAIRS[pair];m=metadata(target)
    root=OUT/'semantic_memory_v3'/pair
    if (root/'completion.json').is_file():return
    source=OUT/'predictions'/pair/'COVER__FRAME_KD_SOFT'
    memory_source=OUT/'convergence'/pair/'COVER__KD_SOFT_4000'
    assert (source/'predictions_seal.json').is_file() and (memory_source/'predictions_seal.json').is_file()
    masks={mode:np.zeros(m['candidate_query_rows'],dtype=bool) for mode in MODES};timings=[]
    for video in sorted(m['videos']):
        started=time.perf_counter();fc=FrozenInputs(pair,video)
        frame=np.load(source/'videos'/video/'scores.npy')
        with np.load(memory_source/'videos'/video/'belief.npz') as a:prior=a['score']
        exclusive=exclusive_group([q['query_text'] for q in fc.queries],target)
        hazards=np.asarray([.1 if any(t in expression_types(q['query_text']) for t in ('MOTION','SPATIAL')) else .02 for q in fc.queries])
        score={mode:np.zeros(fc.margin.shape,dtype=float) for mode in MODES}
        local={mode:np.zeros(fc.margin.shape,dtype=bool) for mode in MODES}
        for ti,track in enumerate(fc.tracks):
            ids=fc.by_track[track]
            cuts=np.r_[0,np.flatnonzero(np.diff(fc.observations[ids,0])>2)+1,len(ids)]
            for start,stop in zip(cuts[:-1],cuts[1:]):
                indices=ids[start:stop]
                p=frame[:,indices].T;memory=np.broadcast_to(prior[:,ti],p.shape)
                mixed=blend_memory(p,memory,exclusive=exclusive)
                reset=blend_memory(p,memory,reset=True,exclusive=exclusive)
                def smooth(value):
                    return categorical_smooth(value,.02) if exclusive else binary_smooth(value,hazards)
                values={'MEAN_ENSEMBLE':mixed,'FRAME_EMA':ema(p),'RESET_NO_HMM':reset}
                for mode,value in values.items():
                    score[mode][:,indices]=value.T;local[mode][:,indices]=plain_decode(value.T,exclusive)
                for mode,value in [('FRAME_HMM',p),('MEMORY_HMM',mixed),('RESET_MEMORY_HMM',reset)]:
                    probability,bits=smooth(value)
                    score[mode][:,indices]=probability.T;local[mode][:,indices]=bits.T
        for qi,q in enumerate(fc.queries):
            if 'MOTION' in expression_types(q['query_text']):
                presence=expit(fc.raw_presence[qi]);gate=np.asarray([presence[fc.frame_index[int(f)]]>=.5 for f,_ in fc.observations])
                for mode in MODES:local[mode][qi]&=gate
        for mode in MODES:
            folder=root/mode/'videos'/video;folder.mkdir(parents=True,exist_ok=True)
            np.save(folder/'mask.npy',local[mode]);np.save(folder/'scores.npy',score[mode].astype('float32'))
            for qi,q in enumerate(fc.queries):masks[mode][q['start']:q['stop']]=local[mode][qi]
        timings.append(dict(video=video,wall_seconds=time.perf_counter()-started))
        print(json.dumps(dict(event='V3_MEMORY_VIDEO',pair=pair,video=video)),flush=True)
    records=[]
    for mode,mask in masks.items():
        seal=seal_mask(root/mode,m,mask,target=target,model=model,method='V3_'+mode,GT_open_attempts=len(opened),
            extra=dict(acquisition='COVER',logical_teacher_calls=3508 if pair=='C4' else 3334,new_teacher_calls=0,new_optimizer_steps=0,
                frame_head=str(source/'head.pt'),track_head=str(memory_source/'head.pt'),
                frame_head_sha256=file_sha(source/'head.pt'),track_head_sha256=file_sha(memory_source/'head.pt'),
                same_two_heads_in_mean_ensemble_control=True,hazard_static=.02,hazard_dynamic=.1,
                ema_alpha=.2,logit_cap=6.,reset_positive=.85,reset_negative=.15,
                exclusive_categorical_with_none=exclusive_group([q['query_text'] for q in fc.queries],target),
                full_video_forward_backward=True,not_strict_frame_causal=True,
                no_forced_frame_presence=True,phase_timings=timings))
        records.append(dict(method=mode,rows=seal['prediction_rows']))
    assert not opened
    atomic_json(root/'completion.json',dict(status='SEALED_V3_MEMORY_CONTROLS',pair=pair,methods=records,
        GT_open_attempts=0,wall_seconds=time.perf_counter()-begin))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pair',choices=['C4','C1'],required=True);a=p.parse_args();run(a.pair)
