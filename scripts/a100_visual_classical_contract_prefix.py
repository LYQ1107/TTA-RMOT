"""Same task-contract decoder intervention for every already-frozen baseline."""
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from flex_tta.runtime.pf_receipts import atomic_json
from flex_uda.a100.mechanism import PAIRS,metadata,generation_guard
from flex_uda.active.task_decode import nonempty_static
from flex_uda.presence.predictions import seal_mask
from scripts.a100_visual_classical_strict_prefix import read_guard,immediate_predictions
from scripts.a100_visual_classical_strong_prefix import METHODS
from scripts.a100_visual_classical_prepare import OUT


def run(pair,seed):
    guard=generation_guard();visible=set();forbidden,reads=read_guard(pair,visible)
    begin=time.perf_counter();base='strong_cover_seed'+str(seed);group=base+'_contract';root=OUT/group/pair
    if (root/'completion.json').is_file():return
    while not (OUT/base/pair/'completion.json').is_file():time.sleep(10)
    source=json.loads((OUT/base/pair/'completion.json').read_text());assert source['GT_open_attempts']==source['forbidden_read_attempts']==0
    protocol=json.loads((OUT/'preregistration_v1.json').read_text())['protocol'][pair]
    model,target=PAIRS[pair];m=metadata(target);receipts=[]
    for vi,video in enumerate(sorted(m['videos'])):
        visible.add(video);bank=m['videos'][video];queries=[q for q in m['episodes'] if q['video_id']==video]
        assert source['receipts'][vi]['video']==video and source['receipts'][vi]['visible_teacher_videos']==sorted(visible)
        observations=np.asarray(bank['observations']);fc=SimpleNamespace(bank=bank,queries=queries);changes={};times={}
        for method in METHODS:
            local=OUT/base/pair/method/'videos'/video;mask=np.load(local/'mask.npy');score=np.load(local/'scores.npy')
            start=time.perf_counter();updated,record=nonempty_static(score,mask,observations,queries,protocol);times[method]=time.perf_counter()-start
            if protocol=='CHECKLIST':assert np.array_equal(updated,mask)
            folder=root/method;dest=folder/'videos'/video;dest.mkdir(parents=True,exist_ok=True)
            np.save(dest/'mask.npy',updated)
            if not (dest/'scores.npy').exists():os.link(local/'scores.npy',dest/'scores.npy')
            immediate_predictions(folder,fc,updated);changes[method]=record
        receipts.append(dict(video=video,visible_teacher_videos=sorted(visible),protocol=protocol,changes=changes,
            decoder_only_seconds=times,GT_open_attempts=len(guard)))
        print(json.dumps(dict(event='TASK_CONTRACT_DECODER_VIDEO',seed=seed,pair=pair,video=video,changed_queries={k:len(v) for k,v in changes.items()})),flush=True)
    methods=[]
    for method in METHODS:
        mask=np.zeros(m['candidate_query_rows'],dtype=bool);folder=root/method
        for video in sorted(m['videos']):
            local=np.load(folder/'videos'/video/'mask.npy')
            for qi,q in enumerate([q for q in m['episodes'] if q['video_id']==video]):mask[q['start']:q['stop']]=local[qi]
        seal=seal_mask(folder,m,mask,target=target,model=model,method=group+'__'+method,GT_open_attempts=len(guard),extra=dict(
            teacher_acquisition='COVER',seed=seed,acquisition_seed=seed,logical_teacher_calls=source['receipts'][-1]['teacher_examples'],
            explicit_task_contract=protocol,contract_source='original preregistration before this goal evidence collection',
            same_decoder_given_to_every_KD_and_classic_arm=True,new_teacher_calls=0,new_neural_updates=0,
            scores_and_ranking_exactly_unchanged=True,decoder_only_quantity_intervention=True,
            video_stream_causal=True,current_video_transductive=True,no_future_teacher_feedback=True))
        methods.append(dict(method=method,rows=seal['prediction_rows']))
    assert not guard and not forbidden
    atomic_json(root/'completion.json',dict(status='SEALED_EXPLICIT_TASK_DECODER',pair=pair,seed=seed,source_group=base,
        methods=methods,GT_open_attempts=0,forbidden_read_attempts=0,receipts=receipts,
        wall_seconds=time.perf_counter()-begin,task_contract_not_inferred_from_GT=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pair',choices=['C4','C1'],required=True);p.add_argument('--seed',type=int,default=0)
    a=p.parse_args();run(a.pair,a.seed)
