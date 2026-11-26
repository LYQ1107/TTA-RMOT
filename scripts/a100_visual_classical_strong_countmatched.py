"""Post-seal quantity controls; all counts come from the methods, never GT."""
import argparse
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from flex_tta.runtime.pf_receipts import atomic_json
from flex_uda.a100.mechanism import PAIRS,metadata,FrozenInputs,generation_guard
from flex_uda.presence.predictions import seal_mask
from scripts.a100_visual_classical_prepare import OUT
from scripts.a100_visual_classical_strong_prefix import METHODS
from flex_uda.active.acquisition import exclusive_group
from scipy.optimize import linear_sum_assignment


def run(pair,group,exclusive_projection=False):
    gt=generation_guard();begin=time.perf_counter();model,target=PAIRS[pair];m=metadata(target)
    root=OUT/(group+('_counts_exclusive' if exclusive_projection else '_counts'))/pair
    if (root/'completion.json').is_file():return
    while not (OUT/group/pair/'completion.json').is_file():time.sleep(10)
    source=json.loads((OUT/group/pair/'completion.json').read_text())
    assert source['GT_open_attempts']==source['forbidden_read_attempts']==0
    comparisons=[(rank,'COHORT') for rank in METHODS]
    count_methods=('SOFT_KD','HARD_KD','SOFT_KD_400','HARD_KD_400') if exclusive_projection else ('SOFT_KD','HARD_KD')
    comparisons += [(rank,count) for count in count_methods for rank in ('COHORT',count)]
    masks={key:np.zeros(m['candidate_query_rows'],dtype=bool) for key in comparisons}
    for video in sorted(m['videos']):
        fc=FrozenInputs(pair,video);scores={};counts={}
        for method in METHODS:
            folder=OUT/group/pair/method/'videos'/video
            scores[method]=np.load(folder/'scores.npy')
            if method in ('COHORT',*count_methods):counts[method]=np.load(folder/'mask.npy')
        exclusive=exclusive_projection and exclusive_group([q['query_text'] for q in fc.queries],target)
        for ids in (np.flatnonzero(fc.observations[:,0]==f) for f in np.unique(fc.observations[:,0])):
            if exclusive:
                for (rank,count),mask in masks.items():
                    assert np.all(counts[count][:,ids].sum(0)<=1)
                    cardinality=counts[count][:,ids].sum(1).astype(int)
                    slots=np.repeat(np.arange(len(fc.queries)),cardinality)
                    assert len(slots)<=len(ids)
                    if not len(slots):continue
                    # Same linear assignment for every compared score. Exactly
                    # one detection per slot, at most one query per detection.
                    slot_ids,candidates=linear_sum_assignment(-scores[rank][slots][:,ids].astype(float))
                    chosen_queries=slots[slot_ids]
                    for qi,q in enumerate(fc.queries):
                        chosen=ids[candidates[chosen_queries==qi]]
                        assert len(chosen)==cardinality[qi]
                        mask[q['start']+chosen]=True
            else:
                for qi,q in enumerate(fc.queries):
                    orders={rank:np.lexsort((fc.observations[ids,1],-fc.margin[qi,ids],-scores[rank][qi,ids])) for rank in METHODS}
                    for (rank,count),mask in masks.items():
                        k=int(counts[count][qi,ids].sum());chosen=ids[orders[rank][:k]]
                        mask[q['start']+chosen]=True
                        assert len(chosen)==k
        print(json.dumps(dict(event='STRONG_COUNTMATCHED_VIDEO',pair=pair,group=group,video=video)),flush=True)
    methods=[]
    for (rank,count),mask in masks.items():
        name=rank+'__AT_'+count+'_COUNT'
        seal=seal_mask(root/name,m,mask,target=target,model=model,method=name,GT_open_attempts=len(gt),extra=dict(
            rank_method=rank,count_method=count,source_group=group,counts_from_GT=False,
            counts_equal_per_query_frame=True,same_teacher_answers=True,same_prefix_information=True,
            common_projection={'exclusive_task':'joint maximum-sum-score assignment to fixed query-count slots; deterministic ties',
                'nonexclusive_task':'descending method score, descending source margin, ascending original ID'} if exclusive_projection else 'descending method score, descending source margin, ascending original ID',
            exclusive_task_constraint_preserved=exclusive_projection,
            exclusivity_not_reapplied=not exclusive_projection,CountSelf_may_differ_due_to_common_projection=True))
        methods.append(dict(method=name,rows=seal['prediction_rows']))
    assert not gt
    atomic_json(root/'completion.json',dict(status='SEALED_STRONG_KD_COUNTMATCHED',pair=pair,methods=methods,
        source_group=group,exclusive_projection=exclusive_projection,GT_open_attempts=0,wall_seconds=time.perf_counter()-begin))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pair',required=True,choices=['C4','C1']);p.add_argument('--group',required=True)
    p.add_argument('--exclusive-projection',action='store_true')
    a=p.parse_args();run(a.pair,a.group,a.exclusive_projection)
