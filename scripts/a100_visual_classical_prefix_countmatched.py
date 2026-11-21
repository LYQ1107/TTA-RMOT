"""GT-free quantity controls for causal exemplar inference vs strong KD."""
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

SOURCES={
    'FULL':('exemplar_prefix','COVER__SUPPORT_MIX'),
    'KERNEL':('exemplar_prefix','COVER__KERNEL'),
    'MEAN':('exemplar_prefix','COVER__MEAN'),
    'NO_MEMORY':('exemplar_prefix','COVER__SUPPORT_GATED'),
    'KD_TRACK':('strict_prefix','COVER__TRACK_SOFT'),
    'KD_FRAME':('strict_prefix','COVER__FRAME_SOFT'),
    'KD_STRUCT':('strict_prefix','COVER__FRAME_STRUCTURED'),
    'ENSEMBLE':('strict_prefix','COVER__ENSEMBLE')}
COMPARISONS=[(rank,'FULL') for rank in SOURCES]+[('FULL','KD_TRACK'),('KD_TRACK','KD_TRACK'),('FULL','KD_STRUCT'),('KD_STRUCT','KD_STRUCT')]


def run(pair,family='exemplar'):
    sources=SOURCES;comparisons=COMPARISONS;group='prefix_countmatched';dependencies=('strict_prefix','exemplar_prefix')
    if family in ('colour','facility'):
        base_group='colour_prefix' if family=='colour' else 'facility_prefix'
        policy='COVER' if family=='colour' else 'FACILITY'
        sources={name:(base_group,policy+'__'+method) for name,method in {
            'FULL':'COHORT','KD':'RGB_KD','NO_GRAPH':'NO_GRAPH','PCD':'PCD_COHORT','COLOR_ONLY':'COLOR_ONLY','STRONG_PRIOR':'STRONG_PRIOR'}.items()}
        comparisons=[(rank,'FULL') for rank in sources]+[('FULL','KD'),('KD','KD')]
        group=family+'_countmatched';dependencies=(base_group,)
    gt=generation_guard();begin=time.perf_counter();model,target=PAIRS[pair];m=metadata(target);root=OUT/group/pair
    if (root/'completion.json').is_file():return
    for dependency in dependencies:
        while not (OUT/dependency/pair/'completion.json').is_file():time.sleep(5)
    masks={key:np.zeros(m['candidate_query_rows'],dtype=bool) for key in comparisons}
    for video in sorted(m['videos']):
        fc=FrozenInputs(pair,video);score={};counts={}
        for name,(source_group,method) in sources.items():
            path=OUT/source_group/pair/method/'videos'/video;score[name]=np.load(path/'scores.npy')
            if name in {count for _,count in comparisons}:counts[name]=np.load(path/'mask.npy')
        frames=[np.flatnonzero(fc.observations[:,0]==f) for f in np.unique(fc.observations[:,0])]
        for (rank,count),output in masks.items():
            local=np.zeros(fc.margin.shape,dtype=bool)
            for ids in frames:
                for qi,q in enumerate(fc.queries):
                    k=int(counts[count][qi,ids].sum())
                    order=np.lexsort((fc.observations[ids,1],-fc.margin[qi,ids],-score[rank][qi,ids]))[:k]
                    local[qi,ids[order]]=True
                    assert int(local[qi,ids].sum())==k
            for qi,q in enumerate(fc.queries):output[q['start']:q['stop']]=local[qi]
        print(json.dumps(dict(event='PREFIX_COUNTMATCHED_VIDEO',pair=pair,video=video)),flush=True)
    methods=[]
    for (rank,count),mask in masks.items():
        name=rank+'__AT_'+count+'_COUNT'
        seal=seal_mask(root/name,m,mask,target=target,model=model,method=name,GT_open_attempts=len(gt),extra=dict(
            rank_method=rank,count_method=count,counts_from_GT=False,counts_equal_per_query_frame=True,
            same_teacher_answers=True,teacher_acquisition='FACILITY' if family=='facility' else 'COVER',
            source_predictions_from_exact_causal_prefixes=True,
            common_projection='per-query descending score, descending raw source margin, ascending original ID',
            exclusivity_not_reapplied=True,CountSelf_may_differ_due_to_independent_query_projection=True))
        methods.append(dict(method=name,rows=seal['prediction_rows']))
    assert not gt
    atomic_json(root/'completion.json',dict(status='SEALED_CAUSAL_COUNTMATCHED',pair=pair,methods=methods,
        GT_open_attempts=0,wall_seconds=time.perf_counter()-begin,no_target_GT_count_used=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pair',required=True,choices=['C4','C1']);p.add_argument('--family',choices=['exemplar','colour','facility'],default='exemplar')
    a=p.parse_args();run(a.pair,a.family)
