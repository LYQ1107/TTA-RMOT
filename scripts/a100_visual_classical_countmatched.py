"""Output-count controls generated exclusively from sealed method predictions."""
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


def generate(pair):
    attempts=generation_guard();begin=time.perf_counter();model,target=PAIRS[pair];m=metadata(target)
    root=OUT/'predictions'/pair
    assert (root/'completion.json').is_file()
    comparisons=[(method,'FULL') for method in ('FULL','KD_HARD','KD_SOFT','FRAME_KD_HARD','FRAME_KD_SOFT','DIRECT_SOFT','NO_CORRECTION','NO_GRAPH')]
    comparisons += [('FULL','KD_SOFT'),('KD_SOFT','KD_SOFT')]
    comparisons += [('FULL','FRAME_KD_SOFT'),('FRAME_KD_SOFT','FRAME_KD_SOFT')]
    names={(rank,count):rank+'__COUNT_'+count for rank,count in comparisons}
    masks={key:np.zeros(m['candidate_query_rows'],dtype=bool) for key in names}
    for video in sorted(m['videos']):
        fc=FrozenInputs(pair,video);track_index={t:i for i,t in enumerate(fc.tracks)}
        counts={count:np.load(root/('COVER__'+count)/'videos'/video/'mask.npy') for _,count in comparisons}
        scores={}
        for method in {r for r,_ in comparisons}:
            path=root/('COVER__'+method)/'videos'/video
            if method.startswith('FRAME_KD_'):scores[method]=np.load(path/'scores.npy')
            else:
                with np.load(path/'belief.npz') as a:scores[method]=a['score']
        for (rank,count),name in names.items():
            local=np.zeros(fc.margin.shape,dtype=bool)
            for frame in fc.frame_index:
                ids=np.flatnonzero(fc.observations[:,0]==frame)
                tids=np.asarray([track_index[int(fc.observations[i,1])] for i in ids])
                for q in range(len(fc.queries)):
                    k=int(counts[count][q,ids].sum())
                    # Source margin is a shared, GT-free deterministic tie rule.
                    values=scores[rank][q,ids] if rank.startswith('FRAME_KD_') else scores[rank][q,tids]
                    chosen=np.lexsort((fc.observations[ids,1],-fc.margin[q,ids],-values))[:k]
                    local[q,ids[chosen]]=True
                    assert int(local[q,ids].sum())==k
            for q,query in enumerate(fc.queries):masks[rank,count][query['start']:query['stop']]=local[q]
        print(json.dumps(dict(event='COUNTMATCHED_VIDEO',pair=pair,video=video)),flush=True)
    receipts=[]
    for key,name in names.items():
        seal=seal_mask(OUT/'countmatched'/pair/name,m,masks[key],target=target,model=model,method=name,
            GT_open_attempts=len(attempts),extra=dict(rank_method=key[0],count_method=key[1],
                acquisition='COVER',same_questions=True,counts_from_GT=False,
                tie_rule='posterior descending, source margin descending, original track id ascending',
                counts_equal_per_query_frame=True))
        receipts.append(dict(method=name,rows=seal['prediction_rows']))
    assert not attempts
    atomic_json(OUT/'countmatched'/pair/'completion.json',dict(status='SEALED_COUNT_CONTROLS',pair=pair,
        methods=receipts,GT_open_attempts=0,wall_seconds=time.perf_counter()-begin))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pair',choices=['C4','C1'],required=True);a=p.parse_args();generate(a.pair)
