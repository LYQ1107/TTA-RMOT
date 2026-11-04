"""CPU-only real Stage-A prior replay; no GT, teacher, or model invocation."""
from argparse import ArgumentParser
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from flex_uda.presence.runtime import RUN,target_metadata,sealed_margins,runtime_guard
from flex_uda.active.v6 import selection_fractions,BeliefStateV6
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from scripts.active_run import PAIRS


def stage_arrays(stage_a,target,model,video,query_count):
    offset=0
    for directory in sorted((RUN/'features'/target/model/'videos').iterdir()):
        local=json.loads((directory/'metadata.json').read_text())
        if local['video_id']==video:break
        offset+=len(local['queries'])*len(local['frames'])
    else:raise ValueError('Missing video features '+video)
    frames=local['frames'];size=query_count*len(frames)
    folder=Path(stage_a)/target/model
    arrays={}
    for name in ('presence_logits','expected_iou_utilities','raw_k'):
        value=np.load(folder/(name+'.npy'),mmap_mode='r')[offset:offset+size]
        arrays[name]=value.reshape((query_count,len(frames))+value.shape[1:])
    return arrays,{int(t):i for i,t in enumerate(frames)}


def main():
    p=ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    attempts=runtime_guard();old=ROOT/'results/presence_active/redesign_v5/execution_map_v1_critical18.json'
    stage=Path(json.loads(old.read_text())['stage_A_root']);records=[]
    for pair,(model,target) in PAIRS.items():
        meta=target_metadata(target);margins,seal=sealed_margins(target,model)
        video=sorted(meta['videos'])[0];bank=meta['videos'][video]
        queries=[q for q in meta['episodes'] if q['video_id']==video]
        arrays,index=stage_arrays(stage,target,model,video,len(queries))
        q=queries[0];scores=np.asarray(margins[q['start']:q['stop']])[None]
        tids=sorted({t for _,t in bank['observations']});lengths=[sum(t==i for _,t in bank['observations']) for i in tids]
        f,prob,mask=selection_fractions(bank['observations'],tids,index,scores,
            arrays['expected_iou_utilities'][:1],arrays['presence_logits'][:1],raw_k=arrays['raw_k'][:1])
        prior=stage/target/model/'PRES_TOPK/predictions_seal.json'
        source=json.loads(prior.read_text());entry=next(e for e in source['episodes']
            if e['video']==video and e['expression']==q['expression_id'])
        assert file_sha(entry['prediction_path'])==entry['prediction_sha256']
        exact=''.join(line+'\n' for line,keep in zip(bank['lines'],mask[0]) if keep)
        assert exact==Path(entry['prediction_path']).read_text()
        means=np.stack([[scores[0,[j for j,(_,i) in enumerate(bank['observations']) if i==t]].mean() for t in tids]])
        state=BeliefStateV6(means,f,prob.max(1),track_lengths=lengths)
        assert (state.pseudo_labels()==-1).all() and np.abs(state.prior).max()<=2
        records.append(dict(pair=pair,video=video,expression=q['expression_id'],tracks=len(tids),
            observations=len(bank['observations']),selected_rows=int(mask.sum()),
            frozen_Stage_A_rows_exact=True,initial_unasked_pseudo_labels=0,
            prior_min=float(state.prior.min()),prior_max=float(state.prior.max()),
            original_prediction_sha256=entry['prediction_sha256'],margin_sha256=seal['sha256']))
    assert not attempts
    result=dict(status='SIX_REAL_FIRST_QUERY_STAGE_A_REPLAYS_PASSED',records=records,
                GPU_model_teacher_calls=0,GT_open_attempts=0,not_RMOT_evaluation=True)
    atomic_json(args.output,result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
