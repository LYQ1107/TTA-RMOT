"""Offline diagnostic only: keep every selected box, replace matched IDs by GT.

This is not a deployable algorithm, a pure semantic metric, or a strict upper
bound for HOTA at every IoU threshold. No oracle artifact is an adaptation input.
"""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from flex_tta.runtime.pf_receipts import atomic_json
from flex_uda.a100.mechanism import PAIRS,metadata,OUT as DIAG
from flex_uda.presence.predictions import seal_mask
from scripts.a100_visual_classical_prepare import OUT
from scripts.a100_mechanism_offline import evaluate


def run(pair):
    model,target=PAIRS[pair];meta=metadata(target);altered=dict(meta,videos={})
    source=OUT/'convergence'/pair/'COVER__KD_SOFT_4000';root=OUT/'identity_diagnosis'/pair/'SAME_MASK_GT_MATCHED_IDS'
    mask=np.zeros(meta['candidate_query_rows'],dtype=bool)
    for video,bank in meta['videos'].items():
        ids=np.load(DIAG/'offline_GT'/pair/video/'union_ids.npy')
        original=np.asarray(bank['observations'])[:,1]
        offset=int(max(ids.max(initial=0),original.max(initial=0)))+1
        replacement=np.where(ids>=0,ids,original+offset)
        lines=[]
        for line,tid in zip(bank['lines'],replacement):
            fields=line.split(',');fields[1]=str(int(tid));lines.append(','.join(fields))
        altered['videos'][video]=dict(bank,lines=lines)
        selection=np.load(source/'videos'/video/'mask.npy')
        for q,query in enumerate([q for q in meta['episodes'] if q['video_id']==video]):
            mask[query['start']:query['stop']]=selection[q]
    receipt=seal_mask(root,altered,mask,target=target,model=model,method='OFFLINE_GT_ID_REPLACEMENT',
        GT_open_attempts=len(meta['videos']),extra=dict(offline_oracle_uses_GT=True,selection_reference=str(source),
        original_boxes_and_selection_preserved=True,only_identity_fields_changed=True,
        GT_matching='existing union one-to-one IoU>=.5',not_strict_theoretical_upper_bound=True))
    receipt['status']='OFFLINE_ORACLE_USES_GT';atomic_json(root/'predictions_seal.json',receipt)
    result=evaluate(root,pair,oracle=True);atomic_json(OUT/'identity_diagnosis'/pair/'completion.json',result)
    print(json.dumps(dict(pair=pair,diagnostic_HOTA=result['HOTA_unrounded'],not_a_deployable_result=True)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pair',required=True,choices=['C4','C1']);a=p.parse_args();run(a.pair)
