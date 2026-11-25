"""All36 target decodes and historical X0 byte equality gate, GT-free."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.xquery_common import *

def generate(x0_only=False):
    setup();require_plan();attempts=v2.deny_gt();start=time.perf_counter();meta,scores,groups=inputs()
    theta=None if x0_only else json.loads((RUN/'thresholds.json').read_text())['models']
    wanted=DECODERS[:1] if x0_only else DECODERS;arrays={};folder=RUN/'masks';folder.mkdir(exist_ok=True)
    for m in MARGINS:
        for d in wanted:arrays[m,d]=np.lib.format.open_memmap(folder/(m+'__'+d+'.npy'),mode='w+',dtype='bool',shape=(len(groups),))
    for video,queries,tracks,frames,ids,unique,inverse in cs.video_banks(meta):
        for m in MARGINS:
            matrix=np.stack([scores[m][q['start']:q['stop']] for q in queries])
            if x0_only:
                means=np.stack([cs.track_mean(x,inverse,len(unique)) for x in matrix]);assignment=means.argmax(0)[inverse]
                masks={'X0_C3':np.arange(17)[:,None]==assignment}
            else:masks=decode(matrix,frames,inverse,theta[m]['threshold'])
            for j,q in enumerate(queries):
                for d in wanted:arrays[m,d][q['start']:q['stop']]=masks[d][j]
        print(json.dumps(dict(event='XQUERY_DECODE_VIDEO',video=video,x0_only=x0_only)),flush=True)
    lines=v2.candidate_lines(meta);records=[];checks=[]
    for (m,d),mask in arrays.items():
        mask.flush();name=m+'__'+d;seal=v2.write_selection(name,mask,meta,lines,protocol='FIXED_XQUERY_DECODE_NO_NEW_TRAINING__'+m)
        if d=='X0_C3' and m in cs.MARGINS:
            original=cs.RUN/'evaluation'/(m+'__C3_QUERY_ARGMAX_TRACK')
            for e in seal['episodes']:
                p=original/'results'/e['video']/e['expression']/'predict.txt'
                if file_sha(p)!=e['prediction_sha256']:raise ValueError('STOP: X0 does not byte-reproduce old C3 '+str(p))
            checks.append(dict(margin=m,identical_files=425,old=str(original),new=name))
        records.append(dict(method=name,margin=m,decoder=d,uses_future_frames=d not in (DECODERS[1],DECODERS[3],DECODERS[5]),
            prediction_rows=seal['prediction_rows'],target_GT_open_attempts=len(attempts),mask_path=str(folder/(name+'.npy')),
            mask_sha256=file_sha(folder/(name+'.npy')),teacher_calls=0,student_updates=0))
    if attempts:raise RuntimeError('target GT opened during decode')
    if len(checks)!=4:raise ValueError('require four historical X0 byte reproductions')
    result=dict(status='X0_ALL6_SEALED_FOUR_REPRODUCTIONS_PASS' if x0_only else 'ALL36_GT_FREE_PREDICTIONS_SEALED',methods=records,
        historical_X0_byte_checks=checks,target_GT_open_attempts=0,student_updates=0,teacher_calls=0,
        CPU_wall_seconds=time.perf_counter()-start,thresholds_sha256=None if x0_only else file_sha(RUN/'thresholds.json'))
    name='x0_gate.json' if x0_only else 'predictions_seal.json';atomic_json(RUN/name,result);atomic_json(PUBLIC/name,result)
    atomic_json(PUBLIC/'input_audit.json',json.loads((RUN/'input_audit.json').read_text()))
    print(json.dumps(dict(event='XQUERY_TARGET_SEALED',status=result['status'],methods=len(records))),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--x0-only',action='store_true');generate(p.parse_args().x0_only)
