"""All28 preregistered selectors; GT-free process, no teacher/model calls."""
import json
from pathlib import Path
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.count_structure_common import *

def generate():
    setup();require_plan();attempts=v2.deny_gt();started=time.perf_counter()
    meta,scores,groups=load_inputs();cal=json.loads((RUN/'source_calibration.json').read_text())
    arrays={};cache=RUN/'masks';cache.mkdir(exist_ok=True)
    for m in MARGINS:
        for c in SELECTORS:arrays[m,c]=np.lib.format.open_memmap(cache/(m+'__'+c+'.npy'),mode='w+',dtype='bool',shape=(len(groups),))
    mu={m:float(scores[m].mean()) for m in MARGINS};priors=[]
    for video,queries,tracks,frames,ids,unique,inverse in video_banks(meta):
        for m in MARGINS:
            matrix=np.stack([scores[m][q['start']:q['stop']] for q in queries])
            masks,details=masks_for_video(matrix,inverse,cal['source_mean'],mu[m],cal)
            for j,q in enumerate(queries):
                priors.append(dict(margin=m,video=video,expression=q['expression_id'],**details[j]))
                for c,mask in masks.items():arrays[m,c][q['start']:q['stop']]=mask[j]
        print(json.dumps(dict(event='COUNT_SELECTORS_VIDEO',video=video)),flush=True)
    lines=v2.candidate_lines(meta);methods=[]
    for (m,c),a in arrays.items():
        a.flush();name=m+'__'+c
        v2.write_selection(name,a,meta,lines,protocol='GT_FREE_SELECTION_NO_NEW_TRAINING__'+m)
        methods.append(dict(method=name,margin=m,selector=c,uses_future_frames=c not in ('C1c_TRACK_CAUSAL','C2_QUERY_ARGMAX'),
            target_GT_open_attempts=len(attempts),teacher_calls=0,student_updates=0,
            mask_path=str(cache/(name+'.npy')),mask_sha256=file_sha(cache/(name+'.npy'))))
    if attempts:raise RuntimeError('GT-free selector touched annotations')
    result=dict(status='ALL28_GT_FREE_SELECTORS_SEALED',methods=methods,source_calibration_sha256=file_sha(RUN/'source_calibration.json'),
        target_GT_open_attempts=0,teacher_calls=0,student_updates=0,CPU_wall_seconds=time.perf_counter()-started,
        target_global_margin_means=mu,source_global_margin_mean=cal['source_mean'])
    atomic_json(RUN/'selectors_seal.json',result);atomic_json(PUBLIC/'selectors_seal.json',result)
    atomic_json(PUBLIC/'em_priors.json',priors)
    print(json.dumps(dict(event='COUNT_ALL28_SEALED',CPU_wall_seconds=result['CPU_wall_seconds'])),flush=True)

if __name__=='__main__':generate()
