#!/usr/bin/env python3
"""Disclose legacy-batched versus sealed ZERO_UPDATE rounding, no GT/inference."""
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.diag_attrib_v2_common import *


if __name__=='__main__':
    require_prereg();attempts=deny_gt();start=time.perf_counter();cpu=time.process_time()
    path=ROOT/'results/runs/flexhook_v2_to_dance_official_detr_neuralsort_corrected_tokenizer/result.json'
    if file_sha(path)!='fbf498a24b73d6b2db6d9c757d313f660c0d5b09a7dbfec5c2635b8ca517ee69':
        raise ValueError('legacy Frozen file changed')
    legacy=json.loads(path.read_text())
    metadata,scores,groups,historical=load_bank()
    tracks={v:load_tracker_lines(TRACKS,v) for v in sorted(legacy)}
    flips=[];covered=old_positive=current_positive=0;maximum=0.
    for q in metadata['episodes']:
        v,sentence=q['video_id'],q['query_text']
        for t,a,b,g in q['groups']:
            for j,i in enumerate(tracks[v].get(t,{})):
                if not historical[a+j]:continue
                pair=legacy[v][str(i)][str(t)][sentence]
                old=float(pair[1])-float(pair[0]);now=float(scores['FROZEN'][a+j])
                covered+=1;old_positive+=old>0;current_positive+=now>0;maximum=max(maximum,abs(old-now))
                if (old>0)!=(now>0):
                    flips.append(dict(video=v,expression=q['expression_id'],frame=t,track_id=i,
                                      legacy_margin=old,sealed_ZERO_UPDATE_margin=now))
    if covered!=3576681 or old_positive!=99316 or current_positive!=99320:
        raise ValueError('unexpected historical-overlap count discrepancy')
    output=dict(status='DISCLOSED_NO_RESULT_REPLACEMENT',covered_candidate_query_rows=covered,
        legacy_batched_positive_rows=old_positive,sealed_ZERO_UPDATE_same_support_positive_rows=current_positive,
        sign_flips=flips,max_margin_difference=maximum,target_GT_open_attempts=attempts,GPU_calls=0,
        legacy_file_sha256=file_sha(path),scope='historical mask of current sealed ZERO_UPDATE, not a replay of early batched logits',
        old_8_206_record_unchanged=True,CPU_wall_seconds=time.perf_counter()-start,CPU_process_seconds=time.process_time()-cpu)
    atomic_json(PUBLIC/'frozen_historical_overlap.json',output)
    print(json.dumps(output,indent=2),flush=True)
