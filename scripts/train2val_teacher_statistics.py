#!/usr/bin/env python3
"""GT-free cached Train teacher distribution and initial Frozen fit; CPU only."""
from collections import defaultdict
import csv
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from flex_uda.data.train2val import ASSETS
from flex_uda.data.manifests import read_jsonl
from flex_uda.diag_fit import metrics
from flex_tta.runtime.pf_receipts import atomic_json,file_sha


def run():
    rows=read_jsonl(ASSETS/'pseudo/NAIVE/soft_labels.jsonl')
    groups={'video':defaultdict(list),'expression':defaultdict(list)}
    records=[]
    for r in rows:
        groups['video'][r['video_id']].append(r);groups['expression'][r['query_id']].append(r)
        records.append(dict(valid=r['valid'],video_id=r['video_id'],query_id=r['query_id'],frame_id=r['frame_id'],
            teacher_probability=r['teacher_prob'],student_margin=r['frozen_margin']))
    initial=metrics(records)
    output=ROOT/'results/train2val'
    for kind,grouped in groups.items():
        summaries=[]
        for key,part in sorted(grouped.items()):
            valid=[r for r in part if r['valid']];pos=[r for r in valid if r['teacher_prob']>=.95]
            summaries.append(dict(group=key,judgments=len(part),valid=len(valid),POS=len(pos),POS_fraction=len(pos)/len(valid),
                mean_teacher_p=float(np.mean([r['teacher_prob'] for r in valid])),
                frozen_selected_rate=float(np.mean([r['frozen_margin']>0 for r in valid])),
                frozen_selected_teacher_POS=sum(r['frozen_margin']>0 for r in pos)))
        with (output/('teacher_by_'+kind+'.csv')).open('w') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(summaries[0]));writer.writeheader();writer.writerows(summaries)
    result=dict(status='CACHED_FROZEN_FIT_ON_TRAIN_TEACHER_NOT_GT_ACCURACY',all=initial,
        teacher_cache_sha256=file_sha(ASSETS/'pseudo/NAIVE/soft_labels.jsonl'),extra_GPU_calls=0,
        extra_teacher_judgments=0,referring_GT_opened=False,Val_inputs_opened=False,training_or_selection_feedback=False)
    atomic_json(output/'frozen_train_fit.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':run()
