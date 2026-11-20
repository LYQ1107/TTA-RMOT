#!/usr/bin/env python3
"""Leave-one-video-out sensitivity of sealed official sufficient statistics."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.pf_diag_common import official_video_values
from scripts.analyze_pf_video_uncertainty import paired_clusters
from flex_tta.runtime.pf_receipts import atomic_json


def analyze(root,output):
    root=Path(root);result={}
    for a,b in [('PSR','FROZEN'),('PSR','NAIVE_PAIRED'),('PSR','UNIFORM_PAIRED'),('MEAN_TEACHER_MATCHED','FROZEN')]:
        videos,x=official_video_values(root/a/'results/pedestrian_detailed.csv')
        other,y=official_video_values(root/b/'results/pedestrian_detailed.csv')
        if videos!=other:raise ValueError('misaligned videos')
        all_delta=float(paired_clusters(x,y,np.ones(25)))
        omission=paired_clusters(x,y,np.ones((25,25))-np.eye(25))
        rows=[dict(omitted_video=v,remaining24_delta_HOTA=float(d),
                   change_from_all25=float(d-all_delta)) for v,d in zip(videos,omission)]
        result[a+'_vs_'+b]=dict(all25_delta_HOTA=all_delta,leave_one_out=rows,
            remaining24_positive=int((omission>0).sum()),remaining24_negative=int((omission<0).sum()),
            remaining24_range=[float(omission.min()),float(omission.max())],
            largest_absolute_influences=sorted(rows,key=lambda r:abs(r['change_from_all25']),reverse=True)[:5])
    atomic_json(output,dict(status='COMPLETE_FROM_SEALED_OFFICIAL_CSV_ONLY',GPU_calls=0,
        threshold_or_checkpoint_selection=False,comparisons=result,
        limitation='population sensitivity, not causal contribution; per-video HOTA deltas are not additive; distinct from training-seed variation'))
    print(json.dumps({k:{f:v[f] for f in ('all25_delta_HOTA','remaining24_positive','remaining24_negative','remaining24_range')} for k,v in result.items()}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evaluation-root',type=Path,default=ROOT/'results/runs/psr_val_20261029_v1/evaluation/merged')
    p.add_argument('--output',type=Path,default=ROOT/'results/psr/full/video_influence.json')
    a=p.parse_args();analyze(a.evaluation_root,a.output)
