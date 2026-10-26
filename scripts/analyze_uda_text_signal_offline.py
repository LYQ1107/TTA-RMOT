#!/usr/bin/env python3
"""Reuse frozen-score query-swap/mixed-frame controls on sealed UDA outputs."""
import argparse
from contextlib import ExitStack
import gzip
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from flex_uda.training import METHODS
from flex_tta.runtime.pf_receipts import atomic_json
from scripts.pf_diag_common import load_source_video,candidate_labels,write_csv,TRACKS
from scripts.diag_pf_query_information import frame_metrics,add,finalize
from flex_tta.dance_io import dance_videos


def scores_for(root,method,bank):
    scores=[];offset=0
    with ExitStack() as stack:
        streams=[stack.enter_context(gzip.open(root/method/'results'/bank['video']/e/'candidate_margins.jsonl.gz','rt')) for e,_ in bank['expressions']]
        for lines in zip(*streams,strict=True):
            rows=[json.loads(s) for s in lines];t=rows[0]['frame'];ids=rows[0]['ids'];end=offset+len(ids)
            if any(r['frame']!=t or r['ids']!=ids for r in rows):raise ValueError('cross-query candidate order differs')
            if not np.array_equal(bank['frame'][offset:end],np.repeat(t,len(ids))) or ids!=bank['ids'][offset:end].tolist():raise ValueError('Frozen alignment differs')
            scores.extend(np.asarray([r['adapted'] for r in rows]).T);offset=end
    scores=np.asarray(scores)
    if scores.shape!=bank['scores'].shape:raise ValueError('query-score matrix shape mismatch')
    return scores


def analyze(root,public):
    root,public=map(lambda p:Path(p).resolve(),(root,public));methods=('FROZEN',)+METHODS
    for method in METHODS:
        audit=json.loads((root/method/'full_audit.json').read_text())
        if audit['status']!='PASS' or audit['valid_episodes']!=425:raise ValueError('all predictions must be sealed before GT text controls')
    totals={m:{} for m in methods};by_video={m:[] for m in methods};video_rows=[]
    for video in dance_videos(TRACKS):
        bank=load_source_video(video);labels,_=candidate_labels(bank)
        for method in methods:
            scores=bank['scores'] if method=='FROZEN' else scores_for(root,method,bank)
            local={}
            for frame,a,b in bank['slices']:
                rows,_=frame_metrics(scores[a:b],labels[a:b])
                for row in rows:add(local,row);add(totals[method],row)
            by_video[method].append(local);video_rows.append(dict(video=video,method=method,**finalize(local)))
        print(json.dumps(dict(event='offline_query_control_video',video=video)),flush=True)
    point={m:finalize(t) for m,t in totals.items()};comparisons={}
    draws=np.random.default_rng(0).integers(0,25,size=(2000,25))
    for method in METHODS:
        fields={}
        for first,second,denominator in (('correct_top1',None,'valid_frames'),('mixed_source_AUROC',None,'mixed_frames'),
            ('correct_top1','all_other_top1','valid_frames'),('mixed_source_AUROC','mixed_visual_AUROC','mixed_frames'),
            ('hard_disjoint_correct_top1','hard_disjoint_top1','hard_disjoint_frames')):
            def effect(m,ix):
                den=sum(by_video[m][i].get(denominator,0) for i in ix)
                num=sum(by_video[m][i].get(first,0)-(by_video[m][i].get(second,0) if second else 0) for i in ix)
                return num/den if den else np.nan
            delta=effect(method,range(25))-effect('FROZEN',range(25))
            boot=np.asarray([effect(method,d)-effect('FROZEN',d) for d in draws]);boot=boot[np.isfinite(boot)]
            fields[first+('_minus_'+second if second else '')]=dict(delta=float(delta) if np.isfinite(delta) else None,
                video_cluster_CI95=np.percentile(boot,[2.5,97.5]).tolist() if len(boot) else None,clusters=25,
                valid_bootstrap_resamples=len(boot))
        comparisons[method+'_vs_FROZEN']=fields
    result=dict(protocol='OFFLINE_TEXT_INFORMATION_DIAGNOSTIC',methods=point,comparisons=comparisons,
        target_GT_only_after_all425_predictions=True,target_GT_used_for_training_or_selection=False,
        mixed_source_AUROC_name='legacy helper name: refers to the supplied method score on within-frame mixed positive/negative candidates',
        visual_control='candidate mean over17 true-query scores, not a separately trained visual network',
        text_swap_control='other real query may match same person; disjoint and hard-disjoint variants also reported',
        uncertainty='paired25-video bootstrap2000/seed0, descriptive same-Val only; no independent-frame claim')
    atomic_json(public/'text_signal_offline.json',result)
    columns=list(dict.fromkeys(k for row in video_rows for k in row))
    write_csv(public/'per_video_text_signal.csv',[{k:r.get(k) for k in columns} for r in video_rows])
    print(json.dumps(result,indent=2));return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--evaluation-root',type=Path,required=True)
    p.add_argument('--public-root',type=Path,default=ROOT/'results/uda/full');a=p.parse_args();analyze(a.evaluation_root,a.public_root)
