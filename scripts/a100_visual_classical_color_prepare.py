"""Extract fixed body/object colour statistics from real target images, no GT."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image
from flex_tta.runtime.pf_receipts import atomic_json
from flex_uda.a100.mechanism import metadata,PAIRS,generation_guard
from flex_uda.active.color_cohort import part_histograms
from scripts.a100_visual_classical_prepare import OUT


def run(pair):
    guard=generation_guard();begin=time.perf_counter();_,target=PAIRS[pair];m=metadata(target);root=OUT/'color_features'/pair
    if (root/'completion.json').is_file():return
    receipts=[]
    for video,bank in sorted(m['videos'].items()):
        path=root/video/'features.npz';rp=path.with_name('receipt.json')
        if rp.is_file():receipts.append(json.loads(rp.read_text()));continue
        started=time.perf_counter();obs=np.asarray(bank['observations']);tracks=sorted(set(obs[:,1].tolist()))
        boxes=np.asarray([[float(x) for x in line.split(',')[2:6]] for line in bank['lines']]);byframe=defaultdict(list)
        for ti,track in enumerate(tracks):
            ids=np.flatnonzero(obs[:,1]==track);selected=ids[np.unique(np.linspace(0,len(ids)-1,min(3,len(ids))).round().astype(int))]
            for index in selected:byframe[int(obs[index,0])].append((ti,int(index)))
        values=[[] for _ in tracks]
        for frame,items in sorted(byframe.items()):
            image_path=Path(bank['image_root'])/(f'{frame:06d}'+('.jpg' if target=='SportsMOT' else '.png'))
            with Image.open(image_path) as opened:
                image=opened.convert('RGB')
                for ti,index in items:values[ti].append(part_histograms(image,boxes[index]))
        def aggregate(v):
            v=np.asarray(v);count=(v.sum(2)>0).sum(0)
            mean=np.divide(v.sum(0),count[:,None],out=np.zeros_like(v[0]),where=count[:,None]>0)
            return np.sqrt(mean)
        feature=np.stack([aggregate(v) for v in values]).astype('float32')
        path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,colours=feature,tracks=tracks)
        receipt=dict(video=video,tracks=len(tracks),real_images=len(byframe),patches=sum(map(len,values)),
            histogram='12x8 HS plus8V, equal probability mass; square-root Hellinger representation',
            parts='whole object, upper body, lower body; fixed relative candidate coordinates',
            invisible_sample_parts=sum(int((np.asarray(v).sum(2)==0).sum()) for v in values),
            missing_colour_does_not_remove_candidates=True,
            GT_open_attempts=len(guard),wall_seconds=time.perf_counter()-started)
        atomic_json(rp,receipt);receipts.append(receipt)
        print(json.dumps(dict(event='COLOR_FEATURE_VIDEO',pair=pair,**receipt)),flush=True)
    assert not guard
    atomic_json(root/'completion.json',dict(status='PURE_IMAGE_COLOUR_FEATURES_READY',pair=pair,GT_open_attempts=0,
        receipts=receipts,wall_seconds=time.perf_counter()-begin,target_labels_used=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pair',required=True,choices=['C4','C1']);a=p.parse_args();run(a.pair)
