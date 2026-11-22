"""Video-prefix visual continuity audit, preserving all selected boxes."""
import argparse
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from flex_uda.a100.mechanism import PAIRS,HISTORY,metadata,FrozenInputs,generation_guard
from flex_uda.active.continuity import unit,stitch
from flex_uda.presence.predictions import seal_mask
from scripts.a100_visual_classical_prepare import OUT
from scripts.a100_visual_classical_strict_prefix import read_guard

VARIANTS=('VISUAL','VISUAL_SEMANTIC')
SOURCES={'KD':('strict_prefix','COVER__TRACK_SOFT'),'EXEMPLAR':('exemplar_prefix','COVER__SUPPORT_MIX')}


def nodes_for(fc,pcd,semantic):
    boxes=np.asarray([[float(x) for x in line.split(',')[2:6]] for line in fc.bank['lines']])
    centers=boxes[:,:2]+.5*boxes[:,2:];result=[]
    for track in fc.tracks:
        ids=fc.by_track[track];first=ids[:3];last=ids[-3:];tail=ids[-5:]
        a=unit(np.asarray(pcd[first],dtype=float));b=unit(np.asarray(pcd[last],dtype=float))
        f=fc.observations[:,0];dt=np.diff(f[tail])
        velocity=np.median(np.diff(centers[tail],axis=0)/dt[:,None],axis=0) if len(tail)>1 else np.zeros(2)
        result.append(dict(track=track,first_frame=int(f[ids[0]]),last_frame=int(f[ids[-1]]),
            start_time=float(f[first].mean()),end_time=float(f[last].mean()),
            start_center=centers[first].mean(0),end_center=centers[last].mean(0),velocity=velocity,
            start_height=float(boxes[first,3].mean()),end_height=float(boxes[last,3].mean()),
            start_visual=unit(a.mean(0)),end_visual=unit(b.mean(0)),
            start_semantic=semantic[:,first].mean(1),end_semantic=semantic[:,last].mean(1)))
    return result


def run(pair):
    gt=generation_guard();visible=set();forbidden,reads=read_guard(pair,visible)
    model,target=PAIRS[pair];m=metadata(target);root=OUT/'continuity_prefix'/pair;begin=time.perf_counter()
    if (root/'completion.json').is_file():return
    receipts=[]
    for video in sorted(m['videos']):
        ready=OUT/'strict_prefix'/pair/'videos'/video/'prefix_receipt.json'
        while not ready.is_file():time.sleep(5)
        visible.add(video);path=root/'videos'/video/'receipt.json'
        if path.is_file():receipts.append(json.loads(path.read_text()));continue
        started=time.perf_counter();fc=FrozenInputs(pair,video)
        pcd=np.load(HISTORY/'features'/target/model/'videos'/video/'pcd.npy',mmap_mode='r')
        # Query zero is fixed for this video; only six endpoint descriptors per
        # source track are read. No target-trained feature weights are used.
        semantic=np.load(OUT/'strict_prefix'/pair/'COVER__FRAME_SOFT'/'videos'/video/'scores.npy')
        nodes=nodes_for(fc,pcd,semantic);links={};maps={}
        for variant in VARIANTS:
            mapping,record=stitch(nodes,use_semantics=variant=='VISUAL_SEMANTIC');maps[variant]=mapping;links[variant]=record
        path.parent.mkdir(parents=True,exist_ok=True)
        atomic_json(path,dict(video=video,mappings=maps,links=links,GT_open_attempts=len(gt),
            prefix_KD_receipt_sha256=file_sha(ready),visible_teacher_videos=sorted(visible),
            wall_seconds=time.perf_counter()-started));receipts.append(json.loads(path.read_text()))
        print(json.dumps(dict(event='CONTINUITY_VIDEO',pair=pair,video=video,
            links={k:len(v['links']) for k,v in links.items()},wall_seconds=time.perf_counter()-started)),flush=True)
    for source,(group,method) in SOURCES.items():
        while not (OUT/group/pair/'completion.json').is_file():time.sleep(5)
    records=[]
    for variant in VARIANTS:
        altered=dict(m,videos={})
        for row in receipts:
            video=row['video'];bank=m['videos'][video];mapping={int(k):v for k,v in row['mappings'][variant].items()};lines=[]
            for line in bank['lines']:
                fields=line.split(',');fields[1]=str(mapping[int(float(fields[1]))]);lines.append(','.join(fields))
            altered['videos'][video]=dict(bank,lines=lines)
        for source,(group,method) in SOURCES.items():
            reference=OUT/group/pair/method;mask=np.zeros(m['candidate_query_rows'],dtype=bool)
            for video in sorted(m['videos']):
                selection=np.load(reference/'videos'/video/'mask.npy')
                for qi,q in enumerate([q for q in m['episodes'] if q['video_id']==video]):mask[q['start']:q['stop']]=selection[qi]
            name=source+'__'+variant;folder=root/name
            seal=seal_mask(folder,altered,mask,target=target,model=model,method='CONTINUITY_'+name,GT_open_attempts=len(gt),
                extra=dict(original_boxes_and_selection_preserved=True,only_IDs_changed=True,
                semantic_reference=str(reference),current_prefix_teacher_only=True,new_teacher_calls=0,
                additional_frame_KD_cue=variant=='VISUAL_SEMANTIC',association='constant velocity plus endpoint frozen appearance; optional teacher-adapted semantic JS',
                no_target_GT_parameter_search=True,not_live_end_to_end_wall=True))
            original=json.loads((reference/'predictions_seal.json').read_text());assert seal['prediction_rows']==original['prediction_rows']
            for a,b in zip(seal['episodes'],original['episodes']):
                def without_id(path):return [line.split(',')[:1]+line.split(',')[2:] for line in Path(path).read_text().splitlines()]
                assert without_id(a['prediction_path'])==without_id(b['prediction_path'])
            records.append(dict(method=name,rows=seal['prediction_rows']))
    assert not gt and not forbidden
    atomic_json(root/'completion.json',dict(status='SEALED_CONTINUITY_SAME_BOX_CONTROLS',pair=pair,
        methods=records,GT_open_attempts=0,forbidden_read_attempts=0,receipts=receipts,
        wall_seconds=time.perf_counter()-begin,waiting_for_upstream_included=True,
        same_boxes_and_counts_verified=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pair',required=True,choices=['C4','C1']);a=p.parse_args();run(a.pair)
