"""Test an independent image constraint before proposing visual-memory learning.

Sparse LK cycles are evidence of local pixel correspondence, not semantic
truth or certified object identity. Provided IDs remain unchanged. No QA,
task predictions, GT, neural fit, or new backbone is used.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from scripts.a100_six_direction import OUT as SIX
OUT=SIX/'pixel_continuity_audit_20261125'
PLAN=OUT/'registration.json'
PUBLIC=ROOT/'docs/a100_six_direction_20261122'
SOURCES=('scripts/a100_pixel_continuity_audit.py','scripts/active_run.py',
         'flex_uda/a100/mechanism.py','flex_uda/a100/runtime.py')


def prepare():
    import numpy as np
    from flex_uda.a100.mechanism import metadata,generation_guard
    guard=generation_guard();assert not PLAN.exists()
    cases=[]
    for target in ('Dance','KITTI-V2','SportsMOT'):
        md=metadata(target);video=sorted(md['videos'])[0]
        frames=np.unique(np.asarray(md['videos'][video]['observations'])[:,0])
        starts=np.unique(np.linspace(0,len(frames)-3,12).astype(int))
        cases.append(dict(target=target,video=video,triplets=[frames[i:i+3].astype(int).tolist() for i in starts]))
    doc=dict(status='REGISTERED_IMAGE_ONLY_SHORT_PIXEL_CONTINUITY_AUDIT',cases=cases,
        source_sha256={n:file_sha(ROOT/n) for n in SOURCES},
        purpose='New candidate-training signal feasibility after old learned graph lacked demonstrated selectivity and provided IDs could mix people. Not a semantic smoothness assumption or a module ablation.',
        evidence='Three successive OBSERVED frames. Source points from inner70%of each provided box, max64ShiTomasi. LK0->1->2->0 and direct1->0;1.5px cycle threshold, normalized1024max-side.',
        association='All unmodified provided candidate boxes considered. Point in exactly1destination box gives a box association, overlaps/none retained explicitly. Require points associate with same destination track in both future images.',
        limitations=['Pixel cycles can still follow background, reflections or repeated texture. They do not certify foreground/object identity.',
          'Even identical physical objects can change query membership. No equality constraint on full semantic scores is justified here.',
          'Reported IDs are provided, never target annotations. Neither track IDs nor output masks are changed.',
          'Different time selection from old32-frame audit; no isolated causal frame-gap comparison.'],
        seed=0,neural_training=0,teacher_calls=0,GT_reads=0,CPU_workers=3,goal_status='ACTIVE')
    assert not guard
    atomic_json(PLAN,doc);atomic_json(PUBLIC/'pixel_continuity_audit_registration_20261125.json',doc)
    print(json.dumps(dict(status=doc['status'],triplets=36)),flush=True)


def worker(case):
    import cv2
    import numpy as np
    from PIL import Image
    from scripts.active_run import video_spec
    from flex_uda.a100.mechanism import metadata,generation_guard
    guard=generation_guard();cv2.setNumThreads(1);cv2.setRNGSeed(0)
    spec=video_spec(case['target'],case['video'],metadata(case['target'])['videos'][case['video']])
    observations={}
    for r in spec['observations']:observations.setdefault(r['frame_id'],{})[r['track_id']]=r['bbox_xywh']
    folder=OUT/case['target'];folder.mkdir(parents=True,exist_ok=True)
    criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01)
    def load(frame):
        path=Path(spec['image_root'])/f"{frame+spec['image_offset']:06d}{spec['extension']}"
        with Image.open(path) as image:
            scale=1024/max(image.size)
            gray=np.asarray(image.convert('L').resize((round(image.width*scale),round(image.height*scale)),Image.Resampling.BICUBIC)).copy()
        boxes={int(t):np.asarray([x,y,x+w,y+h])*scale for t,(x,y,w,h) in observations[frame].items()}
        return gray,boxes
    def flow(a,b,p):
        q,s,e=cv2.calcOpticalFlowPyrLK(a,b,p.reshape(-1,1,2),None,winSize=(21,21),maxLevel=3,criteria=criteria)
        assert q is not None
        return q[:,0],s[:,0].astype(bool)
    def association(points,boxes):
        contains=np.zeros((len(points),len(boxes)),dtype=bool);ids=np.asarray(list(boxes),dtype=int)
        for i,(x0,y0,x1,y1) in enumerate(boxes.values()):
            contains[:,i]=(points[:,0]>=x0)&(points[:,0]<=x1)&(points[:,1]>=y0)&(points[:,1]<=y1)
        counts=contains.sum(1);result=np.full(len(points),-1,dtype=int)
        result[counts>1]=-2
        if len(ids):result[counts==1]=ids[np.argmax(contains[counts==1],axis=1)]
        return result
    records=[]
    for number,frames in enumerate(case['triplets']):
        (a,boxes0),(b,boxes1),(c,boxes2)=[load(f) for f in frames]
        assert a.shape==b.shape==c.shape
        units=[];arrays={}
        for tid,box in sorted(boxes0.items()):
            x0,y0,x1,y1=box;dx=.15*(x1-x0);dy=.15*(y1-y0)
            roi=np.zeros(a.shape,dtype='uint8')
            roi[max(0,int(y0+dy)):min(a.shape[0],int(y1-dy)),max(0,int(x0+dx)):min(a.shape[1],int(x1-dx))]=255
            points=cv2.goodFeaturesToTrack(a,maxCorners=64,qualityLevel=.01,minDistance=3,mask=roi)
            if points is None:units.append(dict(track=int(tid),points=0,cycle_valid=0,same_ID_supported_points=0,other_ID_supported_points=0,ambiguous_or_missing_points=0));continue
            p=points[:,0];q,ok1=flow(a,b,p);r,ok2=flow(b,c,q)
            back,ok3=flow(c,a,r);direct,ok4=flow(b,a,q)
            valid=ok1&ok2&ok3&ok4
            valid&=np.isfinite(q).all(1)&np.isfinite(r).all(1)&np.isfinite(back).all(1)&np.isfinite(direct).all(1)
            valid&=(np.linalg.norm(back-p,axis=1)<=1.5)&(np.linalg.norm(direct-p,axis=1)<=1.5)
            for xy,im in ((q,b),(r,c)):
                valid&=(xy[:,0]>=0)&(xy[:,0]<im.shape[1])&(xy[:,1]>=0)&(xy[:,1]<im.shape[0])
            to1,to2=association(q,boxes1),association(r,boxes2)
            same=valid&(to1==tid)&(to2==tid)
            other=valid&(to1>=0)&(to1==to2)&(to1!=tid)
            unclear=valid&~same&~other
            units.append(dict(track=int(tid),points=len(p),cycle_valid=int(valid.sum()),
                same_ID_supported_points=int(same.sum()),other_ID_supported_points=int(other.sum()),
                ambiguous_or_missing_points=int(unclear.sum())))
            for name,value in dict(p=p,q=q,r=r,valid=valid,to1=to1,to2=to2).items():arrays[str(tid)+'_'+name]=value
        np.savez(folder/f'triplet_{number:02d}.npz',**arrays)
        records.append(dict(frames=frames,units=units))
    assert not guard
    allunits=[u for r in records for u in r['units']]
    keys=('points','cycle_valid','same_ID_supported_points','other_ID_supported_points','ambiguous_or_missing_points')
    summary=dict(target=case['target'],video=case['video'],triplets=len(records),source_boxes=len(allunits),
        totals={k:sum(u[k] for u in allunits) for k in keys},
        boxes_with_at_least3cycle_points=sum(u['cycle_valid']>=3 for u in allunits),
        boxes_with_at_least3same_ID_points=sum(u['same_ID_supported_points']>=3 for u in allunits),
        boxes_with_more_other_ID_than_same_ID_points=sum(u['other_ID_supported_points']>=3 and u['other_ID_supported_points']>u['same_ID_supported_points'] for u in allunits))
    atomic_json(folder/'completion.json',dict(summary=summary,triplets=records,GT_reads=0,teacher_calls=0))
    print(json.dumps(dict(status='PIXEL_CONTINUITY_CASE_COMPLETE',**summary)),flush=True)
    return summary


def main():
    plan=json.loads(PLAN.read_text())
    for n,sha in plan['source_sha256'].items():assert file_sha(ROOT/n)==sha,n
    started=time.perf_counter()
    with ProcessPoolExecutor(max_workers=3) as pool:cases=list(pool.map(worker,plan['cases']))
    result=dict(status='COMPLETE_IMAGE_ONLY_SHORT_PIXEL_CONTINUITY_AUDIT',cases=cases,
        wall_seconds=time.perf_counter()-started,registration_sha256=file_sha(PLAN),
        GT_reads=0,teacher_calls=0,neural_updates=0,RMOT_predictions=0,
        no_semantic_truth_or_HOTA_contribution_claim=True,goal_status='ACTIVE')
    atomic_json(OUT/'completion.json',result);atomic_json(PUBLIC/'pixel_continuity_audit_completion_20261125.json',result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');a=p.parse_args()
    if a.prepare:prepare()
    else:main()
