"""Query-independent per-observation colour inputs, shared by every learner.

Unlike the anchor audit, retain each actual candidate observation. Raw images
and all NAS history remain read-only. This is input preparation, not an RMOT
gain, and existing historical dense PCD remains on NAS for matched controls.
"""
from collections import defaultdict
import concurrent.futures
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from flex_uda.a100.mechanism import metadata,generation_guard
from flex_uda.active.color_cohort import part_histograms
from scripts.a100_anchor_observability import OUT as ANCHOR_RGB
from scripts.a100_six_direction import OUT as SIX,PUBLIC

OUT=SIX/'dense_RGB_20261123';STATE=OUT/'supervisor/state.json'
REGISTRATION=PUBLIC/'dense_RGB_registration_v1.json'
SUCCESS='PASS_DENSE_RGB_SHARED_IMAGE_INPUTS_NO_GT'
TARGETS=('Dance','KITTI-V2','SportsMOT')


def prepare():
    guard=generation_guard();per_target={}
    for target in TARGETS:
        cases=[]
        for video,bank in sorted(metadata(target)['videos'].items()):
            obs=np.asarray(bank['observations'],dtype=int)
            cases.append(dict(target=target,video=video,observations=len(obs),image_frames=len(np.unique(obs[:,0]))))
        per_target[target]=cases
    # Interleave task families so each new stream can get its first video.
    cases=[]
    for index in range(max(map(len,per_target.values()))):
        for target in TARGETS:
            if index<len(per_target[target]):cases.append(per_target[target][index])
    assert len(cases)==74 and not guard
    plan=dict(status='SEALED_DENSE_RGB_SHARED_INPUT_PLAN',cases=cases,
        totals={k:sum(r[k] for r in cases) for k in ('observations','image_frames')},
        shared_across_source_models_and_all_candidate_methods=True,CPU_workers=4,
        no_teacher_QA_GT_prediction_or_neural_forward_inputs=True)
    atomic_json(OUT/'preparation.json',plan)
    (PUBLIC/'dense_RGB_input_plan.json').write_text(json.dumps(plan,separators=(',',':'))+'\n')
    return plan


def extract(case):
    target,video=case['target'],case['video'];folder=OUT/target/video
    if (folder/'receipt.json').is_file():return json.loads((folder/'receipt.json').read_text())
    start=time.perf_counter();bank=metadata(target)['videos'][video]
    obs=np.asarray(bank['observations'],dtype=int)
    boxes=np.asarray([[float(v) for v in line.split(',')[2:6]] for line in bank['lines']])
    assert len(boxes)==len(obs)
    jobs=defaultdict(list)
    for row,(frame,_) in enumerate(obs):jobs[int(frame)].append(row)
    with np.load(ANCHOR_RGB/target/video/'anchor_histograms.npz') as old:
        old_rows=old['observations'];present=old['present'];old_hist=old['histograms'].copy()
    anchor={int(old_rows[ti,si]):(ti,si) for ti,si in np.argwhere(present)}
    folder.mkdir(parents=True,exist_ok=True)
    hist=np.lib.format.open_memmap(folder/'histograms.npy',mode='w+',dtype='float32',shape=(len(obs),3,104))
    valid=np.zeros((len(obs),3),dtype=bool);decode=compute=0.;checks=0
    extension='.jpg' if target=='SportsMOT' else '.png'
    for frame,rows in sorted(jobs.items()):
        stamp=time.perf_counter()
        with Image.open(Path(bank['image_root'])/f'{frame:06d}{extension}') as image:rgb=image.convert('RGB');rgb.load()
        decode+=time.perf_counter()-stamp;stamp=time.perf_counter()
        for row in rows:
            value=part_histograms(rgb,boxes[row]).astype('float32');hist[row]=value;valid[row]=value.sum(1)>0
            if row in anchor:
                ti,si=anchor[row];assert np.array_equal(value,old_hist[ti,si]),(target,video,row)
                checks+=1
        rgb.close();compute+=time.perf_counter()-stamp
    assert checks==len(anchor)
    hist.flush();np.save(folder/'valid_parts.npy',valid);np.save(folder/'observations.npy',obs)
    receipt=dict(target=target,video=video,observations=len(obs),image_frame_reads=len(jobs),
        histogram_shape=list(hist.shape),dtype='float32',feature_bytes=int(hist.nbytes),
        anchor_observations_checked=checks,all_anchor_histograms_byte_equal=True,
        valid_parts=int(valid.sum()),missing_parts=int((~valid).sum()),
        image_decode_wall_seconds=decode,histogram_wall_seconds=compute,worker_wall_seconds=time.perf_counter()-start,
        same_descriptors_available_to_all_ordinary_and_conditional_models=True,
        no_frame_label_or_semantic_identity_inferred=True,raw_features_not_to_Git=True)
    atomic_json(folder/'receipt.json',receipt);del hist
    return receipt


def main():
    reg=json.loads(REGISTRATION.read_text())
    for name,sha in reg['source_sha256'].items():assert file_sha(ROOT/name)==sha,name
    guard=generation_guard();start=time.perf_counter();plan=prepare();STATE.parent.mkdir(parents=True,exist_ok=True)
    receipts=[]
    atomic_json(STATE,dict(status='RUNNING_DENSE_RGB_SHARED_IMAGE_INPUTS_NO_GT',pid=os.getpid(),completed=0,total=74))
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(extract,case) for case in plan['cases']]
        for future in concurrent.futures.as_completed(futures):
            receipts.append(future.result());last=receipts[-1]
            atomic_json(STATE,dict(status='RUNNING_DENSE_RGB_SHARED_IMAGE_INPUTS_NO_GT',pid=os.getpid(),completed=len(receipts),
                total=74,last_completed=dict(target=last['target'],video=last['video']),goal_status='ACTIVE'))
    assert len(receipts)==74 and not guard
    report=dict(status=SUCCESS,videos=len(receipts),
        totals={k:sum(r[k] for r in receipts) for k in ('observations','image_frame_reads','feature_bytes','anchor_observations_checked','valid_parts','missing_parts')},
        all_existing_anchor_histograms_byte_equal=True,actual_parallel_CPU_workers=4,
        process_wall_seconds=time.perf_counter()-start,
        summed_image_decode_seconds=sum(r['image_decode_wall_seconds'] for r in receipts),
        summed_histogram_seconds=sum(r['histogram_wall_seconds'] for r in receipts),
        shared_across_source_models_and_candidate_methods=True,
        no_manual_GT_or_QA_or_quality_prediction_inputs=True,new_neural_forward_calls=0,
        cost_scope='Shared full-observation image-feature preparation, not single-method deployment or semantic learning.',
        no_HOTA_or_retained_module_claim=True,raw_features_not_to_Git=True,goal_status='ACTIVE')
    atomic_json(OUT/'completion.json',report)
    (PUBLIC/'dense_RGB_receipt.json').write_text(json.dumps(report,separators=(',',':'))+'\n')
    atomic_json(STATE,dict(status=SUCCESS,pid=os.getpid(),completed=74,goal_status='ACTIVE'))


if __name__=='__main__':
    try:main()
    except Exception as error:
        previous=json.loads(STATE.read_text()) if STATE.is_file() else {}
        previous.update(status='FAILED_DENSE_RGB_SHARED_INPUTS',pid=os.getpid(),error=repr(error));atomic_json(STATE,previous);raise
