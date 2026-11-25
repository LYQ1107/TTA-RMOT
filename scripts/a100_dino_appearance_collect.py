"""Image-only frozen descriptors for dense main slices and shared paid anchors.

This is input preparation. It does not produce labels, RMOT predictions or
evidence for a successful adaptation method. All future controls share it.
"""
import argparse
from collections import defaultdict
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from flex_uda.a100.mechanism import metadata,generation_guard
from scripts.a100_six_direction import OUT as SIX
from scripts.a100_anchor_observability import OUT as ANCHORS

OUT=SIX/'query_independent_DINO_20261125'
PUBLIC=ROOT/'docs/a100_six_direction_20261122'
PLAN=OUT/'input_plan.json'
SOURCES=('scripts/a100_dino_appearance_collect.py','flex_uda/active/frozen_appearance.py',
         'flex_uda/a100/mechanism.py','flex_uda/a100/runtime.py',
         'flex_uda/active/deterministic_observation_runtime.py','scripts/a100_anchor_observability.py',
         'configs/a100_paths.json')


def input_arrays(target,video):
    bank=metadata(target)['videos'][video]
    obs=np.asarray(bank['observations'],dtype='int64')
    boxes=np.asarray([[float(v) for v in line.split(',')[2:6]] for line in bank['lines']],dtype='float64')
    assert len(boxes)==len(obs) and len(set(map(tuple,obs)))==len(obs)
    return bank,obs,boxes


def prepare():
    guard=generation_guard()
    assets=json.loads((OUT/'asset_verification.json').read_text())
    assert assets['status']=='PASS_FROZEN_APPEARANCE_ASSETS_AND_CPU_FORWARD'
    source=SIX/'joint_teacher_noise_primary_20261125'
    parent_plan=json.loads((SIX/'crossfit_pattern_teacher_primary_20261125/registration.json').read_text())
    selected=defaultdict(set);main={};references=[]
    for case in parent_plan['cases']:
        target,pair,video=case['target'],case['pair'],case['evaluated_video']
        _,obs,_=input_arrays(target,video)
        selected[target,video].update(range(len(obs)));main[target,video]=pair
        path=source/pair/'row_keys.json'
        assert file_sha(path)==case['row_keys_sha256']
        keys=json.loads(path.read_text())['keys'];by_video=defaultdict(list)
        for row in keys:by_video[row[0]].append(row)
        for video,rows in by_video.items():
            _,obs,_=input_arrays(target,video)
            index={tuple(r):i for i,r in enumerate(obs)}
            tracks={int(r[2]) for r in rows}
            for row in rows:selected[target,video].add(index[int(row[3]),int(row[2])])
            with np.load(ANCHORS/target/video/'anchor_histograms.npz') as anchors:
                lookup={int(t):i for i,t in enumerate(anchors['tracks'])}
                for track in tracks:
                    ti=lookup[track]
                    selected[target,video].update(map(int,anchors['observations'][ti,anchors['present'][ti]]))
        references.append(dict(pair=pair,target=target,evaluated_video=case['evaluated_video'],
                               source_row_keys=str(path),row_keys_sha256=case['row_keys_sha256'],data_sha256=case['data_sha256']))
    units=[]
    for (target,video),rows in sorted(selected.items()):
        bank,obs,boxes=input_arrays(target,video);rows=np.asarray(sorted(rows),dtype='int64')
        folder=OUT/target/video;folder.mkdir(parents=True,exist_ok=True);np.save(folder/'selected_rows.npy',rows)
        units.append(dict(target=target,video=video,role=main.get((target,video),'AUX'),rows=len(rows),
             selected_rows_sha256=file_sha(folder/'selected_rows.npy'),
             crop_inputs_sha256=hashlib.sha256(obs[rows].tobytes()+boxes[rows].tobytes()).hexdigest(),
             image_root=bank['image_root']))
    plan=dict(status='REGISTERED_SHARED_IMAGE_ONLY_FROZEN_APPEARANCE_INPUTS',units=units,cases=references,
        rows=sum(u['rows'] for u in units),roles=['C1','C2','C4','AUX'],batch_size=16,
        descriptor='Frozen official-compatible DINOv2 ViT-B/14 CLS768; full candidate ROI clipped, bicubic resize224square, ImageNet mean/std, float32, no text/query input.',
        invalid_crop_rule='Retain observation with zero descriptor and explicit validity=False; never fabricate a semantic No.',
        source_sha256={n:file_sha(ROOT/n) for n in SOURCES},asset_sha256=file_sha(OUT/'asset_verification.json'),
        external_source_sha256=assets['external_source_sha256'],weight_path=assets['weight_path'],weight_sha256=assets['weight_sha256'],
        source_parent_registration_sha256=file_sha(SIX/'crossfit_pattern_teacher_primary_20261125/registration.json'),
        supplied_identically_to_all_future_ordinary_and_candidate_methods=True,
        all_data_DEV=True,pretraining_target_overlap_unknown=True,
        no_manual_target_GT_teacher_answer_or_model_quality_used_in_ROI_selection=True,
        dense_scope='All official candidate observations in each of the three current first videos; auxiliary paid-event crops and their same-track anchors in other videos.',
        new_Qwen_answer_calls=0,neural_training_or_RMOT_predictions=0,
        GPU_roles={'C1':0,'C2':1,'C4':3,'AUX':2},GPU2_shared_existing_services_preserved=True,
        visual_forward_cost_measured_but_not_a_fresh_single_method_deployment=True,goal_status='ACTIVE')
    assert not guard
    atomic_json(PLAN,plan)
    atomic_json(PUBLIC/'query_independent_DINO_input_plan_20261125.json',
                {k:v for k,v in plan.items() if k not in ('units','external_source_sha256')})
    print(json.dumps(dict(status=plan['status'],units=len(units),rows=plan['rows'],roles=plan['GPU_roles'])),flush=True)


def registration():
    plan=json.loads(PLAN.read_text())
    assert file_sha(OUT/'asset_verification.json')==plan['asset_sha256']
    assert file_sha(plan['weight_path'])==plan['weight_sha256']
    for name,sha in plan['source_sha256'].items():assert file_sha(ROOT/name)==sha,name
    for name,sha in plan['external_source_sha256'].items():assert file_sha(name)==sha,name
    return plan


def completed(unit):
    folder=OUT/unit['target']/unit['video'];path=folder/'receipt.json'
    if not path.exists():return None
    result=json.loads(path.read_text());assert result['input_plan_sha256']==file_sha(PLAN)
    for leaf,sha in result['output_sha256'].items():assert file_sha(folder/leaf)==sha
    return result


def extract(unit,model,device,plan):
    import torch
    from PIL import Image
    from flex_uda.active.frozen_appearance import crop_tensor
    reused=completed(unit)
    if reused:return dict(reused,reused_completed_unit=True,new_forward_batches_in_execution=0)
    folder=OUT/unit['target']/unit['video'];rows=np.load(folder/'selected_rows.npy')
    assert file_sha(folder/'selected_rows.npy')==unit['selected_rows_sha256']
    bank,obs,boxes=input_arrays(unit['target'],unit['video'])
    assert hashlib.sha256(obs[rows].tobytes()+boxes[rows].tobytes()).hexdigest()==unit['crop_inputs_sha256']
    shape=(len(rows),768);existing=(folder/'features.npy').exists()
    features=np.lib.format.open_memmap(folder/'features.npy',mode='r+' if existing else 'w+',dtype='float32',shape=shape)
    if existing:
        assert features.shape==shape
    valid=np.load(folder/'valid.npy') if (folder/'valid.npy').exists() else np.zeros(len(rows),dtype=bool)
    done=np.load(folder/'processed.npy') if (folder/'processed.npy').exists() else np.zeros(len(rows),dtype=bool)
    previous_hits=int(done.sum());by_frame=defaultdict(list)
    for i,row in enumerate(rows):
        if not done[i]:by_frame[int(obs[row,0])].append((i,int(row)))
    pending=[];indices=[];batches=0;forward_wall=crop_wall=decode_wall=0.;start=time.perf_counter()
    def flush():
        nonlocal batches,forward_wall
        if not pending:return
        torch.cuda.synchronize();stamp=time.perf_counter()
        with torch.inference_mode():value=model(torch.stack(pending).to(device)).float().cpu().numpy()
        torch.cuda.synchronize();forward_wall+=time.perf_counter()-stamp
        assert value.shape==(len(indices),768) and np.isfinite(value).all()
        features[indices]=value;valid[indices]=True;features.flush();done[indices]=True
        np.save(folder/'valid.npy',valid);np.save(folder/'processed.npy',done)
        batches+=1;pending.clear();indices.clear()
        if batches%64==0:print(json.dumps(dict(status='DINO_REAL_FORWARD_PROGRESS',target=unit['target'],video=unit['video'],completed_rows=int(done.sum()),total_rows=len(rows))),flush=True)
    extension='.jpg' if unit['target']=='SportsMOT' else '.png'
    for frame,frame_rows in sorted(by_frame.items()):
        stamp=time.perf_counter()
        with Image.open(Path(bank['image_root'])/f'{frame:06d}{extension}') as source:
            image=source.convert('RGB');image.load()
        decode_wall+=time.perf_counter()-stamp
        for i,row in frame_rows:
            stamp=time.perf_counter();tensor=crop_tensor(image,boxes[row]);crop_wall+=time.perf_counter()-stamp
            if tensor is None:
                features[i]=0.;valid[i]=False;done[i]=True
            else:
                pending.append(tensor);indices.append(i)
            if len(pending)==plan['batch_size']:flush()
        image.close()
    flush();features.flush();np.save(folder/'valid.npy',valid);np.save(folder/'processed.npy',done)
    np.save(folder/'observations.npy',obs[rows]);assert done.all() and np.isfinite(features).all()
    result=dict(status='COMPLETE_FROZEN_QUERY_INDEPENDENT_APPEARANCE_UNIT',target=unit['target'],video=unit['video'],
        rows=len(rows),valid_rows=int(valid.sum()),invalid_rows=int((~valid).sum()),descriptor_dimension=768,
        dtype='float32',input_plan_sha256=file_sha(PLAN),output_sha256={n:file_sha(folder/n) for n in
            ('features.npy','valid.npy','processed.npy','observations.npy','selected_rows.npy')},
        cached_processed_rows_at_start=previous_hits,new_forward_batches_in_execution=batches,
        image_decode_wall_seconds=decode_wall,crop_preprocess_wall_seconds=crop_wall,
        synchronized_visual_forward_wall_seconds=forward_wall,unit_wall_seconds=time.perf_counter()-start,
        new_Qwen_answer_calls=0,new_training_updates=0,manual_GT_reads=0,raw_descriptor_files_not_to_Git=True)
    atomic_json(folder/'receipt.json',result)
    print(json.dumps(dict(status=result['status'],target=unit['target'],video=unit['video'],rows=len(rows),real_forward_batches=batches)),flush=True)
    return result


def worker(role):
    from flex_uda.active.deterministic_observation_runtime import configure_before_cuda
    execution=configure_before_cuda()
    import torch
    from flex_uda.active.frozen_appearance import load_model
    plan=registration();guard=generation_guard();started=time.perf_counter()
    torch.set_num_threads(2);torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.10 if role=='AUX' else .35)
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
    stamp=time.perf_counter();model=load_model('cuda:0');torch.cuda.synchronize();load_wall=time.perf_counter()-stamp
    reports=[extract(unit,model,'cuda:0',plan) for unit in plan['units'] if unit['role']==role]
    assert not guard
    atomic_json(OUT/'supervisor'/(role+'_completion.json'),dict(status='COMPLETE_IMAGE_ONLY_DESCRIPTOR_ROLE',role=role,
        units=len(reports),rows=sum(r['rows'] for r in reports),real_new_forward_batches=sum(r['new_forward_batches_in_execution'] for r in reports),
        model_load_to_GPU_wall_seconds=load_wall,worker_wall_seconds=time.perf_counter()-started,
        execution=execution,GT_open_attempts=0,new_Qwen_answer_calls=0,new_training_or_RMOT_predictions=0))


def drain(process,log):
    with log.open('a',buffering=1) as stream:
        for line in process.stdout:stream.write(line);print(line,end='',flush=True)


def main():
    plan=registration();supervisor=OUT/'supervisor';supervisor.mkdir(parents=True,exist_ok=True)
    pending=[r for r in plan['roles'] if any(u['role']==r and completed(u) is None for u in plan['units'])]
    for role in pending:
        gpu=plan['GPU_roles'][role]
        residents=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader,nounits'],text=True).strip()
        used=int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
        total=int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.total','--format=csv,noheader,nounits'],text=True).strip())
        if gpu==2:assert total-used>=8192,(gpu,used,total)
        else:assert not residents and used<512,(gpu,residents,used)
        atomic_json(supervisor/(role+'_preflight.json'),dict(GPU=gpu,residents=residents,used_MiB=used,existing_processes_preserved=True))
    jobs=[];started=time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pipe_readers:
        for role in pending:
            gpu=plan['GPU_roles'][role]
            proc=subprocess.Popen([sys.executable,'-u',str(Path(__file__)),'--worker',role],cwd=ROOT,stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',XFORMERS_DISABLED='1'))
            future=pipe_readers.submit(drain,proc,supervisor/(role+'.log'));jobs.append((proc,future,role))
            atomic_json(supervisor/(role+'_launch.json'),dict(pid=proc.pid,GPU=gpu,role=role))
            print(json.dumps(dict(status='STARTED_FROZEN_APPEARANCE_FEATURE_ROLE',role=role,GPU=gpu,pid=proc.pid)),flush=True)
        failures=[]
        for proc,future,role in jobs:
            code=proc.wait();future.result()
            if code:failures.append(dict(role=role,exit_code=code))
    if failures:raise RuntimeError(failures)
    receipts=[completed(unit) for unit in plan['units']];assert all(receipts)
    report=dict(status='COMPLETE_SHARED_FROZEN_QUERY_INDEPENDENT_APPEARANCE_INPUTS',units=len(receipts),
        rows=sum(r['rows'] for r in receipts),valid_rows=sum(r['valid_rows'] for r in receipts),invalid_rows=sum(r['invalid_rows'] for r in receipts),
        elapsed_parent_wall_seconds=time.perf_counter()-started,input_plan_sha256=file_sha(PLAN),
        descriptor_dimension=768,all_future_methods_receive_identical_features=True,manual_GT_reads=0,
        new_Qwen_answer_calls=0,new_training_or_RMOT_quality_evaluations=0,
        extra_encoder_cost_not_a_single_method_cold_deployment=True,raw_arrays_not_to_Git=True,goal_status='ACTIVE')
    atomic_json(OUT/'completion.json',report);atomic_json(PUBLIC/'query_independent_DINO_feature_receipt_20261125.json',report)
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--worker');args=parser.parse_args()
    if args.prepare:prepare()
    elif args.worker:worker(args.worker)
    else:main()
