"""Necessary same-capacity ordinary controls, registered before new HOTA.

The primary and its31files stay immutable. Larger heads do not guarantee
better generalization; add both ordinary128flat and the exact128solver
architecture, not a module-ablation matrix or new main candidate.
"""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from scripts.a100_evidence_visual_primary import (
    OUT as PARENT,PLAN as PARENT_PLAN,PRIMARY,PUBLIC,registration as parent_registration)

OUT=PARENT/'same_capacity_controls_20261126'
PLAN=OUT/'registration.json'
STEM='evidence_visual_capacity_controls_20261126'
COMPARATORS=('FLAT_KD128','SAME_ARCH_KD128')


def prepare():
    parent=parent_registration();assert not PLAN.exists()
    assert not (PARENT/'quality_results.json').exists(),'Register before new primary HOTA is available'
    sources=dict(parent['source_sha256']);sources[str(Path(__file__).resolve().relative_to(ROOT))]=file_sha(__file__)
    doc=dict(status='PREREGISTERED_NECESSARY_SAME_CAPACITY_ORDINARY_CONTROLS',primary=PRIMARY,
        comparators=COMPARATORS,cases=parent['cases'],source_sha256=sources,parent_registration_sha256=file_sha(PARENT_PLAN),
        motive='Larger288flat andSSLheads might generalize differently; preserve ordinary128flat and exact128primarySarchitecture with directteacherBCE as additional opponents. Never count capacity change as main benefit.',
        models='FLAT_KD128:plain9352->128->128->1. SAME_ARCH_KD128:exact VisualHead128gatedcontent architecture ofS. Both original same acquiredteacheranswers/auxiliary,seed0/Adam.0001/clip10/softsign8 andsamephases,properKL.005/min6000/max16000.',
        pixel_information='Same raw9352andimagepairs are available, no additional targets. ParentalreadyincludesSSL288explicitself-supervisedordinarycontrol.',
        execution='After parentall75sealed and its GPUworkershaveexited, new actualfits use0/1/3;GPU2services untouched. Own18outputs+3parentrawreferences sealed before ownGT, wait parent75reproduction before reusing completed officialraw.',
        separate_study_GT_barriers='Parent seals its75beforeitsGT. Companion seals its18beforeownGT; parentGTmayrun concurrently with companion learning in separate guarded processes. No wholecombined93beforeanyGT claim.',
        source_runtime_override='In this isolated new worker only, fit_control factory selects explicit128kind; original numerical source files untouched. Receiptscorrect actualkind/width/params. No monkeypatch in parent processes.',
        new_main_raw_records=6,new_dual_count_records=12,parent_raw_references=3,official_statistical_records=21,paired_effects=18,
        no_new_primary_or_posthoc_best_control_promotion=True,no_new_teacher_or_backbone_calls=True,
        combined_screen='All original8ANDthese2controls, originalstrict raw/BOTHcountpositive inall3cases. No criterion weakening.',
        all_data_DEV=True,no_fullsix_threecontribution_freshcost_success=True,goal_status='ACTIVE')
    atomic_json(PLAN,doc);atomic_json(PUBLIC/(STEM+'_registration.json'),doc)
    print(json.dumps(dict(status=doc['status'],extra_heads=6,records=21)),flush=True)


def registration():
    parent_registration();plan=json.loads(PLAN.read_text())
    assert file_sha(PARENT_PLAN)==plan['parent_registration_sha256']
    for n,sha in plan['source_sha256'].items():assert file_sha(ROOT/n)==sha,n
    return plan


def worker(pair):
    from flex_uda.active.deterministic_observation_runtime import configure_before_cuda
    configure_before_cuda()
    import numpy as np
    import torch
    import flex_uda.active.evidence_visual_learning as learning
    from flex_uda.a100.mechanism import generation_guard,PAIRS
    from flex_uda.active.joint_observation_inputs import JointObservationVideo,use_verified_local_features
    from flex_uda.active.factor_rsi_learning import Inputs
    from flex_uda.active.full_appearance_learning_inputs import augment_dense
    from flex_uda.presence.predictions import seal_mask,verify_predictions
    from scripts.a100_factor_rsi_primary_quality import subset_metadata,projection
    plan=registration();case=next(c for c in plan['cases'] if c['pair']==pair)
    guard=generation_guard();torch.set_num_threads(2);torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.35)
    folder=OUT/pair;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'completion.json').exists():return
    draws=json.loads((PARENT/pair/'primary/draws.json').read_text())['draws']
    original_make=learning.make;heads={};fits={}
    for name in COMPARATORS:
        kind='plain' if name=='FLAT_KD128' else 'visual'
        def factory(mean,std,device,kind=kind,width=128,outputs=1,_actual=kind):
            return original_make(mean,std,device,kind=_actual,width=128,outputs=outputs)
        learning.make=factory
        destination=folder/name;destination.mkdir(exist_ok=True)
        try:head,receipt=learning.fit_control(case,draws,'SOFT_KD',destination,'cuda:0')
        finally:learning.make=original_make
        receipt.update(mode=name,width=128,architecture=kind,ordinary_direct_teacher_BCE=True)
        saved=torch.load(destination/'SOFT_KD.pt',map_location='cpu',weights_only=False);saved['receipt']=receipt
        learning.save(destination/'SOFT_KD.pt',saved);heads[name]=head;fits[name]=receipt
        print(json.dumps(dict(status='SAME_CAPACITY_ORDINARY_ADEQUATE',pair=pair,method=name,fit=receipt)),flush=True)
    use_verified_local_features();video=JointObservationVideo(pair,case['evaluated_video']);inputs=Inputs(video)
    n,nq=len(video.obs),len(video.queries);scores={name:np.empty((nq,n),dtype='float32') for name in heads}
    with torch.no_grad():
        for qi in range(nq):
            for begin in range(0,n,512):
                rows=np.arange(begin,min(begin+512,n));records=[dict(local_query=qi,local_track=int(video.columns[r]),frame_id=int(video.obs[r,0]),track_id=int(video.obs[r,1])) for r in rows]
                a=inputs.arrays(records);base=np.column_stack((a['bags'].reshape(len(rows),-1),a['present'],a['current'],a['coarse'])).astype('float32')
                x=learning.tensors(augment_dense(base,video,rows),'cuda:0')
                for name,head in heads.items():scores[name][qi,rows]=head(x).cpu().numpy()
    source=PARENT/pair/'predictions'/PRIMARY;verify_predictions(source)
    scores[PRIMARY]=np.load(source/'scores.npy');masks={name:value>0 for name,value in scores.items()}
    assert np.array_equal(masks[PRIMARY],np.load(source/'mask.npy'))
    metadata=subset_metadata(case['target'],video.video);tasks=[dict(pair=pair,method=PRIMARY,folder=str(source))]
    for name in COMPARATORS:
        path=folder/'predictions'/name;path.mkdir(parents=True,exist_ok=True)
        np.save(path/'scores.npy',scores[name]);np.save(path/'mask.npy',masks[name])
        seal_mask(path,metadata,masks[name].flatten(),target=case['target'],model=PAIRS[pair][0],method=name,GT_open_attempts=len(guard))
        tasks.append(dict(pair=pair,method=name,folder=str(path)))
    for rank,count in [(PRIMARY,c) for c in COMPARATORS]+[(c,PRIMARY) for c in COMPARATORS]:
        name=rank+'__AT_'+count+'_COUNT';path=folder/'predictions'/name
        select=projection(scores[rank],masks[count],video.obs)
        seal_mask(path,metadata,select.flatten(),target=case['target'],model=PAIRS[pair][0],method=name,GT_open_attempts=len(guard),extra=dict(rank_method=rank,count_method=count,counts_equal_per_query_frame=True,counts_from_GT=False))
        tasks.append(dict(pair=pair,method=name,folder=str(path)))
    assert not guard and len(tasks)==7
    atomic_json(folder/'completion.json',dict(tasks=tasks,fits=fits,GT_open_attempts=0,parent_predictions_unchanged=True))


def drain(p,path):
    with path.open('a',buffering=1) as stream:
        for line in p.stdout:stream.write(line);print(line,end='',flush=True)


def main():
    plan=registration();sup=OUT/'supervisor';sup.mkdir(parents=True,exist_ok=True)
    print('WAITING_PARENT75SEAL_FOR_ORIGINAL_GPU_RELEASE_NO_GT',flush=True)
    while not (PARENT/'all75_sealed_before_GT.json').exists():
        failures=[p for p in (PARENT/'GPU2_failed.json',*[PARENT/c['pair']/'failed.json' for c in plan['cases']]) if p.exists()]
        if failures:raise RuntimeError([str(p) for p in failures])
        time.sleep(2)
    jobs=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        for case,gpu in zip(plan['cases'],(0,1,3)):
            pair=case['pair']
            if (OUT/pair/'completion.json').exists():continue
            owners=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-compute-apps=pid,process_name','--format=csv,noheader,nounits'],text=True).strip()
            assert not owners,(gpu,owners)
            p=subprocess.Popen([sys.executable,'-u',str(Path(__file__)),'--worker',pair],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'))
            future=pool.submit(drain,p,sup/(pair+'.log'));jobs.append((pair,p,future))
            atomic_json(sup/(pair+'_launch.json'),dict(pid=p.pid,GPU=gpu));print(json.dumps(dict(status='STARTED_NECESSARY_SAME_CAPACITY_CONTROL',pair=pair,GPU=gpu,pid=p.pid)),flush=True)
        failed=[]
        for pair,p,future in jobs:
            code=p.wait();future.result()
            if code:failed.append([pair,code])
    assert not failed,failed
    tasks=[]
    from flex_uda.presence.predictions import verify_predictions
    for case in plan['cases']:
        doc=json.loads((OUT/case['pair']/'completion.json').read_text());assert doc['GT_open_attempts']==0;tasks.extend(doc['tasks'])
    for task in tasks:verify_predictions(task['folder'])
    assert len(tasks)==21
    atomic_json(OUT/'all21_sealed_before_GT.json',dict(tasks=tasks,GT_open_attempts=0,separate_parent_GT_barrier=True))
    print('ALL21_SAME_CAPACITY_REFERENCES_AND_OUTPUTS_SEALED_NO_OWN_GT',flush=True)
    while not (PUBLIC/'evidence_visual_primary_20261125_reproduction.json').exists():time.sleep(2)
    from scripts.a100_joint_teacher_noise_pilot import evaluate_one
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:completed=list(pool.map(evaluate_one,tasks))
    results={c['pair']:{} for c in plan['cases']}
    for task,result in completed:results[task['pair']][task['method']]=result
    comparisons=[]
    for pair,values in results.items():
        primary=values[PRIMARY]['HOTA_unrounded']
        for other in COMPARATORS:
            ref=values[other]['HOTA_unrounded']
            comparisons.append(dict(pair=pair,comparator=other,primary_HOTA=primary,ordinary_HOTA=ref,raw_delta_HOTA=primary-ref,
                primary_at_ordinary_count_delta_HOTA=values[PRIMARY+'__AT_'+other+'_COUNT']['HOTA_unrounded']-ref,
                ordinary_at_primary_count_delta_HOTA=primary-values[other+'__AT_'+PRIMARY+'_COUNT']['HOTA_unrounded']))
    eligible=all(min(r['raw_delta_HOTA'],r['primary_at_ordinary_count_delta_HOTA'],r['ordinary_at_primary_count_delta_HOTA'])>0 for r in comparisons)
    parent=json.loads((PARENT/'quality_results.json').read_text())
    atomic_json(OUT/'quality_results.json',dict(status='COMPLETE_NECESSARY_SAME_CAPACITY_CONTROLS',results=results,comparisons=comparisons,
        candidate_eligible_for_larger_screen=eligible,combined_with_parent_all10_eligible=eligible and parent['candidate_eligible_for_larger_screen'],
        no_new_primary_or_posthoc_promotion=True,all_data_DEV=True,goal_status='ACTIVE'))
    from scripts.a100_primary_statistics_common import collect,reproduce
    collect(OUT,PUBLIC,STEM,21);reproduce(PUBLIC,STEM,21,18)
    print(json.dumps(dict(status='COMPLETE_SAME_CAPACITY21_AND18_REPRODUCED',comparisons=comparisons)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--worker');a=p.parse_args()
    if a.prepare:prepare()
    elif a.worker:worker(a.worker)
    else:main()
