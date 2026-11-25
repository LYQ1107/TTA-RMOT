"""Adequate ordinary raw-teacher KD with the same available appearance cues.

Supplement to the immutable fixed-view core: one additional ordinary head,
not a changed primary. No new teacher/encoder or target-label input. Its own
90 raw/count outputs are sealed before any supplement official evaluation.
"""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from flex_uda.a100.mechanism import PAIRS,metadata,generation_guard
from scripts.a100_six_direction import SEEDS,PUBLIC,source
from scripts.a100_view_fusion import OUT as CORE,labels

OUT=CORE/'flat_rgb_supplement';STATE=OUT/'supervisor/state.json'
SUCCESS='PASS_MULTIVIEW_FLAT_RGB_BASELINE_RESEARCH_GOAL_ACTIVE'
BASE='RAW_RGB_FINAL';FULL='DS_FULL'


def conditions():return [dict(pair=p,seed=s,folder=str(OUT/'quality'/p/f'seed{s}')) for s in SEEDS for p in PAIRS]


def run(pair,seed):
    import torch
    from flex_uda.active.confidence_teacher import fit_head
    from flex_uda.active.context_cohort import distillation_features
    from flex_uda.active.semantic_colour import appearance_cues
    from flex_uda.active.view_evidence import targets
    from flex_uda.presence.predictions import seal_mask
    from scripts.a100_visual_classical_strict_prefix import plain,immediate_predictions
    guard=generation_guard();torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    root=OUT/'quality'/pair/f'seed{seed}';core=CORE/'quality'/pair/f'seed{seed}'
    assert (core/'completion.json').is_file()
    if (root/'completion.json').is_file():return
    root.mkdir(parents=True,exist_ok=True);start=time.perf_counter();model,target=PAIRS[pair];m=metadata(target)
    if not (root/FULL).exists():(root/FULL).symlink_to(core/FULL,target_is_directory=True)
    total=np.zeros(1081);square=np.zeros(1081);rows=0;bx=[];bv=[];receipts=[];next_video=0;state=root/'stream.pt'
    if state.is_file():
        saved=torch.load(state,map_location='cpu',weights_only=False)
        total,square,rows,bx,bv,receipts,next_video=[saved[k] for k in ('total','square','rows','bx','bv','receipts','next_video')]
    for vi,(video,bank) in enumerate(sorted(m['videos'].items())):
        if vi<next_video:continue
        begin=time.perf_counter();queries=[q for q in m['episodes'] if q['video_id']==video];texts=[q['query_text'] for q in queries]
        with np.load(source(pair)/'source'/video/'sparse_inputs.npz') as a:pcd,colours,prior,tracks=[a[k] for k in ('pcd','colours','prior','tracks')]
        obs=np.asarray(bank['observations']);lengths=np.asarray([(obs[:,1]==t).sum() for t in tracks])
        cue,available,_,_=appearance_cues(colours,texts,lengths)
        raw=np.concatenate([distillation_features(pcd,colours,texts,'extended'),
            np.where(available,2*cue-1,0).astype('float32')[:,:,None]],axis=2)
        assert raw.shape==prior.shape+(1081,) and np.isfinite(raw).all()
        cells,probability,evidence=labels(pair,video,queries,tracks);qidx,tidx=cells.T
        bx.append(raw[qidx,tidx]);bv.append(probability);current=raw.reshape(-1,1081)
        total+=current.sum(0,dtype=float);square+=np.square(current.astype(float)).sum(0);rows+=len(current)
        mean=total/rows;std=np.maximum(np.sqrt(np.maximum(square/rows-mean**2,0)),1e-3)
        teacher,_,_,_=targets(np.concatenate(bv));weight=1.+3.*np.abs(2.*teacher-1.);x=np.concatenate(bx)
        predictions,fit,checkpoint=fit_head(x,np.zeros(len(x)),np.clip(teacher,.02,.98),current,
            np.zeros(prior.size),mean,std,sample_weight=weight,soft=True,seed=seed)
        fit.update(architecture='1081->64 GELU->1 direct raw-teacher head; fixed source logit zero',
            RGB_and_query_cue_supplied_without_structural_inference=True)
        torch.save(checkpoint,root/'RAW_RGB_latest_head.pt');score=predictions['CONVERGED'].reshape(prior.shape)
        lookup={int(t):i for i,t in enumerate(tracks)};columns=np.asarray([lookup[int(t)] for _,t in obs])
        bits=plain(score[:,columns],False);dest=root/BASE/'videos'/video;dest.mkdir(parents=True,exist_ok=True)
        np.save(dest/'mask.npy',bits);immediate_predictions(root/BASE,type('Context',(),dict(bank=bank,queries=queries))(),bits)
        with np.load(core/'videos'/video/'coarse_scores.npz') as a:
            assert np.array_equal(tracks,a['tracks']);full=a[FULL]
        local=root/'videos'/video;local.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(local/'coarse_scores.npz',tracks=tracks,DS_FULL=full,RAW_RGB_FINAL=score)
        receipt=dict(video=video,index=vi,fits={'RAW_RGB':fit},evidence=evidence,wall_seconds=time.perf_counter()-begin,
            GT_open_attempts=len(guard),current_video_transductive=True,no_future_teacher_or_normalization=True)
        atomic_json(local/'receipt.json',receipt);receipts.append(receipt)
        saved=dict(total=total,square=square,rows=rows,bx=bx,bv=bv,receipts=receipts,next_video=vi+1)
        temp=state.with_suffix('.tmp');torch.save(saved,temp);temp.replace(state)
        print(json.dumps(dict(event='FLAT_RGB_ORDINARY_VIDEO',pair=pair,seed=seed,completed=vi+1,total=len(m['videos']))),flush=True)
    mask=np.zeros(m['candidate_query_rows'],bool)
    for video in sorted(m['videos']):
        bits=np.load(root/BASE/'videos'/video/'mask.npy')
        for qi,q in enumerate(q for q in m['episodes'] if q['video_id']==video):mask[q['start']:q['stop']]=bits[qi]
    seal=seal_mask(root/BASE,m,mask,target=target,model=model,method=BASE,GT_open_attempts=len(guard),
        extra=dict(same_fixed_multiview_QA_as_core=True,no_structural_projection_or_teacher_reliability_model=True,
            head_features='768D PCD +312D query-conditioned RGB +1D same appearance cue',independent_decoder=True))
    assert not guard
    atomic_json(root/'completion.json',dict(status='SEALED_MULTIVIEW_FLAT_RGB_BASELINE',pair=pair,seed=seed,
        methods=[dict(method=BASE,rows=seal['prediction_rows'])],receipts=receipts,GT_open_attempts=0,
        new_teacher_encoder_calls=0,actual_head_fits=len(receipts),wall_seconds=time.perf_counter()-start))


def counts(record):
    # Reuse the already fixed independent-decoder count algorithm in this new
    # process, with the two method names local to this supplement. No source
    # files or other processes' module state are changed.
    import scripts.a100_view_fusion_launch as common
    common.PRIMARY=FULL;common.BASELINES=(BASE,)
    return common.counts(record)


def analyze():
    from scripts.a100_visual_classical_analyze import read_stats
    from scripts.a100_six_direction_analyze import comparison
    from scripts.a100_visual_classical_sparse_observation_analyze import fit_audit
    from scripts.analyze_pf_video_uncertainty import pooled_hota
    assert json.loads((OUT/'all_predictions_sealed_before_GT.json').read_text())['methods']==90
    contrasts={'FULL_vs_FLAT_RGB':(FULL,BASE),'RANK_AT_FULL_COUNT':(FULL+'__AT_'+FULL+'_COUNT',BASE+'__AT_'+FULL+'_COUNT'),
        'RANK_AT_RGB_COUNT':(FULL+'__AT_'+BASE+'_COUNT',BASE+'__AT_'+BASE+'_COUNT')}
    names=sorted({n for pair in contrasts.values() for n in pair});data={};sufficient={};fits={};error=0.;fit_count=0
    for r in conditions():
        root=Path(r['folder']);values={n:read_stats(root/'countmatched'/n if '__AT_' in n else root/n) for n in names}
        completion=json.loads((root/'completion.json').read_text());fit_count+=completion['actual_head_fits']
        data[r['pair'],r['seed']]=values;fits[r['pair']+f"_seed{r['seed']}"]=fit_audit(completion['receipts'])
        for name,row in values.items():
            num,den=row['parent_stats'];error=max(error,abs(float(pooled_hota((num,den),np.ones(len(num))))-row['HOTA']))
            sufficient[r['pair']+f"_seed{r['seed']}/"+name]=dict(parents=row['parent_videos'],HOTA=row['HOTA'],numerator=num.tolist(),denominator=den.tolist())
    assert error<1e-8,error
    assert fit_count==444,fit_count
    repeated={p:dict(mean_HOTA={n:float(np.mean([data[p,s][n]['HOTA'] for s in SEEDS])) for n in (FULL,BASE)},
        comparisons={n:comparison([data[p,s][a] for s in SEEDS],[data[p,s][b] for s in SEEDS]) for n,(a,b) in contrasts.items()}) for p in PAIRS}
    result=dict(status='MULTIVIEW_FLAT_RGB_ORDINARY_BASELINE_ANALYZED',repeated=repeated,fit_audit=fits,
        all90_supplement_outputs_sealed_before_supplement_GT=True,core468_separate_GT_group=True,
        new_teacher_encoder_calls=0,actual_head_fits_total=fit_count,HOTA_parent_sum_max_error=error,
        primary_unchanged=True,no_posthoc_best_arm_promotion=True,all_data_development=True,goal_status='ACTIVE')
    atomic_json(OUT/'analysis.json',result)
    (PUBLIC/'view_fusion_flat_rgb_analysis.json').write_text(json.dumps(result,separators=(',',':'))+'\n')
    (PUBLIC/'view_fusion_flat_rgb_parent_statistics.json').write_text(json.dumps(sufficient,separators=(',',':'))+'\n')
    print(json.dumps(dict(event='FLAT_RGB_ORDINARY_ANALYSIS',repeated=repeated,goal_status='ACTIVE')),flush=True)


def core_GPU_finished():
    path=CORE/'supervisor/state.json'
    if not path.is_file():return False
    return json.loads(path.read_text())['status'] in ('ALL18_FIXED_VIEW_FUSION_RAW_SEALED_GENERATING_COUNTS_NO_GT',
        'ALL468_FIXED_VIEW_FUSION_OUTPUTS_SEALED_OFFICIAL_EVALUATING','FIXED_VIEW_FUSION_OFFICIAL_COMPLETE_ANALYZING',
        'PASS_FIXED_VIEW_FUSION_SIX_SETTINGS_RESEARCH_GOAL_ACTIVE')


def main():
    plan=json.loads((PUBLIC/'view_fusion_flat_rgb_registration_v1.json').read_text())
    assert plan['source_sha256']==file_sha(Path(__file__))
    pending=conditions();done=[];active={};failure=None;start=time.perf_counter();last=0.
    def state(status):atomic_json(STATE,dict(status=status,pid=os.getpid(),completed=len(done),
        finished=done,active={str(g):j for g,(_,_,j) in active.items()},pending=pending,failure=failure,
        wall_seconds=time.perf_counter()-start,goal_status='ACTIVE'))
    state('AWAITING_CORE_MULTIVIEW_GPU_STAGE_RELEASE_NO_GT')
    while not core_GPU_finished():time.sleep(5)
    while pending or active:
        changed=False
        for gpu,(proc,handle,job) in list(active.items()):
            code=proc.poll()
            if code is None:continue
            handle.close();del active[gpu]
            if code or not (Path(job['folder'])/'completion.json').is_file():failure=dict(job=job,exit_code=code,last_log=Path(job['log']).read_text()[-6000:])
            else:done.append(job)
            changed=True
        candidates=[g for g in (0,1,2,3) if g not in active]
        if pending and candidates and not failure:
            output=subprocess.check_output(['nvidia-smi','-i',','.join(map(str,candidates)),'--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
            memory={int(r.split(',')[0]):int(r.split(',')[1]) for r in output.strip().splitlines()}
            for gpu in candidates:
                if not pending or memory[gpu]>=512:continue
                job=pending.pop(0)
                if (Path(job['folder'])/'completion.json').is_file():done.append(dict(job,reused_sealed=True));changed=True;continue
                log=OUT/'supervisor'/(job['pair']+'_seed'+str(job['seed'])+'.log');handle=log.open('a',buffering=1)
                proc=subprocess.Popen([sys.executable,'-u',str(Path(__file__)),'--pair',job['pair'],'--seed',str(job['seed'])],
                    cwd=ROOT,stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4'))
                active[gpu]=proc,handle,dict(job,pid=proc.pid,log=str(log),physical_GPU=gpu);changed=True
        if changed or time.perf_counter()-last>45:
            state('RUNNING_MULTIVIEW_FLAT_RGB_BASELINES_NO_GT' if not failure else 'PRESERVING_ACTIVE_FLAT_RGB_AFTER_FAILURE');last=time.perf_counter()
        if failure and not active:raise RuntimeError(json.dumps(failure))
        if pending or active:time.sleep(3)
    assert len(done)==18
    state('ALL18_FLAT_RGB_BASELINES_SEALED_GENERATING_COUNTS_NO_GT')
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:roots=list(pool.map(counts,conditions()))
    tasks=[]
    for r,count_root in zip(conditions(),roots):
        for folder in (Path(r['folder']),Path(count_root)):
            row=json.loads((folder/'completion.json').read_text());assert row['GT_open_attempts']==0
            assert not any(row.get('CountSelf_changed_query_files',{}).values())
            tasks.extend((str(folder/e['method']),r['pair']) for e in row['methods'])
    assert len(tasks)==90
    atomic_json(OUT/'all_predictions_sealed_before_GT.json',dict(methods=90,conditions=18,GT_open_attempts=0))
    state('ALL90_FLAT_RGB_OUTPUTS_SEALED_AWAITING_CORE_ANALYSIS')
    while not (CORE/'analysis.json').is_file():time.sleep(5)
    from scripts.a100_visual_classical_sparse_observation_evaluate import evaluate_one
    state('ALL90_FLAT_RGB_OUTPUTS_SEALED_OFFICIAL_EVALUATING')
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:results=dict(pool.map(evaluate_one,tasks))
    atomic_json(OUT/'official.json',dict(methods=90,results=results,goal_status='ACTIVE'))
    analyze();state(SUCCESS);print(SUCCESS,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--pair',choices=tuple(PAIRS));parser.add_argument('--seed',type=int);args=parser.parse_args()
    if args.pair:
        import fcntl
        root=OUT/'quality'/args.pair/f'seed{args.seed}';root.mkdir(parents=True,exist_ok=True)
        with (root/'condition.lock').open('a') as lock:fcntl.flock(lock,fcntl.LOCK_EX);run(args.pair,args.seed)
    else:
        try:main()
        except Exception as error:
            previous=json.loads(STATE.read_text()) if STATE.is_file() else {}
            previous.update(status='FAILED_FLAT_RGB_BASELINE_REQUIRES_ROOT_CAUSE',error=repr(error),goal_status='ACTIVE')
            atomic_json(STATE,previous);raise
