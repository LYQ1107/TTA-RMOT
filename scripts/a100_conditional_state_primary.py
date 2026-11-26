"""Full148-unit fixed-evidence primary; no quality access before all seals.

This study tests the conditional-state learning objective. Shared-memory KD
controls receive the final primary observer as well as identical teacher events,
so merely adding the new decoder cannot establish a learning contribution.
It is fixed-QA, full-video transductive DEV research, not fresh active selection
or a cold deployment cost comparison. Old numerical sources stay immutable.
"""
import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json,file_sha

BASE=ROOT/'a100_runs/visual_classical_20261121/six_direction_20261122/fullscope_visual_six_20261126'
OUT=ROOT/'a100_runs/visual_classical_20261121/six_direction_20261122/conditional_state_20261127/primary'
PLAN=OUT/'registration.json'
PUBLIC=ROOT/'docs/a100_six_direction_20261122'
STEM='conditional_state_primary_20261127'
PRIMARY='CONDITIONAL_STATE_MEMORY'
COMPARATORS=('SOFT_KD','HARD_KD','SOFT_POOLED','SOFT_SHARED_MEMORY',
             'HARD_SHARED_MEMORY','FUSED_KD','FUSED_SHARED_MEMORY')
METHODS=(PRIMARY,*COMPARATORS)
EXTRA_SOURCES=('scripts/a100_conditional_state_primary.py',
    'flex_uda/active/conditional_state_runtime.py','flex_uda/active/conditional_state_learning.py',
    'scripts/a100_conditional_state_engineering.py','scripts/a100_conditional_state_primary_contract.py',
    'scripts/a100_conditional_state_statistics.py','scripts/a100_factor_rsi_primary_quality.py',
    'flex_uda/presence/predictions.py','flex_uda/a100/evaluation_reuse.py')


def prepare():
    from flex_uda.a100.mechanism import generation_guard
    guard=generation_guard()
    if PLAN.exists():raise FileExistsError(PLAN)
    parent=json.loads((BASE/'registration.json').read_text())
    engineering=OUT.parent/'engineering/completion.json'
    assert json.loads(engineering.read_text())['status']=='COMPLETE_ALL6_SETTING_REAL_INPUT_ENGINEERING'
    contract=ROOT/'a100_runs/visual_classical_20261121/supervisor/conditional_state_primary_contract_20261127/result.json'
    assert json.loads(contract.read_text())['status'].startswith('PASS_')
    cases=[];paths={BASE/'registration.json',BASE/'all4588_sealed_before_GT.json',engineering,contract}
    for old in parent['cases']:
        case=dict(old);folder=BASE/'units'/case['pair']/case['evaluated_video']
        case.update(parent=str(folder),ordinary_model=str(folder/'ordinary/UNIFORM_SOFT_KD288/UNIFORM_SOFT_KD.pt'),
            original_scores=str(folder/'predictions/UNIFORM_SOFT_KD288/scores.npy'),
            uniform_answers=str(folder/'uniform_answers.jsonl'),uniform_draws=str(folder/'uniform_draws.json'))
        paths.update(Path(case[k]) for k in ('ordinary_model','original_scores','uniform_answers','uniform_draws',
            'observations','temporal_queries','auxiliary_path'))
        paths.update((folder/'unlabeled_normalizer.npz',folder/'generation_complete.json'))
        cases.append(case)
    assert len(cases)==148 and len({c['pair'] for c in cases})==6
    sources=dict(parent['source_sha256'])
    for name,sha in sources.items():assert file_sha(ROOT/name)==sha,name
    sources.update({name:file_sha(ROOT/name) for name in EXTRA_SOURCES})
    plan=dict(status='PREREGISTERED_ALL148_CONDITIONAL_STATE_PRIMARY_BEFORE_FITS',primary=PRIMARY,
        comparators=COMPARATORS,cases=cases,source_sha256=sources,
        input_sha256={str(p):file_sha(p) for p in sorted(paths)},parent_registration_sha256=file_sha(BASE/'registration.json'),
        all_parent_teacher_and_output_seals_sha256=file_sha(BASE/'all4588_sealed_before_GT.json'),
        mathematical_contract_sha256=file_sha(contract),engineering_completion_sha256=file_sha(engineering),
        hypothesis='A learned copy/reset state model, noisy teacher channel and externally observed neural feedback improve referring adaptation beyond adequate pointwise or fused KD with the SAME protocol and memory readout.',
        initialization='Each original adequate UNIFORM_SOFT_KD288, not an engineering checkpoint; identical9352D architecture and metadata-only normalizer for all ordinary heads.',
        rounds=4,solver_steps_per_round=1500,observer_steps_per_round=1500,learning_rate=.0001,
        unlabeled_MC_nodes_per_round=8192,seed=0,
        objective='Negative query/budget weighted structured decision ELBO + .25 AUX soft BCE + .001 mean KL(clamped base keep || keep) + .01 mean KL(.8 || channel). Identical successful engineering objective; primary not forced to copy each noisy decision under a raw .005 constraint.',
        primary_state_learning='Solver learns expected reset positive/negative masses; observer learns expected copy/reset and per-view decision channel. Both use frozen current E statistics, then recompute the complete objective for fixed-fraction backtracking. EM is prior art, not an RSI quality guarantee.',
        ordinary_minimum_additional_steps=6000,ordinary_maximum_additional_steps=16000,
        ordinary_proper_excess_KL_limit=.005,
        ordinary='Three same-architecture288 heads each start the same original KD and fit6000additional steps, with up to16000additional if needed to satisfy original proper excessKL.005; original inherited steps and extra steps recorded separately. Raw soft, hard, and fused targets explicitly distinct; no weakening of teacher baselines.',
        same_information='Same target frames/candidates/query/9352D inputs/100prepaid three-view AUX/fixed object-balanced UNIFORM answers for every head. No new teacher calls or target manual labels.',
        shared_memory_controls='Soft/hard/fused KD get the SAME final primary observer, protocol, original event scope and answers at readout. This is an intentionally privileged comparator, preventing a decoder-only gain claim.',
        fused_control='Uninformative per-node solver prior.5 plus primary finalR and original scopes yields fused paid-anchor targets. AUX raw soft labels retained. This comparator uses learnedR and is not independent pure-teacher evidence.',
        scope='CLIP uniform latent time over distinct actual input observations; repeated input provenance retained; structuredELBO not exactclip marginal or verified Qwen semantics. CROP/frame point-scoped. One likelihood per unique event.',
        protocol='Existing metadata-only v8P query-set heuristic; CHECKLIST video-existence mixture with prior.5, PRESUPPOSED video existence fixed. Neither forces any frame/candidate positive. No GT-derived absence frequency.',
        assumptions=['Oriented(.5,.99) teacher sensitivity/specificity is an assumption, not identified semantic truth.',
            'Observed image/state changes need not detect every ID switch or action change.',
            'Same-Qwen event errors may be correlated; this model is not a proof that they are independent.',
            'Objective descent and parameter movement do not prove HOTA or causal RSI improvement.'],
        MC='8192fresh uniform query-observation samples plus each asked anchor, reciprocal expected multiplicity weights, full E-mass normalizer. Not a fixed32-frame recomputation or an oracle subset.',
        recovery='Atomic completed-round S/R, actual accepted scores and NumPy/Torch/CUDA RNG; unfinished round starts from the same last complete state. Ordinary Adam+step counters saved every500. Completed units not regenerated. All attempts/costs retained.',
        GPUs=[0,1,2,3],GPU_memory_fraction=.20,existing_GPU2_service_preserved=True,
        output_methods_per_unit=22,total_outputs=3256,whole_setting_HOTA_records=132,paired_effects=126,
        success='Primary raw and BOTH CountMatched summary effects positive against ALL7 controls in EACH of6 complete settings. No posthoc best-control promotion.',
        evaluation='Only after all3256 outputs are sealed and workers exit, isolated official TrackEval; sumTPweightedAssA/TPFPFN over all videos. Independent public JSON/NumPy reproduction of132HOTA/126effects.',
        cost='Record actual new phases and attempted/resumed costs. Original source features, original teacher answers and inherited KD fits are prepaid; not fresh cold deployment or equal time.',
        old42_sources_unchanged=True,manual_GT_never_used_for_adaptation_or_model_selection=True,
        all_data_DEV=True,no_module_ablation_matrix=True,goal_status='ACTIVE',goal_complete=False)
    assert not guard;atomic_json(PLAN,plan)
    compact={k:v for k,v in plan.items() if k not in ('cases','input_sha256')}
    compact.update(video_units=148,setting_video_counts={p:sum(c['pair']==p for c in cases) for p in sorted({c['pair'] for c in cases})})
    atomic_json(PUBLIC/(STEM+'_registration.json'),compact)
    print(json.dumps(dict(status=plan['status'],units=148,outputs=3256,pins=len(sources),inputs=len(paths))),flush=True)


def read_plan(*,inputs=False):
    plan=json.loads(PLAN.read_text())
    for name,sha in plan['source_sha256'].items():assert file_sha(ROOT/name)==sha,name
    if inputs:
        for name,sha in plan['input_sha256'].items():assert file_sha(name)==sha,name
    plan['registration_sha256']=file_sha(PLAN)
    return plan


def state(model):return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}


def array_sha(value):
    import numpy as np
    return hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()


def ordinary_fit(context,case,mode,main_target,folder,plan):
    import numpy as np
    import torch
    from flex_uda.active import evidence_visual_learning as core
    folder=Path(folder);folder.mkdir(exist_ok=True,parents=True)
    saved=torch.load(case['ordinary_model'],map_location='cpu',weights_only=False)
    inherited=saved['receipt'];assert inherited['total_steps']>=6000 and inherited['proper_teacher_excess_CE']<=.005
    model=core.make(context.mean,context.std,'cuda:0',kind='plain',width=288);model.load_state_dict(saved['model'])
    draws=json.loads(Path(case['uniform_draws']).read_text())['draws']
    indices=np.array([int(row['uid']) for row in draws]);queries=indices//context.n
    targets=np.asarray(main_target,dtype=float);assert targets.shape==(len(draws),)
    auxiliary=context.ay.copy()
    if mode=='HARD_KD':auxiliary=(auxiliary>=.5).astype(float)
    counts=np.bincount(queries,minlength=context.nq);assert np.all(counts>0)
    x=np.r_[context.features[indices],context.auxiliary]
    y=np.r_[targets,auxiliary]
    w=np.r_[1/(context.nq*counts[queries]),np.full(len(auxiliary),.25/len(auxiliary))]/1.25
    target_sha=array_sha(y);final=folder/'final.pt';partial=folder/'partial.pt'
    if final.exists():
        done=torch.load(final,map_location='cpu',weights_only=False)
        assert done['registration_sha256']==plan['registration_sha256'] and done['target_sha256']==target_sha
        assert done['receipt']['additional_steps']>=plan['ordinary_minimum_additional_steps']
        assert done['receipt']['proper_teacher_excess_CE']<=plan['ordinary_proper_excess_KL_limit']
        model.load_state_dict(done['model']);return model,done['receipt']
    optimizer=torch.optim.Adam(model.parameters(),lr=plan['learning_rate']);completed=0;phases=[]
    if partial.exists():
        own=torch.load(partial,map_location='cpu',weights_only=False)
        assert own['registration_sha256']==plan['registration_sha256'] and own['target_sha256']==target_sha
        model.load_state_dict(own['model']);optimizer.load_state_dict(own['optimizer']);completed=own['completed'];phases=own['phases']
    from flex_uda.active.conditional_state_runtime import save_round
    while True:
        error=core.excess(model,x,y,w,'cuda:0')
        if completed>=plan['ordinary_minimum_additional_steps'] and error<=plan['ordinary_proper_excess_KL_limit']:break
        if completed>=plan['ordinary_maximum_additional_steps']:
            atomic_json(folder/'numerical_failure.json',dict(mode=mode,completed=completed,proper_excess_KL=error,targets_sha256=target_sha))
            raise RuntimeError(('Ordinary sufficiency failed',case['pair'],case['evaluated_video'],mode,completed,error))
        optimizer,phase=core.fit(model,x,y,w,500,'cuda:0',optimizer)
        completed+=500;phases.append(phase)
        save_round(partial,dict(model=state(model),optimizer=optimizer.state_dict(),completed=completed,phases=phases,
            registration_sha256=plan['registration_sha256'],target_sha256=target_sha))
        print(json.dumps(dict(status='CONDITIONAL_STATE_ORDINARY_FIT',pair=case['pair'],video=case['evaluated_video'],mode=mode,additional_steps=completed)),flush=True)
    receipt=dict(mode=mode,additional_steps=completed,inherited_steps=inherited['total_steps'],
        total_steps=inherited['total_steps']+completed,proper_teacher_excess_CE=error,adequacy_limit=.005,
        minimum_additional_steps=plan['ordinary_minimum_additional_steps'],target_sha256=target_sha,phase_fits=phases,
        inherited_model_sha256=file_sha(case['ordinary_model']),same9352D_and_same_architecture_as_primaryS=True)
    save_round(final,dict(model=state(model),receipt=receipt,target_sha256=target_sha,registration_sha256=plan['registration_sha256']))
    atomic_json(folder/'fit_receipt.json',receipt);return model,receipt


def probability_scores(probability):
    import numpy as np
    p=np.asarray(probability,dtype=float)
    assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
    return (np.log(np.maximum(p,np.finfo(float).tiny))-
            np.log1p(-np.minimum(p,np.nextafter(1.,0.)))).astype('float32')


def inference(context,models):
    import numpy as np
    import torch
    from flex_uda.active import evidence_visual_learning as core
    result={name:np.empty(context.n*context.nq,dtype='float32') for name in models}
    with torch.no_grad():
        for begin in range(0,context.n*context.nq,512):
            rows=np.arange(begin,min(begin+512,context.n*context.nq));x=core.tensors(context.features[rows],'cuda:0')
            for name,model in models.items():result[name][rows]=model(x).cpu().numpy()
    return {name:x.reshape(context.nq,context.n) for name,x in result.items()}


def run_unit(case,gpu,plan,guard):
    import numpy as np
    import torch
    from scipy.special import expit
    from flex_uda.active.conditional_state_runtime import learn,load_round
    from flex_uda.active.fullscope_visual_runtime import Context
    from flex_uda.active.conditional_state_learning import StateObserver,scoped_bound
    from scripts.a100_conditional_state_engineering import source_events
    from flex_uda.presence.predictions import seal_mask,verify_predictions
    from scripts.a100_factor_rsi_primary_quality import subset_metadata,projection
    from flex_uda.a100.mechanism import PAIRS
    folder=OUT/'units'/case['pair']/case['evaluated_video'];folder.mkdir(parents=True,exist_ok=True)
    finished=folder/'generation_complete.json'
    if finished.exists():
        done=json.loads(finished.read_text());assert done['registration_sha256']==plan['registration_sha256']
        for task in done['tasks']:verify_predictions(task['folder'])
        return done
    started=time.perf_counter();fitroot=folder/'primary'
    learned=learn(case,gpu,plan,fitroot)
    checkpoint=load_round(fitroot/'latest.pt',plan['registration_sha256'])
    assert len(checkpoint['rounds'])==plan['rounds']
    assert learned['solver_attempted_additional_steps']>=6000 and learned['observer_attempted_steps']>=6000
    context=Context(case,fitroot,None,gpu);events,provenance=source_events(context,case)
    assert len(events)==sum(context.quotas.values())
    assert file_sha(fitroot/'memory_inputs.npz')==learned['memory_inputs_sha256']
    with np.load(fitroot/'memory_inputs.npz') as saved:change,base=saved['change'].copy(),saved['base'].copy()
    observer=StateObserver().to('cuda:0');observer.load_state_dict(checkpoint['observer']);observer.eval()
    with torch.no_grad():
        keep=observer(torch.as_tensor(change.reshape(-1,8),device='cuda:0'),torch.as_tensor(base.reshape(-1),device='cuda:0')).cpu().numpy().reshape(base.shape)
        reliability=observer.reliability().cpu().numpy()
    def memory(scores):
        return scoped_bound(expit(scores),keep,context.video.obs,events,reliability,
            protocol=learned['protocol'],existence_prior=.5,iterations=8)['posterior']
    primary_scores=checkpoint['accepted_scores']
    scores={PRIMARY:probability_scores(memory(primary_scores))}
    fused=memory(np.zeros_like(primary_scores))
    draws=json.loads(Path(case['uniform_draws']).read_text())['draws'];indices=np.array([int(r['uid']) for r in draws])
    raw=np.array([float(r['probability']) for r in draws])
    targets={'SOFT_KD':raw,'HARD_KD':(raw>=.5).astype(float),'FUSED_KD':fused.reshape(-1)[indices]}
    heads={};fits={};ordinary_started=time.perf_counter()
    for mode,target in targets.items():heads[mode],fits[mode]=ordinary_fit(context,case,mode,target,folder/'ordinary'/mode,plan)
    ordinary_seconds=time.perf_counter()-ordinary_started
    prediction_started=time.perf_counter();scores.update(inference(context,heads))
    pooled=np.empty_like(scores['SOFT_KD'])
    for track in np.unique(context.video.obs[:,1]):
        at=np.flatnonzero(context.video.obs[:,1]==track)
        pooled[:,at]=probability_scores(expit(scores['SOFT_KD'][:,at].astype(float)).mean(1))[:,None]
    scores['SOFT_POOLED']=pooled
    for head,name in [('SOFT_KD','SOFT_SHARED_MEMORY'),('HARD_KD','HARD_SHARED_MEMORY'),('FUSED_KD','FUSED_SHARED_MEMORY')]:
        scores[name]=probability_scores(memory(scores[head]))
    assert set(scores)==set(METHODS)
    masks={name:value>0 for name,value in scores.items()};md=subset_metadata(case['target'],case['evaluated_video']);tasks=[]
    def seal(name,mask,extra):
        destination=folder/'predictions'/name;destination.mkdir(exist_ok=True,parents=True)
        np.save(destination/'mask.npy',mask)
        seal_mask(destination,md,mask.flatten(),target=case['target'],model=PAIRS[case['pair']][0],method=name,
            GT_open_attempts=len(guard),extra=dict(seed=0,registration_sha256=plan['registration_sha256'],fixed_teacher_replay=True,**extra))
        tasks.append(dict(pair=case['pair'],video=case['evaluated_video'],method=name,folder=str(destination)))
    for name in METHODS:
        assert np.isfinite(scores[name]).all();destination=folder/'predictions'/name;destination.mkdir(exist_ok=True,parents=True)
        np.save(destination/'scores.npy',scores[name]);seal(name,masks[name],dict(same_information=True))
    for rank,count in [(PRIMARY,c) for c in COMPARATORS]+[(c,PRIMARY) for c in COMPARATORS]:
        name=rank+'__AT_'+count+'_COUNT';mask=projection(scores[rank],masks[count],context.video.obs)
        seal(name,mask,dict(rank_method=rank,count_method=count,counts_from_GT=False,counts_equal_per_query_frame=True))
    assert len(tasks)==22 and not guard
    receipt=dict(status='SEALED_ALL22_CONDITIONAL_STATE_VIDEO_OUTPUTS',pair=case['pair'],video=case['evaluated_video'],
        tasks=tasks,registration_sha256=plan['registration_sha256'],GT_open_attempts=0,neural_seed=0,
        teacher_events=len(events),logical_budget=sum(context.quotas.values()),new_teacher_calls=0,
        primary_fit=learned,ordinary_fits=fits,shared_primary_observer_for_KD_controls=True,
        ordinary_phase_wall_seconds=ordinary_seconds,prediction_seal_phase_wall_seconds=time.perf_counter()-prediction_started,
        current_attempt_wall_seconds=time.perf_counter()-started,cost_has_prepaid_inputs_not_fresh_deployment=True)
    atomic_json(finished,receipt)
    print(json.dumps(dict(status=receipt['status'],pair=case['pair'],video=case['evaluated_video'],GPU=gpu,outputs=22)),flush=True)
    return receipt


def worker(gpu):
    from flex_uda.active.deterministic_observation_runtime import configure_before_cuda
    execution=configure_before_cuda()
    import torch
    from flex_uda.a100.mechanism import generation_guard
    guard=generation_guard();torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.20,0)
    plan=read_plan(inputs=True);completed=[]
    for i,case in enumerate(plan['cases']):
        if i%4!=gpu:continue
        done=run_unit(case,gpu,plan,guard);completed.append(dict(pair=case['pair'],video=case['evaluated_video'],outputs=len(done['tasks'])))
        torch.cuda.empty_cache()
    assert not guard
    atomic_json(OUT/'supervisor'/f'GPU{gpu}_completion.json',dict(status='COMPLETE_CONDITIONAL_STATE_GPU_WORKER',GPU=gpu,
        execution=execution,completed=completed,GT_open_attempts=guard,registration_sha256=plan['registration_sha256']))


def evaluate_and_collect(plan,tasks):
    from flex_uda.a100.evaluation_reuse import evaluate
    from scripts.a100_conditional_state_statistics import collect,reproduce
    assert len(tasks)==3256
    def one(task):return evaluate(task['folder'],task['pair'])
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for _ in pool.map(one,tasks):pass
    collect();reproduce()


def main():
    plan=read_plan(inputs=True);sup=OUT/'supervisor';sup.mkdir(parents=True,exist_ok=True)
    lock=(sup/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    barrier=OUT/'all3256_sealed_before_GT.json'
    if not barrier.exists():
        processes=[]
        def drain(process,path):
            with path.open('a',buffering=1) as stream:
                for line in process.stdout:
                    stream.write(line);print(line,end='',flush=True)
            return process.wait()
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures={}
            for gpu in (0,1,2,3):
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1')
                process=subprocess.Popen([sys.executable,'-u',__file__,'--worker',str(gpu)],cwd=ROOT,env=env,
                    stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
                processes.append(process);futures[pool.submit(drain,process,sup/f'GPU{gpu}_{process.pid}.log')]=gpu
                atomic_json(sup/f'GPU{gpu}_launch.json',dict(GPU=gpu,pid=process.pid,registration_sha256=plan['registration_sha256'],time_unix=time.time()))
                print(json.dumps(dict(status='STARTED_FULL_CONDITIONAL_STATE_PRIMARY',GPU=gpu,pid=process.pid)),flush=True)
            failed=[]
            for future in concurrent.futures.as_completed(futures):
                code=future.result();gpu=futures[future]
                print(json.dumps(dict(status='CONDITIONAL_STATE_WORKER_EXIT',GPU=gpu,exit_code=code)),flush=True)
                if code:failed.append(dict(GPU=gpu,exit_code=code))
        if failed:
            atomic_json(sup/f'failure_{int(time.time())}.json',dict(failed=failed,preserve_completed_units=True))
            raise RuntimeError(failed)
        from flex_uda.presence.predictions import verify_predictions
        tasks=[]
        for case in plan['cases']:
            path=OUT/'units'/case['pair']/case['evaluated_video']/'generation_complete.json'
            done=json.loads(path.read_text());assert done['registration_sha256']==plan['registration_sha256'] and done['GT_open_attempts']==0
            for task in done['tasks']:verify_predictions(task['folder'])
            tasks.extend(done['tasks'])
        assert len(tasks)==3256
        atomic_json(barrier,dict(status='ALL3256_CURRENT_STUDY_OUTPUTS_SEALED_BEFORE_GT',tasks=tasks,GT_open_attempts=0,
            registration_sha256=plan['registration_sha256'],parent_teacher_all_attempts_seal_sha256=file_sha(BASE/'all4588_sealed_before_GT.json')))
    sealed=json.loads(barrier.read_text());assert sealed['GT_open_attempts']==0 and sealed['registration_sha256']==plan['registration_sha256']
    read_plan(inputs=True)
    print(json.dumps(dict(status='ALL3256_SEALED_START_ISOLATED_OFFICIAL_CPU_EVALUATION',outputs=3256)),flush=True)
    evaluate_and_collect(plan,sealed['tasks'])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--worker',type=int)
    args=parser.parse_args()
    if args.prepare:prepare()
    elif args.worker is not None:worker(args.worker)
    else:main()
