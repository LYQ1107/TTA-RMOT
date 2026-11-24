"""Unified object/state, conditional-learning and learned-observer study.

Six settings, seed0, native binary teacher replies and own-video prefixes.
All raw and dual-count outputs seal before any manual-GT evaluation.
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
import numpy as np
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from flex_uda.a100.mechanism import PAIRS,metadata,generation_guard
from scripts.a100_six_direction import OUT as SIX,PUBLIC

OUT=SIX/'native_binary_quality_20261124'
PLAN=PUBLIC/'native_binary_quality_registration_20261124.json'
STATE=OUT/'supervisor/state.json'
PRIMARY='FULL_CONDITIONAL_REFERENCE_LEARNED_FINAL'


def recipes():
    result={}
    ordinary={
        'JOINT_KD':dict(family='pair',risk='joint'),
        'MARGINAL_KD':dict(family='pair',risk='marginal'),
        'POINT_KD':dict(family='point',risk='joint'),
        'LOGIT_KD':dict(family='point',risk='logit'),
        'COARSE_UNIT_SOFT':dict(family='coarse',hard=False,confidence=False),
        'COARSE_UNIT_HARD':dict(family='coarse',hard=True,confidence=False),
        'COARSE_CONF_SOFT':dict(family='coarse',hard=False,confidence=True),
        'COARSE_CONF_HARD':dict(family='coarse',hard=True,confidence=True)}
    for policy in ('LEARNED','FROZEN'):
        for conditional in (True,False):
            for reference in (True,False):
                name='FULL_'+('CONDITIONAL' if conditional else 'LIKELIHOOD')+'_'+('REFERENCE' if reference else 'LOCAL')+'_'+policy
                result[name]=dict(family='structured',conditional=conditional,reference=reference,identity=True,policy=policy)
        for name,recipe in ordinary.items():result[name+'_'+policy]=dict(recipe,policy=policy)
    result['NO_IDENTITY_CONDITIONAL_REFERENCE_LEARNED']=dict(
        family='structured',conditional=True,reference=True,identity=False,policy='LEARNED')
    return result


RECIPES=recipes()
METHODS=tuple(name+'_'+snapshot for name,recipe in RECIPES.items()
    for snapshot in (('FINAL',) if recipe['family']=='structured' else ('FINAL','MEAN')))
COMPARATORS=tuple(n for n in METHODS if n!=PRIMARY)


def registration():
    plan=json.loads(PLAN.read_text())
    assert plan['neural_seeds']==[0] and plan['primary']==PRIMARY
    assert plan['recipes']==RECIPES and tuple(plan['methods'])==METHODS
    for name,sha in plan['source_sha256'].items():assert file_sha(ROOT/name)==sha,name
    return plan


def scheduled(recipe):
    return recipe['family']!='coarse' or not(recipe.get('hard') or recipe.get('confidence'))


def dense_scores(video,visual,models,mean,std,local,device):
    import torch
    from flex_uda.active.native_binary_models import log_probabilities,pair_marginal_logs
    from flex_uda.active.native_binary_curriculum import normalize_pair
    n=len(video.obs);nq=len(video.queries);successor=np.empty(n,dtype=int);predecessor=np.empty(n,dtype=int)
    for frame in np.unique(video.obs[:,0]):
        ids=np.flatnonzero(video.obs[:,0]==frame);ids=ids[np.argsort(video.obs[ids,1],kind='stable')]
        successor[ids]=np.roll(ids,-1);predecessor[ids]=np.roll(ids,1)
    scores={name:np.lib.format.open_memmap(local/(name+'_scores.npy'),mode='w+',dtype='float64',shape=(nq,n)) for name in METHODS}
    with torch.no_grad():
        for qi in range(nq):
            logrows={name:np.zeros((n,2,2),dtype='float64') for name in METHODS}
            for begin in range(0,n,1024):
                ids=np.arange(begin,min(begin+1024,n));partners=successor[ids]
                ta,tb=video.columns[ids],video.columns[partners]
                raw=dict(anchors_A=video.raw_bags[qi,ta],present_A=video.present[ta],
                    anchors_B=video.raw_bags[qi,tb],present_B=video.present[tb],
                    first=video.dense_rows(np.full(len(ids),qi),ids),
                    second=video.dense_rows(np.full(len(ids),qi),partners))
                normalized=normalize_pair(raw,mean,std)
                data={k:torch.as_tensor(v,device=device) for k,v in normalized.items()}
                data.update(point_anchors=data['anchors_A'],point_present=data['present_A'],point_current=data['first'],
                    coarse_inputs=torch.as_tensor(((visual.coarse[qi,ta]-mean)/std).astype('float32'),device=device))
                for name,(final,early) in models.items():
                    recipe=RECIPES[name];final_log=log_probabilities(final,data,recipe)
                    ispair=recipe['family'] in ('structured','pair')
                    final_binary=pair_marginal_logs(final_log) if ispair else final_log[:,None,:].expand(-1,2,-1)
                    logrows[name+'_FINAL'][ids]=final_binary.cpu().numpy()
                    if recipe['family']!='structured':
                        early_log=log_probabilities(early,data,recipe)
                        early_binary=pair_marginal_logs(early_log) if ispair else early_log[:,None,:].expand(-1,2,-1)
                        logrows[name+'_MEAN'][ids]=(torch.logaddexp(final_binary,early_binary)-np.log(2.)).cpu().numpy()
            for name,value in logrows.items():
                recipe=RECIPES[name.rsplit('_',1)[0]]
                # Pair distributions contribute outgoing A and incoming B.
                # Point/coarse models are already each candidate's own opinion.
                binary=np.logaddexp(value[:,0,:],value[predecessor,1,:])-np.log(2.) if recipe['family'] in ('structured','pair') else value[:,0,:]
                scores[name][qi]=binary[:,0]-binary[:,1]
    for score in scores.values():score.flush()
    return scores


def worker(pair):
    plan=registration()
    from flex_uda.active.deterministic_observation_runtime import configure_before_cuda
    execution=configure_before_cuda()
    import torch
    from flex_uda.active.joint_observation_inputs import JointObservationVideo,use_verified_local_features
    from flex_uda.active.native_binary_inputs import inventory,NativeVisual
    from flex_uda.active.native_binary_models import make_model
    from flex_uda.active.native_binary_curriculum import observe_and_solve,replay_solver
    from scripts.a100_visual_classical_strict_prefix import immediate_predictions
    from flex_uda.presence.predictions import seal_mask
    guard=generation_guard();torch.set_num_threads(2);torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.15)
    use_verified_local_features();source,target=PAIRS[pair];md=metadata(target)
    root=OUT/pair/'seed0';root.mkdir(parents=True,exist_ok=True)
    if(root/'completion.json').is_file():return
    stream=root/'stream.pt';previous={};actors={};receipts=[];next_video=0
    if stream.is_file():
        saved=torch.load(stream,map_location='cpu',weights_only=False)
        previous,actors,receipts,next_video=[saved[k] for k in ('previous','actors','receipts','next_video')]
    total=np.zeros(1101);square=np.zeros(1101);number=0;past={p:[] for p in ('LEARNED','FROZEN')}
    started=time.perf_counter()
    for vi,video_id in enumerate(sorted(md['videos'])):
        stamp=time.perf_counter();v=JointObservationVideo(pair,video_id)
        actions,quotas=inventory(pair,video_id);visual=NativeVisual(v,actions)
        total+=visual.norm_sum;square+=visual.norm_square;number+=visual.norm_rows
        mean=total/number;std=np.sqrt(np.maximum(square/number-mean**2,0)).clip(min=.01)
        local=root/'videos'/video_id;local.mkdir(parents=True,exist_ok=True)
        if vi<next_video:
            for p in past:past[p].append(visual.record(json.loads((local/'policies'/p/'paid.json').read_text())['paid']))
            continue
        models={};fits={};paths={};new_previous={};new_actors={};current={}
        def progress(stage,**extra):
            atomic_json(root/'progress.json',dict(stage=stage,video=video_id,completed_videos=vi,
                total_videos=len(md['videos']),completed_arms=len(models),**extra))
        for policy in past:
            name='FULL_CONDITIONAL_REFERENCE_'+policy;folder=local/'policies'/policy;folder.mkdir(parents=True,exist_ok=True)
            completed=folder/'closed.pt'
            if completed.is_file():
                saved=torch.load(completed,map_location='cpu',weights_only=False)
                checkpoint,actor=saved['checkpoint'],saved['actor']
                assert checkpoint['recipe']==RECIPES[name] and np.array_equal(checkpoint['mean'],mean) and np.array_equal(checkpoint['std'],std)
                paid=json.loads((folder/'paid.json').read_text())['paid'];sequence=json.loads((folder/'sequence.json').read_text())['sequence']
                report=json.loads((folder/'receipt.json').read_text());record=visual.record(paid)
                model=make_model(RECIPES[name],1101,checkpoint['width']).cuda().eval();model.load_state_dict(checkpoint['model'])
            else:
                round_path=folder/'round_latest.pt'
                resume=torch.load(round_path,map_location='cpu',weights_only=False) if round_path.is_file() else None
                def callback(stage,state):
                    temporary=round_path.with_suffix('.tmp');torch.save(state,temporary);temporary.replace(round_path)
                    progress('CLOSED_OBSERVER_SOLVER_LOOP_NO_GT',policy=policy,round=stage,
                        actor_reward=state['rounds'][-1]['credit']['reward'])
                    print(json.dumps(dict(pair=pair,video=video_id,policy=policy,round=stage,
                        reward=state['rounds'][-1]['credit']['reward'])),flush=True)
                result=observe_and_solve(visual,actions,quotas,mean,std,past=past[policy],previous=previous.get(name),
                    actor_state=actors.get(policy),learned=policy=='LEARNED',device='cuda:0',callback=callback,resume=resume)
                model,checkpoint,actor,paid,sequence,record,report=[result[k] for k in ('model','checkpoint','actor','paid','sequence','record','report')]
                atomic_json(folder/'paid.json',dict(paid=paid,GT_open_attempts=0))
                atomic_json(folder/'sequence.json',dict(sequence=sequence))
                atomic_json(folder/'receipt.json',report)
                temporary=completed.with_suffix('.tmp');torch.save(dict(checkpoint=checkpoint,actor=actor),temporary);temporary.replace(completed)
            assert report['teacher_budget']['unique_teacher_bundles']==sum(quotas.values())
            models[name]=(model,model);fits[name]=report;new_previous[name]=checkpoint;new_actors[policy]=actor
            paths[policy]=(paid,sequence);current[policy]=record
        for name,recipe in RECIPES.items():
            if name in models:continue
            path=local/'heads'/(name+'.pt');path.parent.mkdir(parents=True,exist_ok=True)
            if path.is_file():
                saved=torch.load(path,map_location='cpu',weights_only=False);checkpoint=saved['checkpoint'];report=saved['report']
                assert checkpoint['recipe']==recipe and np.array_equal(checkpoint['mean'],mean) and np.array_equal(checkpoint['std'],std)
                model=make_model(recipe,1101,checkpoint['width']).cuda().eval();model.load_state_dict(checkpoint['model'])
                early=make_model(recipe,1101,checkpoint['width']).cuda().eval();early.load_state_dict(saved['early'])
            else:
                paid,sequence=paths[recipe['policy']]
                model,early,checkpoint,report=replay_solver(visual,paid,sequence,recipe,mean,std,
                    past=past[recipe['policy']],previous=previous.get(name),device='cuda:0',scheduled=scheduled(recipe))
                temporary=path.with_suffix('.tmp');torch.save(dict(checkpoint=checkpoint,
                    early={k:a.detach().cpu().clone() for k,a in early.state_dict().items()},report=report),temporary);temporary.replace(path)
            models[name]=(model,early);fits[name]=report;new_previous[name]=checkpoint
            if recipe['family']!='structured':
                assert all(f['teacher_fit_gate_passed'] and f['excess_risk']<=.005 for f in report['fits'])
            progress('FITTING_MATCHED_REPLAY_AND_ADEQUATE_KD_NO_GT',last_arm=name,
                excess_risk=report['fits'][-1]['excess_risk'])
        prediction_start=time.perf_counter();scores=dense_scores(v,visual,models,mean,std,local,'cuda:0')
        for name,score in scores.items():
            bits=np.asarray(score)>0;folder=root/name/'videos'/video_id;folder.mkdir(parents=True,exist_ok=True)
            np.save(folder/'mask.npy',bits);immediate_predictions(root/name,v,bits)
        torch.cuda.synchronize()
        receipt=dict(video=video_id,fits=fits,policy_budgets={p:fits['FULL_CONDITIONAL_REFERENCE_'+p]['teacher_budget'] for p in past},
            normalization_rows=number,past_and_current_video_only=True,GT_open_attempts=len(guard),
            prediction_wall_seconds=time.perf_counter()-prediction_start,wall_seconds=time.perf_counter()-stamp,
            prepaid_replay_shared_training_not_cold_deployment=True)
        atomic_json(local/'receipt.json',receipt);receipts.append(receipt)
        temporary=stream.with_suffix('.tmp');torch.save(dict(previous=new_previous,actors=new_actors,
            receipts=receipts,next_video=vi+1),temporary);temporary.replace(stream)
        previous,actors=new_previous,new_actors
        for policy in past:past[policy].append(current[policy])
        del models,scores,v,visual;torch.cuda.empty_cache()
    methods=[]
    for name in METHODS:
        mask=np.zeros(md['candidate_query_rows'],dtype=bool)
        for video_id in sorted(md['videos']):
            bits=np.load(root/name/'videos'/video_id/'mask.npy')
            for qi,q in enumerate(q for q in md['episodes'] if q['video_id']==video_id):mask[q['start']:q['stop']]=bits[qi]
        row=seal_mask(root/name,md,mask,target=target,model=source,method=name,GT_open_attempts=len(guard),
            extra=dict(seed=0,native_binary_teacher=True,own_policy_prefix=True,current_video_transductive=True))
        methods.append(dict(method=name,rows=row['prediction_rows']))
    assert not guard
    atomic_json(root/'completion.json',dict(status='SEALED_NATIVE_BINARY_RAW_NO_GT',pair=pair,seed=0,
        methods=methods,receipts=receipts,execution=execution,GT_open_attempts=0,wall_seconds=time.perf_counter()-started))


def counts(pair):
    # Exact existing float64 rank/count code, scoped only inside this process.
    from scripts import a100_joint_observation_six_setting as original
    original.OUT,original.METHODS,original.PRIMARY,original.COMPARATORS=OUT,METHODS,PRIMARY,COMPARATORS
    original.counts(pair)


def main():
    plan=registration();started=time.perf_counter();pending=list(PAIRS);active={};finished=[];failure=None
    def state(status):
        view={}
        for gpu,(proc,handle,job) in active.items():
            p=Path(job['folder'])/'progress.json';view[str(gpu)]=dict(job,progress=json.loads(p.read_text()) if p.is_file() else {})
        row=dict(status=status,pid=os.getpid(),active=view,pending=pending,finished=finished,failure=failure,
            wall_seconds=time.perf_counter()-started,neural_seeds=[0],goal_status='ACTIVE')
        atomic_json(STATE,row);print(json.dumps(row),flush=True)
    STATE.parent.mkdir(parents=True,exist_ok=True)
    while pending or active:
        for gpu,(proc,handle,job) in list(active.items()):
            code=proc.poll()
            if code is None:continue
            handle.close();del active[gpu]
            if code==0 and(Path(job['folder'])/'completion.json').is_file():finished.append(job)
            else:failure=dict(job=job,exit_code=code,error_tail=Path(job['log']).read_text()[-1800:])
        for pair in list(pending):
            if(OUT/pair/'seed0/completion.json').is_file():pending.remove(pair);finished.append(dict(pair=pair,reused_sealed=True))
        candidates=[g for g in (2,3,0,1) if g not in active]
        if pending and candidates and not failure:
            raw=subprocess.check_output(['nvidia-smi','-i',','.join(map(str,candidates)),
                '--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
            memory={int(r.split(',')[0]):int(r.split(',')[1]) for r in raw.strip().splitlines()}
            for gpu in candidates:
                limit=8192 if gpu in (2,3) else 512
                if not pending or memory[gpu]>=limit:continue
                if gpu in (0,1):
                    owners=subprocess.check_output(['nvidia-smi','-i',str(gpu),
                        '--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
                    if owners:continue
                # Start the two Dance settings on the user's available 2/3.
                pair=('C1' if 'C1' in pending else 'C6' if 'C6' in pending else pending[0]);pending.remove(pair)
                log=OUT/'supervisor'/(pair+'_seed0.log');handle=log.open('a',buffering=1)
                proc=subprocess.Popen([sys.executable,'-u',str(Path(__file__)),'--pair',pair],cwd=ROOT,
                    stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,env=dict(os.environ,
                        CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2'))
                job=dict(pair=pair,pid=proc.pid,physical_GPU=gpu,folder=str(OUT/pair/'seed0'),log=str(log),preflight_memory_MiB=memory[gpu])
                active[gpu]=proc,handle,job
        state('PRESERVING_HEALTHY_NATIVE_WORKERS_AFTER_FAILURE' if failure else 'RUNNING_UNIFIED_NATIVE_BINARY_SIX_SETTINGS_NO_GT')
        if failure and not active:raise RuntimeError(json.dumps(failure))
        if pending or active:time.sleep(30)
    state('ALL_NATIVE_RAW_SEALED_GENERATING_DUAL_COUNTS_NO_GT')
    ordinary_fits=[]
    for pair in PAIRS:
        raw=json.loads((OUT/pair/'seed0/completion.json').read_text());assert raw['GT_open_attempts']==0
        for video in raw['receipts']:
            for name,report in video['fits'].items():
                if RECIPES[name]['family']=='structured':continue
                for fit in report['fits']:
                    assert fit['teacher_fit_gate_passed'] and fit['excess_risk']<=.005
                    ordinary_fits.append(dict(pair=pair,video=video['video'],arm=name,
                        excess_risk=fit['excess_risk'],risk_units=fit['risk_units']))
    atomic_json(OUT/'all_ordinary_fit_gate_before_GT.json',dict(
        status='PASS_ALL_NATIVE_ORDINARY_PREFIX_FIT_GATES_BEFORE_GT',fits=ordinary_fits,GT_open_attempts=0))
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:list(pool.map(counts,PAIRS))
    tasks=[]
    for pair in PAIRS:
        root=OUT/pair/'seed0'
        for folder in (root,root/'countmatched'):
            complete=json.loads((folder/'completion.json').read_text());assert complete['GT_open_attempts']==0
            tasks.extend((str(folder/r['method']),pair) for r in complete['methods'])
    expected=6*(len(METHODS)+2*len(COMPARATORS));assert len(tasks)==len(set(tasks))==expected
    atomic_json(OUT/'all_predictions_sealed_before_GT.json',dict(methods=expected,raw_methods=6*len(METHODS),
        CountSelf_aliases=6*len(METHODS),GT_open_attempts=0,primary=PRIMARY,source_sha256=plan['source_sha256']))
    state('ALL_NATIVE_PREDICTIONS_SEALED_ISOLATED_OFFICIAL_EVALUATION')
    from scripts.a100_visual_classical_sparse_observation_evaluate import evaluate_one
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as pool:results=dict(pool.map(evaluate_one,tasks))
    atomic_json(OUT/'official.json',dict(methods=expected,results=results))
    state('ANALYZING_COMPLETE_NATIVE_BINARY_SIX_SETTING_STUDY')
    for script in ('scripts/a100_native_binary_analyze.py','scripts/a100_native_binary_statistics_reproduce.py'):
        subprocess.run([sys.executable,'-u',script],cwd=ROOT,check=True,
            env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2'))
    state('PASS_UNIFIED_NATIVE_BINARY_SIX_SETTING_EXECUTION_AND_REPRODUCTION_GOAL_ACTIVE')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--pair',choices=tuple(PAIRS));args=parser.parse_args()
    if args.pair:worker(args.pair)
    else:main()
