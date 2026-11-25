"""Complete new residual-learning primary, seven adequate/fair controls.

Four GPU roles0/1/2/3; no old method or module-ablation queue. All66 raw and
both count projections sealed before isolated official evaluation.
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
from scripts.a100_six_direction import OUT as SIX
from scripts.a100_residual_rsi_inputs import OUT as INPUTS,PLAN as INPUT_PLAN

OUT=SIX/'residual_rsi_primary_20261125'
PLAN=OUT/'registration.json'
PUBLIC=ROOT/'docs/a100_six_direction_20261122'
STEM='residual_rsi_primary_20261125'
PRIMARY='RESIDUAL_RSI'
COMPARATORS=('SOFT_KD','HARD_KD','IPW_KD','IMPUTED_KD','R_DIRECT','SOFT_POOLED','UNIFORM_SOFT_KD')
METHODS=(PRIMARY,*COMPARATORS)
ADDED_SOURCES=('scripts/a100_residual_rsi_primary.py','flex_uda/active/residual_rsi_learning.py',
 'flex_uda/active/residual_risk.py','scripts/a100_residual_risk_contract.py',
 'scripts/a100_factor_rsi_primary_quality.py','scripts/a100_primary_statistics_common.py',
 'scripts/a100_semantic_pattern_teacher_statistics.py','scripts/a100_joint_teacher_noise_pilot.py',
 'flex_uda/a100/evaluation_reuse.py','flex_uda/presence/predictions.py',
 'flex_uda/active/deterministic_observation_runtime.py')


def prepare():
    import numpy as np
    from flex_uda.a100.mechanism import generation_guard
    guard=generation_guard();assert not PLAN.exists()
    inputs=json.loads((INPUTS/'completion.json').read_text());parent=json.loads(INPUT_PLAN.read_text())
    contract=ROOT/'a100_runs/visual_classical_20261121/supervisor/residual_risk_contract_20261125.json'
    assert json.loads(contract.read_text())['expected_gradient_maximum_error']<1e-12
    cases=[]
    for record in inputs['cases']:
        pair=record['pair'];folder=INPUTS/pair
        for n,sha in record['output_sha256'].items():assert file_sha(folder/n)==sha
        aux=SIX/'fullsix_learning_common_20261125'/pair/'AUX/teacher_inputs.npz'
        q=np.load(folder/'query.npy');doc=json.loads((folder/'actions.json').read_text())
        assert all((q==int(k)).sum()>=n for k,n in doc['quotas'].items())
        cases.append(dict(pair=pair,target=record['target'],evaluated_video=record['video'],
            pool_folder=str(folder),pool_actions=record['pool_actions'],logical_budget=record['logical_budget'],
            queries=record['queries'],pool_output_sha256=record['output_sha256'],
            auxiliary_path=str(aux),auxiliary_sha256=file_sha(aux)))
    sources=dict(parent['source_sha256']);sources.update({n:file_sha(ROOT/n) for n in ADDED_SOURCES})
    plan=dict(status='PREREGISTERED_PREDICTABLE_RESIDUAL_RSI_COMPLETE_PRIMARY',primary=PRIMARY,
        comparators=COMPARATORS,cases=cases,source_sha256=sources,
        input_plan_sha256=file_sha(INPUT_PLAN),input_completion_sha256=file_sha(INPUTS/'completion.json'),
        mathematical_contract_sha256=file_sha(contract),
        hypothesis='A learned teacher-response forecaster plus actual residual correction and solver-aware acquisition can improve cross-domain RMOT beyond sufficient same-information KD without treating forecaster agreement as truth.',
        human_visual_inspiration='Sweep every query first, then focus on visual hypotheses whose external verification is expected to correct the current understanding.',
        classical_ideas='Predictable augmented importance sampling/control variates and residual-gradient sampling. These are prior ideas, not claimed new estimators or semantic guarantees.',
        loop='ForecasterR predicts original frozen-teacher responses and pre-answer squared residual; solverS changes gradient geometry used byR-guided acquisition. Both neural weights update; external Qwen is frozen, not falsely called a jointly trained Qwen.',
        teacher_invariance='Original answers never rewritten. R gives a control variate; signed residual correction uses requested originalteacher response and its recorded acquisition probability.',
        sampling='WITH replacement within every query; fixed original per-query budgets and four predeclared front-loaded stages. Positive.2uniform mixture on every action, no hard fresh-track/probe exclusion; repeated logical queries are explicit cached replay, not extra unique evidence.',
        finite_pool='104/4068/95 actions in three declared historical policy-union libraries; full support only inside these pools. No claim about outside-pool risk or full dense-target support.',
        risk='Macro-query mean of action losses. Every draw uses R forecast fixed BEFORE answer; proxy-sum plus(y-m)/recordedPi yields signed coefficients. They are not probabilities and are never clipped to[0,1].',
        bounded_model='9352->128SiLU->128SiLU->1,logit8tanh(raw/8), shared unlabelled-pool normalization for ALL heads. Forecaster same hidden capacity with mean/residual2heads.',
        training='seed0/Adam.001/clipgrad10. Everysolver softaux500, then500/500/500/4000 phases; hard baseline hardens both main and auxiliary labels. Ordinary properKL<=.005,min6000,max16000; R softaux500+four500steps. Same optimizer continuity/cyclic batches; no schedule confound hidden as method gain.',
        auxiliary='Same100same-target-collection3view events for all opponents; separately stored, explicit transduction. No clean target validation labels.',
        controls={'SOFT_KD':'All actual primary draws plus sameaux, ordinary soft BCE with query-balanced weights.',
          'HARD_KD':'Threshold all main/auxiliary teacher probabilities at.5; otherwise same learner.',
          'IPW_KD':'Same draws with correct within-query with-replacement importance weights; enough optimization.',
          'IMPUTED_KD':'Same prequential R mean snapshots, same phase path/capacity and full action-pool features, sufficient BCE fit; no residual correction.',
          'R_DIRECT':'Actual final learnedR mean head itself; never promote it to primary posthoc.',
          'SOFT_POOLED':'Arithmetic probability mean of adequateSOFT over providedTrackID, strong ordinary comparison even for time-varying tasks.',
          'UNIFORM_SOFT_KD':'Same perquery budget but distinct uniform actions; can have MORE unique teacher facts than the primary. Separate adequateGPU2head.'},
        failure_constraints=['Do not train on new scene-context answers: paired72 did not improveKITTI; oldphysicsF/H residuals are not motion labels.',
          'No free latent relabeling, no half-main-label split, no GT-dependent per-setting gates or checkpoints.',
          'Do not apply historical no-replacement LURE with the wrong probability or invent old propensities.',
          'Expected fixed-model teacher-risk identity does not imply fitted-model generalization, true-semantic risk or HOTA improvement.'],
        primary_prior_work=[dict(url='https://arxiv.org/html/2607.08347',scope='Fixed-model active testing with LURE/control variates; our declared sampling is a simpler with-replacement learning candidate, no borrowing its CI theorem.'),
          dict(url='https://proceedings.neurips.cc/paper_files/paper/2021/file/09e7655fc1dc8fa7c9d6c4478313d5e6-Paper.pdf',scope='Predictable adaptive nuisance estimation; no assumed generic adaptive-training guarantee.')],
        neural_seeds=[0],GPU_roles={'C1':0,'C2':1,'C4':3,'UNIFORM':2},GPU2_fraction=.10,
        all_existing_GPU2_services_preserved=True,new_Qwen_calls=0,source_features_answers_prepaid=True,
        not_fresh_cold_cost_or_online_teacher_deployment=True,all_data_DEV=True,
        new_main_raw_records=24,new_dual_count_records=42,official_statistical_records=66,paired_effects=63,
        CountSelf_exact_aliases=24,all66sealed_before_current_study_GT=True,no_module_ablation_matrix=True,
        success_screen='Same strict exploratory screen: raw and BOTH CountMatched effects positive versus ALL7 controls in EVERY one-video case. No posthoc highest-control promotion.',
        no_fullsix_threecontributions_or_RSI_success_claim=True,goal_status='ACTIVE')
    assert not guard
    atomic_json(PLAN,plan);atomic_json(PUBLIC/(STEM+'_registration.json'),plan)
    print(json.dumps(dict(status=plan['status'],cases=3,pins=len(sources),records=66)),flush=True)


def registration():
    plan=json.loads(PLAN.read_text())
    for n,sha in plan['source_sha256'].items():assert file_sha(ROOT/n)==sha,n
    assert file_sha(INPUT_PLAN)==plan['input_plan_sha256']
    for case in plan['cases']:
        for n,sha in case['pool_output_sha256'].items():assert file_sha(Path(case['pool_folder'])/n)==sha
        assert file_sha(case['auxiliary_path'])==case['auxiliary_sha256']
    return plan


def uniform_worker():
    from flex_uda.active.deterministic_observation_runtime import configure_before_cuda
    configure_before_cuda()
    import torch
    from flex_uda.a100.mechanism import generation_guard
    from flex_uda.active.residual_rsi_learning import uniform_draws,fit_ordinary
    guard=generation_guard();plan=registration();torch.set_num_threads(2);torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(.10)
    for case in plan['cases']:
        folder=OUT/case['pair'];(folder/'ordinary').mkdir(parents=True,exist_ok=True)
        path=folder/'uniform_draws.json'
        if path.exists():draws=json.loads(path.read_text())['draws']
        else:
            draws,receipt=uniform_draws(case,'cuda:0');atomic_json(path,dict(draws=draws,broker=receipt))
        model,receipt=fit_ordinary(case,draws,'UNIFORM_SOFT_KD',folder/'ordinary','cuda:0');del model
        print(json.dumps(dict(status='ADEQUATE_UNIFORM_DISTILLATION_COMPLETE',pair=case['pair'],fit=receipt)),flush=True)
    assert not guard
    atomic_json(OUT/'GPU2_complete.json',dict(status='ALL3_UNIFORM_CONTROLS_COMPLETE',GT_reads=0))


def worker(pair):
    from flex_uda.active.deterministic_observation_runtime import configure_before_cuda
    execution=configure_before_cuda()
    import numpy as np
    import torch
    from scipy.special import logsumexp
    from flex_uda.a100.mechanism import generation_guard,PAIRS
    from flex_uda.active.joint_observation_inputs import JointObservationVideo,use_verified_local_features
    from flex_uda.active.factor_rsi_learning import Inputs
    from flex_uda.active.full_appearance_learning_inputs import augment_dense
    from flex_uda.active.residual_rsi_learning import train_primary,fit_ordinary,fit_imputed,MeanHead,tensors
    from flex_uda.presence.predictions import seal_mask
    from scripts.a100_factor_rsi_primary_quality import subset_metadata,projection
    plan=registration();guard=generation_guard();case=next(c for c in plan['cases'] if c['pair']==pair)
    folder=OUT/pair;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'main_completion.json').exists():return
    torch.set_num_threads(2);torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.35)
    use_verified_local_features();started=time.perf_counter()
    model,forecaster,learned=train_primary(case,folder/'primary','cuda:0',file_sha(PLAN))
    (folder/'ordinary').mkdir(exist_ok=True);heads={PRIMARY:model,'R_DIRECT':MeanHead(forecaster)};fits={}
    for name in ('SOFT_KD','HARD_KD','IPW_KD'):
        heads[name],fits[name]=fit_ordinary(case,learned['draws'],name,folder/'ordinary','cuda:0')
    heads['IMPUTED_KD'],fits['IMPUTED_KD']=fit_imputed(case,folder/'primary',folder/'ordinary','cuda:0')
    uniform=folder/'ordinary/UNIFORM_SOFT_KD.pt'
    while not uniform.exists():
        if (OUT/'GPU2_failed.json').exists():raise RuntimeError((OUT/'GPU2_failed.json').read_text())
        time.sleep(2)
    ud=json.loads((folder/'uniform_draws.json').read_text())['draws']
    heads['UNIFORM_SOFT_KD'],fits['UNIFORM_SOFT_KD']=fit_ordinary(case,ud,'UNIFORM_SOFT_KD',folder/'ordinary','cuda:0')
    video=JointObservationVideo(pair,case['evaluated_video']);inputs=Inputs(video);n,nq=len(video.obs),len(video.queries)
    scores={name:np.empty((nq,n),dtype='float32') for name in heads}
    with torch.no_grad():
        for qi in range(nq):
            for begin in range(0,n,512):
                rows=np.arange(begin,min(begin+512,n));records=[dict(local_query=qi,local_track=int(video.columns[r]),frame_id=int(video.obs[r,0]),track_id=int(video.obs[r,1])) for r in rows]
                a=inputs.arrays(records);base=np.column_stack((a['bags'].reshape(len(rows),-1),a['present'],a['current'],a['coarse'])).astype('float32')
                x=tensors(augment_dense(base,video,rows),'cuda:0')
                for name,head in heads.items():scores[name][qi,rows]=head(x).cpu().numpy()
    z=scores['SOFT_KD'].astype('float64');pooled=np.empty_like(z)
    for ti in range(len(video.tracks)):
        rows=np.flatnonzero(video.columns==ti)
        pooled[:,rows]=(logsumexp(-np.logaddexp(0,-z[:,rows]),axis=1)-logsumexp(-np.logaddexp(0,z[:,rows]),axis=1))[:,None]
    scores['SOFT_POOLED']=pooled.astype('float32');bits={name:values>0 for name,values in scores.items()}
    metadata=subset_metadata(case['target'],video.video);tasks=[]
    for name in METHODS:
        path=folder/'predictions'/name;path.mkdir(parents=True,exist_ok=True)
        assert np.isfinite(scores[name]).all() and np.array_equal(projection(scores[name],bits[name],video.obs),bits[name])
        np.save(path/'scores.npy',scores[name]);np.save(path/'mask.npy',bits[name])
        seal_mask(path,metadata,bits[name].flatten(),target=case['target'],model=PAIRS[pair][0],method=name,
            GT_open_attempts=len(guard),extra=dict(seed=0,same9352D_inputs=True,teacher_answers_unchanged=True,prepaid_replay_not_fresh_cost=True))
        tasks.append(dict(pair=pair,method=name,folder=str(path)))
    for rank,count in [(PRIMARY,n) for n in COMPARATORS]+[(n,PRIMARY) for n in COMPARATORS]:
        name=rank+'__AT_'+count+'_COUNT';path=folder/'predictions'/name;selected=projection(scores[rank],bits[count],video.obs)
        seal_mask(path,metadata,selected.flatten(),target=case['target'],model=PAIRS[pair][0],method=name,
            GT_open_attempts=len(guard),extra=dict(rank_method=rank,count_method=count,counts_equal_per_query_frame=True,counts_from_GT=False))
        tasks.append(dict(pair=pair,method=name,folder=str(path)))
    assert len(tasks)==22 and not guard
    atomic_json(folder/'main_completion.json',dict(status='SEALED_ALL22_RESIDUAL_PRIMARY_OUTPUTS_NO_GT',
        tasks=tasks,ordinary_fits=fits,parameter_changes=learned['parameter_changes'],rounds=learned['rounds'],
        logical_draws=learned['logical_draws'],unique_teacher_actions=learned['broker']['unique_teacher_bundles'],
        execution=execution,GT_open_attempts=0,worker_wall_seconds=time.perf_counter()-started))


def drain(process,path):
    with path.open('a',buffering=1) as stream:
        for line in process.stdout:stream.write(line);print(line,end='',flush=True)


def main():
    plan=registration();supervisor=OUT/'supervisor';supervisor.mkdir(parents=True,exist_ok=True)
    pending=[(c['pair'],gpu,['--worker',c['pair']]) for c,gpu in zip(plan['cases'],(0,1,3)) if not (OUT/c['pair']/'main_completion.json').exists()]
    if not (OUT/'GPU2_complete.json').exists():pending.append(('UNIFORM',2,['--uniform-worker']))
    for label,gpu,_ in pending:
        owners=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-compute-apps=pid,process_name','--format=csv,noheader,nounits'],text=True).strip()
        used=int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
        total=int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.total','--format=csv,noheader,nounits'],text=True).strip())
        if gpu==2:assert total-used>=8192,(used,total)
        else:assert not owners and used<512,(gpu,used,owners)
        atomic_json(supervisor/(label+'_preflight.json'),dict(GPU=gpu,residents=owners,used_MiB=used,all_preserved=True))
    jobs=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as readers:
        for label,gpu,args in pending:
            proc=subprocess.Popen([sys.executable,'-u',str(Path(__file__)),*args],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'))
            future=readers.submit(drain,proc,supervisor/(label+'.log'));jobs.append((label,proc,future))
            atomic_json(supervisor/(label+'_launch.json'),dict(pid=proc.pid,GPU=gpu));print(json.dumps(dict(status='STARTED_NEW_RESIDUAL_PRIMARY_ROLE',role=label,GPU=gpu,pid=proc.pid)),flush=True)
        failed=[]
        for label,proc,future in jobs:
            code=proc.wait();future.result()
            if code:failed.append(dict(role=label,exit_code=code))
    if failed:raise RuntimeError(failed)
    from flex_uda.presence.predictions import verify_predictions
    tasks=[]
    for case in plan['cases']:
        done=json.loads((OUT/case['pair']/'main_completion.json').read_text());assert done['GT_open_attempts']==0;tasks.extend(done['tasks'])
    assert len(tasks)==66
    for task in tasks:verify_predictions(task['folder'])
    atomic_json(OUT/'all66_sealed_before_GT.json',dict(tasks=tasks,GT_open_attempts=0,registration_sha256=file_sha(PLAN)))
    print('ALL66_NEW_RESIDUAL_PRIMARY_OUTPUTS_SEALED_ISOLATED_EVALUATION',flush=True)
    from scripts.a100_joint_teacher_noise_pilot import evaluate_one
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:completed=list(pool.map(evaluate_one,tasks))
    results={c['pair']:{} for c in plan['cases']}
    for task,result in completed:results[task['pair']][task['method']]=result
    comparisons=[]
    for pair,values in results.items():
        primary=values[PRIMARY]['HOTA_unrounded']
        for other in COMPARATORS:
            reference=values[other]['HOTA_unrounded']
            comparisons.append(dict(pair=pair,comparator=other,primary_HOTA=primary,ordinary_HOTA=reference,raw_delta_HOTA=primary-reference,
                primary_at_ordinary_count_delta_HOTA=values[PRIMARY+'__AT_'+other+'_COUNT']['HOTA_unrounded']-reference,
                ordinary_at_primary_count_delta_HOTA=primary-values[other+'__AT_'+PRIMARY+'_COUNT']['HOTA_unrounded']))
    eligible=all(min(r['raw_delta_HOTA'],r['primary_at_ordinary_count_delta_HOTA'],r['ordinary_at_primary_count_delta_HOTA'])>0 for r in comparisons)
    atomic_json(OUT/'quality_results.json',dict(status='COMPLETE_NEW_RESIDUAL_RSI_PRIMARY',results=results,comparisons=comparisons,
        candidate_eligible_for_larger_screen=eligible,all_data_DEV=True,no_fullsix_threecontribution_or_fresh_cost_success=True,goal_status='ACTIVE'))
    from scripts.a100_primary_statistics_common import collect,reproduce
    collect(OUT,PUBLIC,STEM,66);reproduce(PUBLIC,STEM,66,63)
    print(json.dumps(dict(status='COMPLETE_NEW_RESIDUAL_PRIMARY_ALL66_REPRODUCED',comparisons=comparisons)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--worker');p.add_argument('--uniform-worker',action='store_true');args=p.parse_args()
    try:
        if args.prepare:prepare()
        elif args.worker:worker(args.worker)
        elif args.uniform_worker:uniform_worker()
        else:main()
    except Exception as error:
        if args.uniform_worker:atomic_json(OUT/'GPU2_failed.json',dict(error=repr(error)))
        if args.worker:atomic_json(OUT/args.worker/'failed.json',dict(error=repr(error)))
        raise
