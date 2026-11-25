"""Complete primary comparison of semantic opinion-pattern R <-> neural S.

Reuse the completed eight identical-data opponents exactly. Fit only the new
primary and a sufficiently optimized pattern-fuse-then-distill opponent.
"""
import argparse
import concurrent.futures
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json, file_sha
from scripts import a100_joint_teacher_noise_pilot as parent

OUT = parent.OUT.parent/'semantic_pattern_teacher_primary_20261125'
PUBLIC = parent.PUBLIC
PLAN = OUT/'registration.json'
PRIMARY = 'JOINT_SEMANTIC_PATTERN_RSI'
NEW_ORDINARY = 'PATTERN_FUSED_SOFT_KD'
COMPARATORS = (*parent.COMPARATORS, NEW_ORDINARY)
METHODS = (PRIMARY, *COMPARATORS)
STEM = 'semantic_pattern_teacher_primary_20261125'
SOURCES = (*parent.SOURCES, 'flex_uda/active/semantic_teacher_patterns.py',
           'scripts/a100_semantic_pattern_teacher_primary.py',
           'scripts/a100_semantic_pattern_teacher_contract.py',
           'scripts/a100_semantic_pattern_teacher_statistics.py')


def input_path(pair):
    return parent.OUT/pair/'teacher_inputs.npz'


def prepare():
    from flex_uda.a100.mechanism import generation_guard
    guard = generation_guard()
    previous = parent.registration()
    audit = parent.OUT/'independent_sealed_prediction_audit.json'
    assert audit.is_file()
    assert json.loads((parent.OUT/'all75_sealed_before_GT.json').read_text())['GT_open_attempts'] == 0
    reused = {}
    for case in previous['cases']:
        pair = case['pair']
        for name in parent.COMPARATORS:
            folder = parent.OUT/pair/'predictions'/name
            for leaf in ('scores.npy', 'mask.npy', 'predictions_seal.json'):
                path = folder/leaf
                reused[str(path)] = file_sha(path)
    plan = dict(status='PREREGISTERED_COMPLETE_SEMANTIC_PATTERN_PRIMARY', primary=PRIMARY,
        comparators=COMPARATORS, cases=previous['cases'], neural_seeds=[0],
        source_sha256={name: file_sha(ROOT/name) for name in SOURCES},
        parent_registration_sha256=file_sha(parent.PLAN), reused_prediction_sha256=reused,
        completed_parent_audit_sha256=file_sha(audit),
        hypothesis='Jointly learn correlated teacher Yes/No patterns and visual referring predictions with weak semantic orientation instead of clustering teacher confidence.',
        observation='Three binary teacher opinions p>.5; p=.5 is uninformative. Missing views exactly marginalized over eight patterns. Every pattern possible under both latent classes.',
        prior='Dirichlet alpha=1+10*product(.8 correct,.2 incorrect); each class/view marginal pseudo-counts8/2, matching existing DS prior strength. No targetGT or inferred majority truth.',
        prevalence='Teacher-only initialization learns selected-evidence prevalence with Beta(2,2) MAP; this is not query absence or video prevalence.',
        teacher_initialization='One deterministic teacher-only MAP-EM fit, at most200 steps, relative objective tolerance1e-8, reused for joint and fused-KD.',
        solver=previous['solver'], optimizer=previous['optimizer'],
        feedback='S bootstrap500 mean-KD; four correlated R/S rounds, S500/500/500/4000 steps. Backtrack or restore on penalized observed-objective decrease.',
        image_prior_clip=[.05,.95], ordinary_minimum_steps=6000, ordinary_maximum_steps=16000,
        ordinary_proper_excess_KL_limit=.005, new_neural_heads=6,
        exact_completed_ordinary_heads_reused=18, exact_completed_raw_opponents_reused=24,
        no_weakened_or_retrained_completed_opponents=True,
        execution='GPU0/1/3 joint primary; GPU2 sequential new fused-KD heads, user-authorized sharing with existing inference service preserved; new GPU2 process memory fraction.10.',
        raw_records=30, dual_count_records=54, official_statistical_records=84,
        paired_effects=81, CountSelf_exact_aliases=30,
        success_screen='Raw and BOTH CountMatched effects positive versus ALL9 opponents in EVERY one-video case. Same strict screen as parent; no retrospective success redefinition.',
        all84_sealed_before_this_study_GT=True, no_module_ablation_matrix=True,
        source_teacher_responses=[c['logical_source_responses'] for c in previous['cases']],
        new_teacher_GPU_calls=0, source_features_prepaid=True,
        auxiliary_scope=previous['auxiliary_scope'],
        same_target_pictures_queries_teacher_answers_and_candidates_for_all=True,
        no_target_manual_GT_for_training_acquisition_or_checkpoint_selection=True,
        all_current_data_DEV_with_previous_evaluation_exposure=True,
        no_online_acquisition_full_six_setting_three_contribution_or_fresh_cost_claim=True,
        identifiability_HOTA_and_RSI_improvement_not_guaranteed=True, goal_status='ACTIVE')
    assert not guard
    atomic_json(PLAN, plan)
    atomic_json(PUBLIC/(STEM+'_registration.json'), plan)
    print(json.dumps(dict(status=plan['status'], pins=len(SOURCES), new_neural_heads=6)), flush=True)


def registration():
    plan = json.loads(PLAN.read_text())
    parent.registration()
    assert file_sha(parent.PLAN) == plan['parent_registration_sha256']
    assert file_sha(parent.OUT/'independent_sealed_prediction_audit.json') == plan['completed_parent_audit_sha256']
    for name, expected in plan['source_sha256'].items():
        assert file_sha(ROOT/name) == expected, name
    for name, expected in plan['reused_prediction_sha256'].items():
        assert file_sha(name) == expected, name
    return plan


def solve(features, probability, initial, folder, device):
    import numpy as np
    import torch
    from flex_uda.active.joint_teacher_noise import make_solver, predict, fit_solver
    from flex_uda.active.semantic_teacher_patterns import SemanticPatterns, accept_solver
    root = folder/'joint_primary'
    root.mkdir(parents=True, exist_ok=True)
    model = make_solver(device)
    observer = SemanticPatterns.from_state(initial)
    final, saved = root/'final.pt', root/'closed.pt'
    if final.exists():
        state = torch.load(final, map_location='cpu', weights_only=False)
        assert state['registration_sha256'] == file_sha(PLAN)
        model.load_state_dict(state['model'])
        return model, state
    rounds, completed = [], -1
    if saved.exists():
        state = torch.load(saved, map_location='cpu', weights_only=False)
        assert state['registration_sha256'] == file_sha(PLAN)
        model.load_state_dict(state['model'])
        observer = SemanticPatterns.from_state(state['observer'])
        rounds, completed, bootstrap = state['rounds'], state['completed'], state['bootstrap']
    else:
        bootstrap_path = root/'bootstrap.pt'
        if bootstrap_path.exists():
            state = torch.load(bootstrap_path, map_location='cpu', weights_only=False)
            assert state['registration_sha256'] == file_sha(PLAN)
            model.load_state_dict(state['model'])
            bootstrap = state['fit']
        else:
            bootstrap = fit_solver(model, features, np.nanmean(probability, axis=1), device, steps=500)
            torch.save(dict(model=model.state_dict(), fit=bootstrap, registration_sha256=file_sha(PLAN)), bootstrap_path)
    for stage in range(completed+1, 4):
        prior = predict(model, features, device)
        old_posterior = observer.posterior(probability, prior)
        observer_update = observer.update(probability, prior)
        posterior = observer.posterior(probability, prior)
        previous = copy.deepcopy(model.state_dict())
        fit = fit_solver(model, features, posterior, device, steps=4000 if stage == 3 else 500)
        accepted = accept_solver(model, previous, observer, probability, features, device)
        row = dict(stage=stage, observer_update=observer_update, solver_fit_proposal=fit,
                   solver_acceptance=accepted,
                   observer_mean_posterior_change=float(np.abs(posterior-old_posterior).mean()),
                   latent_posterior_not_ground_truth=True)
        rounds.append(row)
        atomic_json(root/('round'+str(stage)+'.json'), row)
        state = dict(model={k: v.detach().cpu().clone() for k,v in model.state_dict().items()},
                     observer=observer.state(), rounds=rounds, completed=stage, bootstrap=bootstrap,
                     data_sha256=file_sha(input_path(folder.name)), registration_sha256=file_sha(PLAN), seed=0)
        temp = saved.with_suffix('.tmp')
        torch.save(state, temp)
        temp.replace(saved)
        print(json.dumps(dict(status='SEMANTIC_PATTERN_CLOSED_ROUND', pair=folder.name, stage=stage,
                              observer_change=observer_update['maximum_parameter_change'],
                              solver_change=accepted['maximum_parameter_change'])), flush=True)
    temp = final.with_suffix('.tmp')
    torch.save(state, temp)
    temp.replace(final)
    return model, state


def ordinary_worker():
    from flex_uda.active.deterministic_observation_runtime import configure_before_cuda
    configure_before_cuda()
    import numpy as np
    import torch
    from flex_uda.active.semantic_teacher_patterns import SemanticPatterns
    from flex_uda.active.joint_teacher_noise import make_solver, fit_solver
    from flex_uda.a100.mechanism import generation_guard
    plan, guard = registration(), generation_guard()
    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(.10)
    for case in plan['cases']:
        pair=case['pair']
        normal=OUT/pair/'ordinary'
        normal.mkdir(parents=True,exist_ok=True)
        path=normal/(NEW_ORDINARY+'.pt')
        if path.exists():
            saved=torch.load(path,map_location='cpu',weights_only=False)
            assert saved['registration_sha256']==file_sha(PLAN) and saved['fit']['adequate_passed']
            continue
        with np.load(input_path(pair)) as saved:
            features,probability=saved['features'],saved['probability']
        initial=json.loads((OUT/pair/'initial_observer.json').read_text())
        assert initial['registration_sha256']==file_sha(PLAN)
        y=SemanticPatterns.from_state(initial['state']).posterior(probability)
        weight=np.ones(len(features))
        model=make_solver('cuda:0')
        fit=fit_solver(model,features,y,'cuda:0',steps=6000,weight=weight,adequate=True)
        state=dict(model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                   fit=fit,name=NEW_ORDINARY,seed=0,data_sha256=case['data_sha256'],
                   registration_sha256=file_sha(PLAN),
                   target_sha256=hashlib.sha256(y.tobytes()+weight.tobytes()).hexdigest())
        temp=path.with_suffix('.tmp')
        torch.save(state,temp); temp.replace(path)
        print(json.dumps(dict(status='ADEQUATE_PATTERN_FUSED_ORDINARY_COMPLETE',pair=pair,
                              steps=fit['steps'],proper_excess_KL=fit['excess_KL'])),flush=True)
        del model
    assert not guard
    atomic_json(OUT/'ordinary_gpu2_complete.json',dict(status='COMPLETE_THREE_NEW_ADEQUATE_HEADS',
                GT_open_attempts=0,shared_GPU2_existing_processes_preserved=True,new_neural_seeds=0))


def worker(pair):
    from flex_uda.active.deterministic_observation_runtime import configure_before_cuda
    execution = configure_before_cuda()
    import numpy as np
    import torch
    from flex_uda.active.joint_teacher_noise import make_solver
    from flex_uda.active.factor_rsi_learning import Inputs
    from flex_uda.active.joint_observation_inputs import JointObservationVideo, use_verified_local_features
    from flex_uda.a100.mechanism import generation_guard, PAIRS
    from flex_uda.presence.predictions import seal_mask, verify_predictions
    from scripts.a100_factor_rsi_primary_quality import subset_metadata, projection
    plan, guard = registration(), generation_guard()
    folder = OUT/pair
    if (folder/'main_completion.json').exists():
        return
    started = time.perf_counter()
    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(.35)
    use_verified_local_features()
    case = next(c for c in plan['cases'] if c['pair'] == pair)
    with np.load(input_path(pair)) as saved:
        features, probability = saved['features'], saved['probability']
    initial_path = folder/'initial_observer.json'
    assert initial_path.is_file(), 'Run registered contract before GPU fits'
    initial = json.loads(initial_path.read_text())
    assert initial['data_sha256'] == case['data_sha256'] and initial['registration_sha256'] == file_sha(PLAN)
    primary, state = solve(features, probability, initial['state'], folder, 'cuda:0')
    model = make_solver('cuda:0')
    normal = folder/'ordinary'
    normal.mkdir(parents=True, exist_ok=True)
    path = normal/(NEW_ORDINARY+'.pt')
    while not path.exists():
        failure=OUT/'ordinary_gpu2_failed.json'
        if failure.exists(): raise RuntimeError(failure.read_text())
        time.sleep(2)
    ordinary = torch.load(path, map_location='cpu', weights_only=False)
    assert ordinary['registration_sha256'] == file_sha(PLAN) and ordinary['fit']['adequate_passed']
    model.load_state_dict(ordinary['model'])
    fit = ordinary['fit']
    fitted = {PRIMARY:primary, NEW_ORDINARY:model}
    video = JointObservationVideo(pair, case['evaluated_video'])
    inputs = Inputs(video)
    keys = json.loads((parent.OUT/pair/'row_keys.json').read_text())['keys']
    queries = {r['expression_id']:i for i,r in enumerate(video.queries)}
    tracks = {int(t):i for i,t in enumerate(video.tracks)}
    points = [i for i,row in enumerate(keys) if row[0] == video.video]
    rows = [dict(local_query=queries[keys[i][1]], local_track=tracks[int(keys[i][2])],
                 frame_id=int(keys[i][3]), track_id=int(keys[i][2])) for i in points]
    assert np.array_equal(parent.pack(inputs.arrays(rows)), features[points])
    scores = {name:np.empty((len(video.queries), len(video.obs)), dtype='float32') for name in fitted}
    with torch.no_grad():
        for qi in range(len(video.queries)):
            for begin in range(0,len(video.obs),512):
                indices = np.arange(begin,min(begin+512,len(video.obs)))
                rows = [dict(local_query=qi, local_track=int(video.columns[r]), frame_id=int(video.obs[r,0]),
                             track_id=int(video.obs[r,1])) for r in indices]
                x = torch.as_tensor(parent.pack(inputs.arrays(rows)), device='cuda:0')
                for name,head in fitted.items():
                    scores[name][qi,indices] = head(x).cpu().numpy()
    for name in parent.COMPARATORS:
        scores[name] = np.load(parent.OUT/pair/'predictions'/name/'scores.npy')
    bits = {name:z>0 for name,z in scores.items()}
    metadata = subset_metadata(case['target'],case['evaluated_video'])
    tasks = []
    for name in METHODS:
        if name in parent.COMPARATORS:
            root = parent.OUT/pair/'predictions'/name
            assert np.array_equal(np.load(root/'mask.npy'), bits[name])
            verify_predictions(root)
        else:
            root = folder/'predictions'/name
            root.mkdir(parents=True,exist_ok=True)
            assert np.isfinite(scores[name]).all()
            np.save(root/'scores.npy',scores[name]); np.save(root/'mask.npy',bits[name])
            assert np.array_equal(projection(scores[name],bits[name],video.obs),bits[name])
            seal_mask(root,metadata,bits[name].flatten(),target=case['target'],model=PAIRS[pair][0],
                      method=name,GT_open_attempts=len(guard),extra=dict(seed=0,matched_data_sha256=case['data_sha256'],
                      full_raw_logits_retained=True, no_new_teacher_calls=True))
        tasks.append(dict(pair=pair,method=name,folder=str(root)))
    for rank,count in [(PRIMARY,n) for n in COMPARATORS]+[(n,PRIMARY) for n in COMPARATORS]:
        name = rank+'__AT_'+count+'_COUNT'
        root = folder/'predictions'/name
        selected = projection(scores[rank],bits[count],video.obs)
        seal_mask(root,metadata,selected.flatten(),target=case['target'],model=PAIRS[pair][0],
                  method=name,GT_open_attempts=len(guard),extra=dict(rank_method=rank,count_method=count,
                  counts_equal_per_query_frame=True,counts_from_GT=False))
        tasks.append(dict(pair=pair,method=name,folder=str(root)))
    assert len(tasks)==28 and not guard
    atomic_json(folder/'main_completion.json',dict(status='SEALED_ALL28_SEMANTIC_PATTERN_CASE_OUTPUTS_NO_GT',
                tasks=tasks,GT_open_attempts=0,ordinary_fit=fit,joint_rounds=state['rounds'],execution=execution,
                wall_seconds=time.perf_counter()-started,reused_ordinary_raw_scores=8,not_full_cold_deployment=True))


def main():
    plan=registration()
    assert (OUT/'contract.json').is_file()
    supervisor=OUT/'supervisor'
    supervisor.mkdir(parents=True,exist_ok=True)
    pending=[(gpu,c) for gpu,c in zip((0,1,3),plan['cases']) if not (OUT/c['pair']/'main_completion.json').exists()]
    for gpu,case in pending:
        pids=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
        used=int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
        assert not pids and used<512,(gpu,pids,used)
    jobs=[]
    pending_ordinary=any(not (OUT/c['pair']/'ordinary'/(NEW_ORDINARY+'.pt')).exists() for c in plan['cases'])
    ordinary_proc=None
    if pending_ordinary:
        used=int(subprocess.check_output(['nvidia-smi','-i','2','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
        total=int(subprocess.check_output(['nvidia-smi','-i','2','--query-gpu=memory.total','--format=csv,noheader,nounits'],text=True).strip())
        residents=subprocess.check_output(['nvidia-smi','-i','2','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader,nounits'],text=True).strip()
        assert total-used>=8192,(used,total)
        atomic_json(supervisor/'GPU2_shared_preflight.json',dict(used_MiB=used,total_MiB=total,
                    existing_processes=residents,new_process_memory_fraction=.10,user_authorized_sharing=True,
                    no_existing_process_stopped_or_modified=True,not_exclusive_deployment_cost=True))
        log=supervisor/'ordinary_GPU2.log'
        stream=log.open('a',buffering=1)
        ordinary_proc=subprocess.Popen([sys.executable,'-u',str(Path(__file__)),'--ordinary-worker'],cwd=ROOT,
             stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,
             env=dict(os.environ,CUDA_VISIBLE_DEVICES='2',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'))
        (OUT/'ordinary_gpu2_failed.json').unlink(missing_ok=True)
        jobs.append((ordinary_proc,stream,log,dict(pair='ORDINARY_GPU2')))
        atomic_json(supervisor/'ordinary_GPU2_launch.json',dict(pid=ordinary_proc.pid,GPU=2,pairs=[c['pair'] for c in plan['cases']]))
        print(json.dumps(dict(status='STARTED_PATTERN_FUSED_ORDINARY_SHARED_GPU2',GPU=2,pid=ordinary_proc.pid)),flush=True)
    for gpu,case in pending:
        log=supervisor/(case['pair']+'.log')
        stream=log.open('a',buffering=1)
        proc=subprocess.Popen([sys.executable,'-u',str(Path(__file__)),'--worker',case['pair']],cwd=ROOT,
             stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,
             env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'))
        jobs.append((proc,stream,log,case))
        atomic_json(supervisor/(case['pair']+'_launch.json'),dict(pid=proc.pid,GPU=gpu,pair=case['pair']))
        print(json.dumps(dict(status='STARTED_SEMANTIC_PATTERN_MAIN',GPU=gpu,pair=case['pair'],pid=proc.pid)),flush=True)
    failures=[]
    for proc,stream,log,case in jobs:
        code=proc.wait(); stream.close()
        if code:
            failure=dict(pair=case['pair'],exit_code=code,tail=log.read_text()[-5000:])
            failures.append(failure)
            if proc is ordinary_proc: atomic_json(OUT/'ordinary_gpu2_failed.json',failure)
    if failures: raise RuntimeError(failures)
    from flex_uda.presence.predictions import verify_predictions
    tasks=[]
    for case in plan['cases']:
        done=json.loads((OUT/case['pair']/'main_completion.json').read_text())
        assert done['GT_open_attempts']==0
        tasks.extend(done['tasks'])
    assert len(tasks)==84
    for task in tasks: verify_predictions(task['folder'])
    atomic_json(OUT/'all84_sealed_before_GT.json',dict(tasks=tasks,GT_open_attempts=0,registration_sha256=file_sha(PLAN)))
    print('ALL84_SEMANTIC_PATTERN_OUTPUTS_SEALED_ISOLATED_OFFICIAL_EVALUATION',flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        completed=list(pool.map(parent.evaluate_one,tasks))
    results={c['pair']:{} for c in plan['cases']}
    for task,value in completed: results[task['pair']][task['method']]=value
    comparisons=[]
    for pair,values in results.items():
        primary=values[PRIMARY]['HOTA_unrounded']
        for other in COMPARATORS:
            reference=values[other]['HOTA_unrounded']
            comparisons.append(dict(pair=pair,comparator=other,primary_HOTA=primary,ordinary_HOTA=reference,
                raw_delta_HOTA=primary-reference,
                primary_at_ordinary_count_delta_HOTA=values[PRIMARY+'__AT_'+other+'_COUNT']['HOTA_unrounded']-reference,
                ordinary_at_primary_count_delta_HOTA=primary-values[other+'__AT_'+PRIMARY+'_COUNT']['HOTA_unrounded']))
    eligible=all(min(r['raw_delta_HOTA'],r['primary_at_ordinary_count_delta_HOTA'],r['ordinary_at_primary_count_delta_HOTA'])>0 for r in comparisons)
    atomic_json(OUT/'quality_results.json',dict(status='COMPLETE_SEMANTIC_PATTERN_PRIMARY',results=results,
                comparisons=comparisons,candidate_eligible_for_larger_screen=eligible,all_data_DEV=True,
                no_six_setting_or_three_innovation_or_fresh_cost_claim=True,goal_status='ACTIVE'))
    for flag in ('--collect','--reproduce'):
        subprocess.run([sys.executable,'-u','scripts/a100_semantic_pattern_teacher_statistics.py',flag],cwd=ROOT,check=True)
    print(json.dumps(dict(status='COMPLETE_SEMANTIC_PATTERN_MAIN_REPRODUCED',comparisons=comparisons)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--worker')
    parser.add_argument('--ordinary-worker',action='store_true')
    args=parser.parse_args()
    if args.prepare: prepare()
    elif args.worker: worker(args.worker)
    elif args.ordinary_worker: ordinary_worker()
    else: main()
