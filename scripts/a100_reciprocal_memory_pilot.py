"""Next whole primary screen: teacher-checked reciprocal visual memory.

Fixed paid QA replay. This pilot tests reciprocal neural learning, not fresh
online teacher acquisition or full deployment cost. No module ablations.
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
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from scripts.a100_factor_rsi_primary_pilot import OUT as PARENT,verify_plan
from scripts.a100_six_direction import OUT as SIX
OUT=SIX/'reciprocal_memory_primary_20261125';PLAN=OUT/'registration.json'
PRIMARY='MUTUAL_VISUAL_MEMORY_RSI'
COMPARATORS=('SOFT_INDEPENDENT_DUAL_KD','HARD_INDEPENDENT_DUAL_KD',
    'SOFT_INDEPENDENT_DUAL_POOLED','HARD_INDEPENDENT_DUAL_POOLED',
    'PRIOR_SOFT_MULTITASK_KD','PRIOR_HARD_MULTITASK_KD')
METHODS=(PRIMARY,*COMPARATORS)
RECIPES={'PRIMARY':dict(mutual=True,hard=False),'SOFT_DUAL':dict(mutual=False,hard=False),
         'HARD_DUAL':dict(mutual=False,hard=True)}
SOURCES=('scripts/a100_reciprocal_memory_pilot.py','flex_uda/active/reciprocal_memory.py',
         'scripts/a100_reciprocal_memory_contract.py','scripts/a100_reciprocal_memory_statistics.py',
         'scripts/a100_factor_rsi_primary_quality.py','flex_uda/presence/predictions.py',
         'flex_uda/a100/evaluation_reuse.py')


def prepare():
    from flex_uda.a100.mechanism import generation_guard
    from flex_uda.active.reciprocal_memory import folds
    guard=generation_guard();parent=verify_plan();cases=[]
    for case in parent['cases']:
        case=dict(case);path=PARENT/case['pair']/'paid_answers.json'
        paid=json.loads(path.read_text())['paid'];held=folds(paid)
        case.update(paid_answers=str(path),paid_answers_sha256=file_sha(path),heldout_rows=held)
        cases.append(case)
    plan=dict(status='PREREGISTERED_RECIPROCAL_FULL_INPUT_PRIMARY_BEFORE_FITS',cases=cases,
        primary=PRIMARY,comparators=COMPARATORS,recipes=RECIPES,neural_seeds=[0],
        source_sha256={n:file_sha(ROOT/n) for n in SOURCES},parent_registration_sha256=file_sha(PARENT/'registration.json'),
        hypothesis='Preserve complete ordered inputs; visual/state innovation gates predictive continuity; reciprocal students and new heldout-teacher checks shape unlabeled learning.',
        students='Complete original-input128 plus complete-input-and-dense-track-moments128; independent ordinary duals use the identical two architectures.',
        teacher_budget_per_case_unchanged=True,new_teacher_GPU_calls=0,fixed_paid_QA_replay=True,
        current_and_future_probe_groups_excluded_from_current_fit=True,
        fold_definition='Three metadata-hash folds, five percent of query-track-frame groups each, minimum two; retain at least one training WHOLE per query and one per role.',
        training_steps_per_feedback_round=500,feedback_rounds=3,final_minimum_steps=4000,final_maximum_steps=16000,
        ordinary_all_prefix_and_all_final_student_proper_excess_KL_limit=.005,
        primary_final_both_students_proper_excess_KL_limit=.005,
        ordinary_teacher_Hedge_readout_same_as_primary=True,
        teacher_Hedge='Clipped(.001,.999) teacher logloss, eta=sqrt(8log2/(3*(-log(.001))^2)); bound only for realized validation sequence.',
        graph_weights='Unlabeled RGB/state innovations and time gaps, robust median scales; query type from existing lexical rules, not GT.',
        curriculum='Model-disagreement sampling with .2 uniform exploration; opinions are a regularizer, not true labels.',
        memory='Dense full-track mean/std in feature space; full-video transductive, original ordered anchors/current retained.',
        raw_methods=21,dual_CountMatched_methods=36,official_statistical_records=57,
        old_reference_raw_records=6,CountSelf_exact_aliases=21,
        all57_prediction_seals_before_current_study_GT=True,
        success_screen='Primary raw and both CountMatched directions better than all six ordinary controls in every case; screening only, no population significance or six-setting claim.',
        same_visual_features_and_paid_teacher_answers=True,no_manual_GT_for_training_sampling_weights_or_checkpoints=True,
        no_module_ablation_matrix=True,existing_DEV_only=True,not_independent_blind_validation=True,
        no_fresh_online_acquisition_or_same_execution_full_cold_cost_claim=True,goal_status='ACTIVE')
    assert not guard;atomic_json(PLAN,plan)
    print(json.dumps(dict(status=plan['status'],cases=[dict(pair=c['pair'],budget=c['budget'],heldout_rows=list(map(len,c['heldout_rows']))) for c in cases])),flush=True)


def registration():
    plan=json.loads(PLAN.read_text());verify_plan()
    assert file_sha(PARENT/'registration.json')==plan['parent_registration_sha256']
    for n,s in plan['source_sha256'].items():assert file_sha(ROOT/n)==s,n
    return plan


def solve(name,inputs,paid,heldout,folder,device):
    import numpy as np
    import torch
    from flex_uda.active.reciprocal_memory import models,train,probabilities,update_experts,curriculum
    recipe=RECIPES[name];root=folder/'fits'/name;root.mkdir(parents=True,exist_ok=True)
    pair=models(device);weights=np.full((2,3),.5);generator=np.random.default_rng(0);rounds=[];completed=-1
    final=root/'final.pt'
    if final.exists():
        checkpoint=torch.load(final,map_location='cpu',weights_only=False)
        assert checkpoint['paid_answers_sha256']==file_sha(Path(folder)/'paid_answers.json')
        for m,s in zip(pair,checkpoint['models']):m.load_state_dict(s);m.eval()
        return pair,np.asarray(checkpoint['weights']),checkpoint
    saved=root/'closed.pt'
    if saved.exists():
        state=torch.load(saved,map_location='cpu',weights_only=False)
        for m,s in zip(pair,state['models']):m.load_state_dict(s)
        weights=np.asarray(state['weights']);generator.bit_generator.state=state['generator_state']
        rounds=state['rounds'];completed=state['completed']
    def states():return [{k:v.detach().cpu().clone() for k,v in m.state_dict().items()} for m in pair]
    for stage in range(completed+1,3):
        forbidden=set(i for f in heldout[stage:] for i in f);ids=[i for i in range(len(paid)) if i not in forbidden]
        training=[paid[i] for i in ids];probe=[paid[i] for i in heldout[stage]]
        assert not set(ids)&set(heldout[stage])
        edges=None;selection=None
        if recipe['mutual']:
            path=root/('selection'+str(stage)+'.pt')
            if path.exists():
                s=torch.load(path,map_location='cpu',weights_only=False);selection=s['selection']
                generator.bit_generator.state=s['generator_state'];edges=inputs.edge_data(np.asarray(selection['selected_edge_ids']))
            else:
                edges,selection=curriculum(pair,inputs,weights,generator,device)
                torch.save(dict(selection=selection,generator_state=generator.bit_generator.state),path)
        fit=train(pair,inputs,training,device=device,steps=500,hard=recipe['hard'],mutual=recipe['mutual'],
            edges=edges,weights=weights,minimum_gate=not recipe['mutual'])
        prediction=probabilities(pair,inputs.arrays(probe),device)
        weights,critic=update_experts(weights,prediction,probe)
        row=dict(stage=stage,fit=fit,critic=critic,curriculum=selection,training_indices=ids,
            withheld_indices=heldout[stage],current_and_future_probe_groups_excluded=True)
        rounds.append(row);atomic_json(root/('round'+str(stage)+'.json'),row)
        state=dict(models=states(),weights=weights,rounds=rounds,completed=stage,generator_state=generator.bit_generator.state)
        temp=saved.with_suffix('.tmp');torch.save(state,temp);temp.replace(saved)
        print(json.dumps(dict(status='RECIPROCAL_PRIMARY_OR_ORDINARY_CLOSED_ROUND',pair=folder.name,
            recipe=name,stage=stage,teacher_maximum_excess_KL=fit['maximum_excess_KL'],teacher_weights=weights.tolist())),flush=True)
    edges=None;selection=None
    if recipe['mutual']:
        path=root/'final_selection.pt'
        if path.exists():
            s=torch.load(path,map_location='cpu',weights_only=False);selection=s['selection']
            generator.bit_generator.state=s['generator_state'];edges=inputs.edge_data(np.asarray(selection['selected_edge_ids']))
        else:
            edges,selection=curriculum(pair,inputs,weights,generator,device)
            torch.save(dict(selection=selection,generator_state=generator.bit_generator.state),path)
    try:
        fit=train(pair,inputs,paid,device=device,steps=4000,hard=recipe['hard'],mutual=recipe['mutual'],
            edges=edges,weights=weights,minimum_gate=True)
    except Exception:
        torch.save(dict(models=states(),weights=weights,rounds=rounds),root/'failed_final_state.pt');raise
    checkpoint=dict(models=states(),weights=weights,rounds=rounds,final_fit=fit,final_curriculum=selection,
        recipe=recipe,paid_answers_sha256=file_sha(folder/'paid_answers.json'),mean=inputs.mean,std=inputs.std,
        neural_seed=0,no_GT_selection=True,teacher_weights_from_pre_final_heldout_models_not_a_final_risk_bound=True)
    temp=final.with_suffix('.tmp');torch.save(checkpoint,temp);temp.replace(final)
    return pair,weights,checkpoint


def worker(case_name):
    from flex_uda.active.deterministic_observation_runtime import configure_before_cuda
    execution=configure_before_cuda()
    import numpy as np
    import torch
    from flex_uda.a100.mechanism import generation_guard,PAIRS
    from flex_uda.active.joint_observation_inputs import JointObservationVideo,use_verified_local_features
    from flex_uda.active.reciprocal_memory import MemoryInputs,tensor_data
    from scripts.a100_factor_rsi_primary_quality import subset_metadata,projection
    from flex_uda.presence.predictions import seal_mask
    plan=registration();guard=generation_guard();torch.set_num_threads(2);torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(.35);use_verified_local_features()
    case=next(c for c in plan['cases'] if c['pair']==case_name);folder=OUT/case_name;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'completion.json').exists():return
    path=Path(case['paid_answers']);assert file_sha(path)==case['paid_answers_sha256']
    copied=folder/'paid_answers.json'
    if copied.exists():assert file_sha(copied)==file_sha(path)
    else:copied.write_bytes(path.read_bytes())
    paid=json.loads(copied.read_text())['paid'];start=time.perf_counter()
    v=JointObservationVideo(case_name,case['video']);inputs=MemoryInputs(v);pairs={};weights={};receipts={}
    for name in RECIPES:
        pairs[name],weights[name],receipts[name]=solve(name,inputs,paid,case['heldout_rows'],folder,'cuda:0')
    probability={name:np.empty((len(v.queries),len(v.obs)),dtype='float32') for name in RECIPES}
    with torch.no_grad():
        for qi in range(len(v.queries)):
            for begin in range(0,len(v.obs),512):
                ids=np.arange(begin,min(begin+512,len(v.obs)))
                data=tensor_data(inputs.at(np.full(len(ids),qi),ids),'cuda:0')
                for name,models in pairs.items():
                    p=sum(float(weights[name][j,0])*model(**data)[:,0].sigmoid() for j,model in enumerate(models))
                    probability[name][qi,ids]=p.cpu().numpy()
    def logit(p):
        p=p.clip(1e-7,1-1e-7);return (np.log(p)-np.log1p(-p)).astype('float32')
    scores={PRIMARY:logit(probability['PRIMARY']),COMPARATORS[0]:logit(probability['SOFT_DUAL']),COMPARATORS[1]:logit(probability['HARD_DUAL'])}
    for base,name in (('SOFT_DUAL',COMPARATORS[2]),('HARD_DUAL',COMPARATORS[3])):
        p=probability[base].copy()
        for ti in range(len(v.tracks)):
            ids=np.flatnonzero(v.columns==ti);p[:,ids]=p[:,ids].mean(1)[:,None]
        scores[name]=logit(p)
    for name,old in ((COMPARATORS[4],'ORDINARY_SOFT_MULTITASK_KD'),(COMPARATORS[5],'ORDINARY_HARD_MULTITASK_KD')):
        scores[name]=np.load(PARENT/case_name/'predictions'/old/'scores.npy')
    masks={n:s>0 for n,s in scores.items()};md=subset_metadata(case['target'],case['video']);tasks=[]
    for name in METHODS:
        root=folder/'predictions'/name;root.mkdir(parents=True,exist_ok=True)
        assert scores[name].shape==(len(v.queries),len(v.obs)) and np.isfinite(scores[name]).all()
        np.save(root/'scores.npy',scores[name]);np.save(root/'mask.npy',masks[name])
        assert np.array_equal(projection(scores[name],masks[name],v.obs),masks[name])
        seal_mask(root,md,masks[name].flatten(),target=case['target'],model=PAIRS[case_name][0],method=name,
            GT_open_attempts=len(guard),extra=dict(seed=0,fixed_paid_QA_replay=True,new_teacher_calls=0,source_features_prepaid=True))
        tasks.append(dict(pair=case_name,method=name,folder=str(root)))
    for rank,count in [(PRIMARY,n) for n in COMPARATORS]+[(n,PRIMARY) for n in COMPARATORS]:
        name=rank+'__AT_'+count+'_COUNT';root=folder/'predictions'/name;bits=projection(scores[rank],masks[count],v.obs)
        seal_mask(root,md,bits.flatten(),target=case['target'],model=PAIRS[case_name][0],method=name,
            GT_open_attempts=len(guard),extra=dict(rank_method=rank,count_method=count,counts_equal_per_query_frame=True,counts_from_GT=False))
        tasks.append(dict(pair=case_name,method=name,folder=str(root)))
    assert len(tasks)==19 and not guard
    fit_summary={n:dict(final=receipt['final_fit'],rounds=[r['fit'] for r in receipt['rounds']],weights=np.asarray(receipt['weights']).tolist()) for n,receipt in receipts.items()}
    atomic_json(folder/'completion.json',dict(status='SEALED_COMPLETE_RECIPROCAL_PRIMARY_AND_ORDINARY_NO_GT',
        tasks=tasks,GT_open_attempts=0,fit_summary=fit_summary,execution=execution,wall_seconds=time.perf_counter()-start,
        no_fresh_teacher_calls=True,not_full_cold_deployment=True,CountSelf_exact=True))
    print(json.dumps(dict(status='SEALED_RECIPROCAL_CASE_ALL19_OUTPUTS',pair=case_name)),flush=True)


def evaluate_one(task):
    from flex_uda.a100.evaluation_reuse import evaluate
    return task,evaluate(task['folder'],task['pair'])


def main():
    plan=registration();jobs=[];folder=OUT/'supervisor';folder.mkdir(parents=True,exist_ok=True)
    pending=[(gpu,c) for gpu,c in zip((0,1,3),plan['cases']) if not (OUT/c['pair']/'completion.json').exists()]
    for gpu,case in pending:
        pids=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
        used=int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
        assert not pids and used<512,(gpu,pids,used)
    for gpu,case in pending:
        log=folder/(case['pair']+'.log');handle=log.open('a',buffering=1)
        proc=subprocess.Popen([sys.executable,'-u',str(Path(__file__)),'--worker',case['pair']],cwd=ROOT,
            stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'))
        jobs.append((proc,handle,log,case));atomic_json(folder/(case['pair']+'_launch.json'),dict(pid=proc.pid,GPU=gpu,pair=case['pair']))
        print(json.dumps(dict(status='STARTED_NEW_RECIPROCAL_MEMORY_MAIN_CASE',pair=case['pair'],GPU=gpu,pid=proc.pid)),flush=True)
    failures=[]
    for proc,handle,log,case in jobs:
        code=proc.wait();handle.close()
        if code:failures.append(dict(pair=case['pair'],exit_code=code,tail=log.read_text()[-5000:]))
    if failures:raise RuntimeError(failures)
    from flex_uda.presence.predictions import verify_predictions
    tasks=[]
    for case in plan['cases']:
        done=json.loads((OUT/case['pair']/'completion.json').read_text());assert done['GT_open_attempts']==0
        tasks.extend(done['tasks'])
    assert len(tasks)==57
    for t in tasks:verify_predictions(t['folder'])
    atomic_json(OUT/'all57_sealed_before_GT.json',dict(tasks=tasks,GT_open_attempts=0,registration_sha256=file_sha(PLAN)))
    print('ALL57_RECIPROCAL_PRIMARY_OUTPUTS_SEALED_ISOLATED_OFFICIAL_EVALUATION',flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:completed=list(pool.map(evaluate_one,tasks))
    results={c['pair']:{} for c in plan['cases']}
    for task,value in completed:results[task['pair']][task['method']]=value
    comparisons=[]
    for pair,values in results.items():
        p=values[PRIMARY]['HOTA_unrounded']
        for other in COMPARATORS:
            o=values[other]['HOTA_unrounded']
            comparisons.append(dict(pair=pair,comparator=other,primary_HOTA=p,ordinary_HOTA=o,raw_delta_HOTA=p-o,
                primary_at_ordinary_count_delta_HOTA=values[PRIMARY+'__AT_'+other+'_COUNT']['HOTA_unrounded']-o,
                ordinary_at_primary_count_delta_HOTA=p-values[other+'__AT_'+PRIMARY+'_COUNT']['HOTA_unrounded']))
    atomic_json(OUT/'quality_results.json',dict(status='COMPLETE_RECIPROCAL_PRIMARY_SCREEN_NOT_GOAL_COMPLETE',results=results,
        comparisons=comparisons,every_raw_comparison_positive=all(r['raw_delta_HOTA']>0 for r in comparisons),
        every_both_count_comparison_positive=all(min(r['primary_at_ordinary_count_delta_HOTA'],r['ordinary_at_primary_count_delta_HOTA'])>0 for r in comparisons),
        candidate_eligible_for_larger_screen=all(min(r['raw_delta_HOTA'],r['primary_at_ordinary_count_delta_HOTA'],r['ordinary_at_primary_count_delta_HOTA'])>0 for r in comparisons),
        no_three_contribution_or_six_setting_or_fresh_RSI_acquisition_claim=True,all_data_DEV=True,goal_status='ACTIVE'))
    for flag in ('--collect','--reproduce'):
        subprocess.run([sys.executable,'-u','scripts/a100_reciprocal_memory_statistics.py',flag],cwd=ROOT,check=True)
    print(json.dumps(dict(status='COMPLETE_RECIPROCAL_PRIMARY_SCREEN_REPRODUCED',comparisons=comparisons)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--worker');a=p.parse_args()
    if a.prepare:prepare()
    elif a.worker:worker(a.worker)
    else:main()
