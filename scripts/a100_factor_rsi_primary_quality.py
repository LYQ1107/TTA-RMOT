"""Adequate ordinary KD and isolated, all-sealed quality for the new pilot.

This tests one primary against ordinary comparators. No module ablations,
parameter selection using manual truth, or old-native quality reads occur.
"""
import argparse
import concurrent.futures
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json, file_sha
from scripts.a100_factor_rsi_primary_pilot import OUT, PLAN, PRIMARY, COMPARATORS, verify_plan
QUALITY_PLAN = OUT/'quality_registration.json'
METHODS = (PRIMARY, *COMPARATORS)


def prepare():
    plan=verify_plan()
    sources=('scripts/a100_factor_rsi_primary_quality.py','flex_uda/presence/predictions.py',
             'flex_uda/a100/evaluation_reuse.py','scripts/active_eval.py')
    row=dict(status='PREREGISTERED_MAIN_VS_ORDINARY_QUALITY_BEFORE_NEW_GT',
        primary_registration_sha256=file_sha(PLAN),source_sha256={p:file_sha(ROOT/p) for p in sources},
        primary=PRIMARY,comparators=COMPARATORS,neural_seeds=[0],
        cases=plan['cases'],ordinary_prefix_min_steps=500,ordinary_final_min_steps=4000,
        ordinary_each_prefix_max_steps=16000,widths=[128,256],per_role_attainable_excess_KL_limit=.005,
        ordinary_checkpoint_selection='Minimum maximum per-role proper teacher excess risk; no manual labels.',
        all_methods_same_raw_inputs_and_paid_answers=True,
        raw_readout='Whole-expression logit>0 for every method; no method-specific count or output calibration.',
        projections='Per query/frame fixed donor count; descending recipient whole logit, stable original observation order for ties.',
        raw_methods=15,dual_CountMatched_methods=24,CountSelf_exact_aliases=15,official_unique_evaluations=39,
        all_39_prediction_seals_before_isolated_GT=True,
        success_screen='Primary raw improvement over every registered adequate comparator in every case, with both CountMatched directions also reported. Screening only, not population significance or six-setting success.',
        no_module_ablation_matrix=True,all_data_DEV=True,source_features_prepaid=True,full_cold_deployment_cost_not_measured=True)
    atomic_json(QUALITY_PLAN,row);print(json.dumps(dict(status=row['status'],evaluations=39)),flush=True)


def registration():
    plan=verify_plan();q=json.loads(QUALITY_PLAN.read_text())
    assert file_sha(PLAN)==q['primary_registration_sha256']
    for name,sha in q['source_sha256'].items():assert file_sha(ROOT/name)==sha,name
    return plan,q


def adequate(name,inputs,paid,prefixes,folder):
    import numpy as np
    import torch
    from flex_uda.active.factor_evidence import OrdinaryFactorKD
    from flex_uda.active.factor_rsi_learning import fit
    hard='HARD' in name;whole_only='_WHOLE_' in name
    for width in (128,256):
        torch.manual_seed(0);model=OrdinaryFactorKD(width=width).cuda()
        reports=[];failed=False;root=folder/name/('width'+str(width));root.mkdir(parents=True,exist_ok=True)
        for pi,n in enumerate(prefixes):
            checkpoint=root/('prefix'+str(pi)+'.pt')
            if checkpoint.exists():
                state=torch.load(checkpoint,map_location='cpu',weights_only=False)
                assert state['labels']==n and state['teacher_data_sha256']==file_sha(folder/'paid_answers.json')
                model.load_state_dict(state['model']);reports.append(state['report']);continue
            minimum=4000 if pi==len(prefixes)-1 else 500
            steps=0;best=None;best_report=None;best_risk=float('inf');attempts=[]
            while steps<16000:
                chunk=minimum if steps==0 else min(1000,16000-steps)
                report=fit(model,inputs,paid[:n],steps=chunk,device='cuda:0',hard=hard,whole_only=whole_only)
                steps+=chunk;attempts.append(report)
                risk=max(r['excess_KL'] for r in report['roles'].values())
                assert np.isfinite(risk)
                if risk<best_risk:
                    best_risk=risk;best_report=report
                    best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                if best_risk<=.005:break
            model.load_state_dict(best)
            report=dict(best_report,total_actual_steps=steps,optimization_attempts=attempts,
                best_maximum_per_role_excess_KL=best_risk,teacher_fit_gate_passed=best_risk<=.005,
                fit_prefix=pi,paid_prefix_length=n,width=width)
            if best_risk>.005:
                atomic_json(root/'failed_adequacy.json',report);failed=True;break
            temporary=checkpoint.with_suffix('.tmp');torch.save(dict(model=best,report=report,labels=n,
                teacher_data_sha256=file_sha(folder/'paid_answers.json')),temporary);temporary.replace(checkpoint)
            reports.append(report)
            print(json.dumps(dict(status='NEW_FACTOR_ORDINARY_PROPER_FIT',case=folder.name,method=name,
                prefix=pi,width=width,steps=steps,max_excess_KL=best_risk)),flush=True)
        if not failed:
            result=dict(model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                width=width,reports=reports,name=name,hard=hard,whole_only=whole_only,
                teacher_data_sha256=file_sha(folder/'paid_answers.json'))
            torch.save(result,folder/(name+'.pt'));return model.eval(),result
        del model;torch.cuda.empty_cache()
    raise RuntimeError(('Ordinary comparator not yet adequate; preserve gate and repair unfinished fit.',folder,name))


def subset_metadata(target,video):
    from flex_uda.a100.mechanism import metadata
    md=metadata(target);queries=[copy.deepcopy(q) for q in md['episodes'] if q['video_id']==video]
    n=len(md['videos'][video]['observations'])
    for qi,q in enumerate(queries):
        assert q['stop']-q['start']==n
        q['start'],q['stop']=qi*n,(qi+1)*n
    return dict(episodes=queries,videos={video:md['videos'][video]},candidate_query_rows=len(queries)*n)


def projection(scores,donor,observations):
    import numpy as np
    projected=np.zeros_like(donor,dtype=bool)
    for frame in np.unique(observations[:,0]):
        ids=np.flatnonzero(observations[:,0]==frame)
        for qi in range(len(scores)):
            k=int(donor[qi,ids].sum())
            order=np.argsort(-scores[qi,ids],kind='stable')
            projected[qi,ids[order[:k]]]=True
            assert int(projected[qi,ids].sum())==k
    return projected


def case_worker(case):
    import numpy as np
    import torch
    from flex_uda.a100.mechanism import generation_guard,PAIRS
    from flex_uda.active.factor_evidence import FactorStudent,OrdinaryFactorKD
    from flex_uda.active.factor_rsi_learning import Inputs,tensor_data
    from flex_uda.active.joint_observation_inputs import JointObservationVideo,use_verified_local_features
    from flex_uda.presence.predictions import seal_mask
    guard=generation_guard();use_verified_local_features();pair=case['pair'];folder=OUT/pair
    if (folder/'all_case_predictions_complete.json').exists():return
    doc=json.loads((folder/'paid_answers.json').read_text());paid=doc['paid'];prefixes=doc['fit_prefix_lengths']
    assert len(paid)==case['budget'] and prefixes[-1]==len(paid) and doc['manual_GT_open_attempts']==0
    v=JointObservationVideo(pair,case['video']);inputs=Inputs(v)
    primary=torch.load(folder/'primary.pt',map_location='cpu',weights_only=False)
    assert np.array_equal(primary['mean'],inputs.mean) and np.array_equal(primary['std'],inputs.std)
    model=FactorStudent(width=primary['width']).cuda().eval();model.load_state_dict(primary['model'])
    models={PRIMARY:model};fits={PRIMARY:primary['final_fit']}
    for name in COMPARATORS:
        if (folder/(name+'.pt')).exists():
            state=torch.load(folder/(name+'.pt'),map_location='cpu',weights_only=False)
            assert state['teacher_data_sha256']==file_sha(folder/'paid_answers.json')
            model=OrdinaryFactorKD(width=state['width']).cuda().eval();model.load_state_dict(state['model'])
        else:model,state=adequate(name,inputs,paid,prefixes,folder)
        assert all(r['teacher_fit_gate_passed'] and r['best_maximum_per_role_excess_KL']<=.005 for r in state['reports'])
        fits[name]=state['reports'];models[name]=model
    scores={name:np.empty((len(v.queries),len(v.obs)),dtype='float32') for name in METHODS}
    with torch.no_grad():
        for qi in range(len(v.queries)):
            for begin in range(0,len(v.obs),512):
                ids=np.arange(begin,min(begin+512,len(v.obs)))
                requests=[dict(local_query=qi,local_track=int(v.columns[i]),frame_id=int(v.obs[i,0]),track_id=int(v.obs[i,1])) for i in ids]
                data=tensor_data(inputs.arrays(requests),'cuda:0')
                for name,learner in models.items():scores[name][qi,ids]=learner(**data)[:,0].cpu().numpy()
    md=subset_metadata(case['target'],case['video']);masks={name:score>0 for name,score in scores.items()}
    tasks=[];countself={}
    for name in METHODS:
        path=folder/'predictions'/name;path.mkdir(parents=True,exist_ok=True)
        assert np.isfinite(scores[name]).all()
        np.save(path/'scores.npy',scores[name]);np.save(path/'mask.npy',masks[name])
        seal_mask(path,md,masks[name].flatten(),target=case['target'],model=PAIRS[pair][0],method=name,
            GT_open_attempts=len(guard),extra=dict(seed=0,first_video_DEV_only=True,teacher_data_sha256=file_sha(folder/'paid_answers.json')))
        tasks.append(dict(method=name,folder=str(path),pair=pair))
        countself[name]=int((projection(scores[name],masks[name],v.obs)!=masks[name]).sum());assert countself[name]==0
    for rank,count in [(PRIMARY,c) for c in COMPARATORS]+[(c,PRIMARY) for c in COMPARATORS]:
        name=rank+'__AT_'+count+'_COUNT';path=folder/'predictions'/name
        bits=projection(scores[rank],masks[count],v.obs)
        seal_mask(path,md,bits.flatten(),target=case['target'],model=PAIRS[pair][0],method=name,
            GT_open_attempts=len(guard),extra=dict(seed=0,rank_method=rank,count_method=count,
                count_equal_per_query_frame=True,counts_from_GT=False,diagnostic_not_deployment=True))
        tasks.append(dict(method=name,folder=str(path),pair=pair))
    assert not guard and len(tasks)==13
    atomic_json(folder/'all_case_predictions_complete.json',dict(status='PASS_CASE_RAW_DUAL_COUNT_SEALED_NO_GT',
        tasks=tasks,CountSelf_changed_rows=countself,GT_open_attempts=0,ordinary_fits=fits,
        source_features_prepaid=True,old_native_GT_not_read=True))
    print(json.dumps(dict(status='NEW_FACTOR_MAIN_ORDINARY_CASE_13_OUTPUTS_SEALED',pair=pair)),flush=True)
    del models,model,v,inputs;torch.cuda.empty_cache()


def worker():
    from flex_uda.active.deterministic_observation_runtime import configure_before_cuda
    execution=configure_before_cuda()
    import torch
    plan,q=registration();torch.set_num_threads(2);torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.35)
    pending=list(plan['cases'])
    while pending:
        ready=[c for c in pending if (OUT/c['pair']/'primary_fit_complete.json').exists()]
        if not ready:time.sleep(20);continue
        for case in ready:case_worker(case);pending.remove(case)
    atomic_json(OUT/'all_quality_case_workers_complete.json',dict(execution=execution,target_GT_read=False))


def evaluate_task(task):
    from flex_uda.a100.evaluation_reuse import evaluate
    return task['pair'],task['method'],evaluate(task['folder'],task['pair'])


def main():
    plan,q=registration();folder=OUT/'supervisor';folder.mkdir(parents=True,exist_ok=True)
    gpu=1
    owners=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
    used=int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
    assert not owners and used<512,(gpu,used,owners)
    log=folder/'adequate_ordinary_quality_gpu1.log'
    with log.open('a',buffering=1) as handle:
        process=subprocess.Popen([sys.executable,'-u',str(Path(__file__)),'--worker'],cwd=ROOT,
            stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'))
        atomic_json(folder/'quality_gpu1_launch.json',dict(pid=process.pid,GPU=gpu,log=str(log),source_sha256=file_sha(Path(__file__))))
        print(json.dumps(dict(status='STARTED_ADEQUATE_ORDINARY_THEN_PRIMARY_QUALITY',GPU=gpu,pid=process.pid)),flush=True)
        code=process.wait()
    if code:raise RuntimeError(dict(exit_code=code,log=str(log),tail=log.read_text()[-6000:]))
    # A completed primary fit is not itself the frozen-weight audit. Require
    # both teacher workers' final seals before opening any evaluation truth.
    while not (OUT/'primary_collection_complete.json').exists():time.sleep(20)
    for gpu in (0,3):
        audit=json.loads((OUT/('teacher_gpu'+str(gpu))/'completion.json').read_text())
        assert audit['teacher']['backend_unmodified'] and audit['target_GT_read'] is False
    # Numerical GPU work and every raw/counterfactual prediction have ended.
    tasks=[]
    from flex_uda.presence.predictions import verify_predictions
    for c in plan['cases']:
        row=json.loads((OUT/c['pair']/'all_case_predictions_complete.json').read_text());assert row['GT_open_attempts']==0
        tasks.extend(row['tasks'])
    assert len(tasks)==q['official_unique_evaluations']==39
    for task in tasks:verify_predictions(task['folder'])
    atomic_json(OUT/'all_39_predictions_sealed_before_GT.json',dict(tasks=tasks,GT_open_attempts=0,
        quality_registration_sha256=file_sha(QUALITY_PLAN),target_GT_not_used_to_select_outputs=True))
    print('ALL_39_NEW_FACTOR_MAIN_ORDINARY_OUTPUTS_SEALED_ISOLATED_EVALUATION',flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:records=list(pool.map(evaluate_task,tasks))
    results={c['pair']:{} for c in plan['cases']}
    for pair,name,result in records:results[pair][name]=result
    comparisons=[]
    for pair,values in results.items():
        primary=values[PRIMARY]['HOTA_unrounded']
        for name in COMPARATORS:
            ordinary=values[name]['HOTA_unrounded']
            comparisons.append(dict(pair=pair,comparator=name,primary_HOTA=primary,ordinary_HOTA=ordinary,
                raw_delta_HOTA=primary-ordinary,
                primary_at_ordinary_count_delta_HOTA=values[PRIMARY+'__AT_'+name+'_COUNT']['HOTA_unrounded']-ordinary,
                ordinary_at_primary_count_delta_HOTA=primary-values[name+'__AT_'+PRIMARY+'_COUNT']['HOTA_unrounded']))
    raw_positive=all(r['raw_delta_HOTA']>0 for r in comparisons)
    counts_positive=all(min(r['primary_at_ordinary_count_delta_HOTA'],r['ordinary_at_primary_count_delta_HOTA'])>0 for r in comparisons)
    result=dict(status='PASS_PRIMARY_SCREEN_EXECUTED_NOT_GOAL_COMPLETE',results=results,comparisons=comparisons,
        every_raw_comparison_positive=raw_positive,every_both_count_comparison_positive=counts_positive,
        candidate_eligible_for_larger_quality_screen=raw_positive and counts_positive,
        all_data_DEV=True,one_video_per_case_not_population_significance=True,
        RSI_module_contribution_unproven=True,six_setting_goal_not_complete=True,
        full_cold_cost_not_measured=True,old_native_incomplete_not_scored=True)
    atomic_json(OUT/'quality_results.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='results'}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--worker',action='store_true');a=p.parse_args()
    if a.prepare:prepare()
    elif a.worker:worker()
    else:main()
