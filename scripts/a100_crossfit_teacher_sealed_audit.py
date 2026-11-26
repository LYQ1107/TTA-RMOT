"""Post-seal independent CPU fit, opposite-track feedback and output counts."""
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    import numpy as np
    import torch
    from scipy.special import expit
    from flex_tta.runtime.pf_receipts import atomic_json,file_sha
    from flex_uda.a100.mechanism import generation_guard
    from flex_uda.presence.predictions import verify_predictions
    from flex_uda.active.joint_teacher_noise import make_solver
    from flex_uda.active.crossfit_teacher_feedback import group_folds,outside_predictions
    from flex_uda.active.semantic_teacher_patterns import SemanticPatterns
    from scripts.a100_crossfit_teacher_primary import OUT,PUBLIC,PLAN,STEM,PRIMARY,SPLIT,METHODS,reference_folder,registration,parent
    from scripts.a100_joint_teacher_noise_sealed_audit import independent_cpu_risk,text_counts
    from scripts.a100_factor_rsi_primary_quality import subset_metadata

    torch.set_num_threads(2);guard=generation_guard();plan=registration()
    barrier=json.loads((OUT/'all93_sealed_before_GT.json').read_text())
    assert len(barrier['tasks'])==93 and barrier['GT_open_attempts']==0
    assert barrier['registration_sha256']==file_sha(PLAN)
    seals={}
    for task in barrier['tasks']:
        key=task['pair'],task['method'];assert key not in seals
        seals[key]=verify_predictions(task['folder'])
    heads=rounds=count_files=self_checks=0;reports=[]
    for case in plan['cases']:
        pair=case['pair'];folder=OUT/pair
        with np.load(parent.input_path(pair)) as data:features,probability=data['features'],data['probability']
        keys=json.loads((parent.parent.OUT/pair/'row_keys.json').read_text())['keys']
        folds=group_folds(keys);assert np.array_equal(folds,np.load(folder/'folds.npy'))
        groups=[{(r[0],int(r[2])) for r,f in zip(keys,folds) if f==k} for k in (0,1)]
        assert not groups[0]&groups[1]
        mean=np.nanmean(probability,axis=1);models=[];ordinary=[];initial=[]
        adaptation_steps=0;fit_wall=0.
        for k in (0,1):
            path=folder/('ordinary_fold'+str(k)+'.pt');saved=torch.load(path,map_location='cpu',weights_only=False)
            assert saved['registration_sha256']==file_sha(PLAN) and saved['data_sha256']==case['data_sha256']
            assert saved['seed']==0 and saved['trained_fold']==k
            selected=folds==k;target=mean[selected]
            assert saved['target_sha256']==hashlib.sha256(target.tobytes()).hexdigest()
            model=make_solver('cpu');model.load_state_dict(saved['model']);model.eval()
            risk=independent_cpu_risk(model,features[selected],target,np.ones(len(target)))
            fit=saved['fit'];error=max(abs(risk[n]-fit[n]) for n in risk)
            assert fit['adequate_requested'] and fit['adequate_passed'] and 6000<=fit['steps']<=16000
            assert risk['excess_KL']<=.005+1e-6 and error<1e-5
            ordinary.append(dict(fold=k,events=len(target),checkpoint_sha256=file_sha(path),CPU_risk=risk,maximum_report_error=error))
            models.append(model);initial.append(saved['model']);heads+=1
            adaptation_steps+=fit['steps'];fit_wall+=fit['wall_seconds']
        before_outside=outside_predictions(models,features,folds,'cpu')
        pair_state=torch.load(folder/'pair_final.pt',map_location='cpu',weights_only=False)
        assert pair_state['registration_sha256']==file_sha(PLAN) and pair_state['data_sha256']==case['data_sha256']
        previous=None
        for stage,row in enumerate(pair_state['rounds']):
            r,s=row['observer_update'],row['solver_acceptance']
            assert r['after']>=r['before']-1.01e-9 and s['after']>=s['before']-1.01e-7
            assert abs(r['after']-s['before'])<1e-7
            if previous is not None:assert abs(previous-r['before'])<1e-7
            previous=s['after'];rounds+=1
            assert len(row['solver_fits'])==2
            for fit in row['solver_fits']:
                assert fit['steps']==(4000 if stage==3 else 500)
                adaptation_steps+=fit['steps'];fit_wall+=fit['wall_seconds']
        assert len(pair_state['rounds'])==4
        for model,state in zip(models,pair_state['models']):model.load_state_dict(state)
        outside=outside_predictions(models,features,folds,'cpu')
        observer=SemanticPatterns.from_state(pair_state['observer'])
        initial_observer=SemanticPatterns.from_state(json.loads((parent.OUT/pair/'initial_observer.json').read_text())['state'])
        target_path=folder/'final_targets.npy'
        assert file_sha(target_path)==pair_state['final_target_sha256']
        target=np.load(target_path);expected=observer.posterior(probability,outside)
        target_error=float(np.abs(target-expected).max());assert target_error<1e-5
        path=folder/'final_head.pt';saved=torch.load(path,map_location='cpu',weights_only=False)
        assert saved['target_sha256']==file_sha(target_path) and saved['registration_sha256']==file_sha(PLAN)
        assert saved['seed']==0 and saved['data_sha256']==case['data_sha256']
        model=make_solver('cpu');model.load_state_dict(saved['model']);model.eval()
        risk=independent_cpu_risk(model,features,target,np.ones(len(target)))
        fit=saved['fit'];error=max(abs(risk[n]-fit[n]) for n in risk)
        assert fit['adequate_passed'] and 6000<=fit['steps']<=16000 and risk['excess_KL']<=.005+1e-6 and error<1e-5
        heads+=1;adaptation_steps+=fit['steps'];fit_wall+=fit['wall_seconds']
        def teacher_proxy(p):
            p=np.clip(p,1e-12,1-1e-12)
            return float((-mean*np.log(p)-(1-mean)*np.log1p(-p)).mean())
        metadata=subset_metadata(case['target'],case['evaluated_video']);obs=np.asarray(metadata['videos'][case['evaluated_video']]['observations'])
        for name in METHODS:
            root=reference_folder(pair,name) if name in parent.COMPARATORS else folder/'predictions'/name
            scores,bits=np.load(root/'scores.npy'),np.load(root/'mask.npy')
            assert scores.shape==bits.shape==(len(metadata['episodes']),len(obs))
            assert np.isfinite(scores).all() and np.array_equal(bits,scores>0)
            for frame in np.unique(obs[:,0]):
                indices=np.flatnonzero(obs[:,0]==frame)
                for qi in range(len(bits)):
                    k=int(bits[qi,indices].sum());chosen=indices[np.lexsort((indices,-scores[qi,indices]))][:k]
                    assert set(chosen)==set(indices[bits[qi,indices]])
            self_checks+=1
        for (p,name),seal in seals.items():
            if p!=pair or 'count_method' not in seal:continue
            by={(r['video'],r['expression']):r for r in seals[p,seal['count_method']]['episodes']}
            for row in seal['episodes']:
                assert text_counts(row['prediction_path'])==text_counts(by[row['video'],row['expression']]['prediction_path'])
                count_files+=1
        reports.append(dict(pair=pair,ordinary_fold_heads=ordinary,final_solver_CPU_risk=risk,final_fit_report_error=error,
            direct_TrackID_group_overlap=0,opposite_fold_mean_proxy_CE_before=teacher_proxy(before_outside),
            opposite_fold_mean_proxy_CE_after=teacher_proxy(outside),
            observer_parameter_change=float(np.abs(observer.table-initial_observer.table).max()),
            peer_parameter_changes=[max(float((v-old[k]).abs().max()) for k,v in current.items()) for current,old in zip(pair_state['models'],initial)],
            CPU_GPU_posterior_maximum_difference=target_error,
            final_corrected_target_mean=float(target.mean()),target_label_changes_from_mean=int(np.count_nonzero((target>.5)!=(mean>.5))),
            final_CPU_penalized_proxy_objective=observer.objective(probability,outside),
            CPU_GPU_proxy_objective_difference=abs(observer.objective(probability,outside)-previous),
            total_adaptation_optimizer_steps=adaptation_steps,neural_fit_wall_seconds_sum=fit_wall,
            fit_wall_excludes_R_feedback_IO_features_teacher_and_not_full_cold=True,
            shared_observer_indirect_coupling_and_teacher_proxy_not_true_validation=True))
    assert heads==9 and rounds==12 and self_checks==33 and not guard
    result=dict(status='PASS_ALL93_SEALS_NINE_NEW_HEAD_CPU_RISKS_AND_GROUP_DISJOINT_FEEDBACK_COUNTS',
        records=93,adequate_heads_recomputed=heads,feedback_rounds_checked=rounds,exact_CountSelf_aliases=self_checks,
        projected_query_files=count_files,cases=reports,manual_GT_reads=0,new_fits_teacher_calls_or_predictions=0,
        no_task_accuracy_or_RSI_contribution_claim=True,source_sha256=file_sha(Path(__file__)))
    atomic_json(OUT/'independent_sealed_prediction_audit.json',result)
    (PUBLIC/(STEM+'_sealed_audit.json')).write_text(json.dumps(result,separators=(',',':'))+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
