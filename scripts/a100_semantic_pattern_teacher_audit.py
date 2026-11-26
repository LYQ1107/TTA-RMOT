"""Post-seal CPU teacher-risk, feedback objective and physical-count audit."""
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    import numpy as np
    import torch
    from flex_tta.runtime.pf_receipts import atomic_json,file_sha
    from flex_uda.a100.mechanism import generation_guard
    from flex_uda.active.joint_teacher_noise import make_solver,predict
    from flex_uda.active.semantic_teacher_patterns import SemanticPatterns,PATTERNS
    from flex_uda.presence.predictions import verify_predictions
    from scripts.a100_semantic_pattern_teacher_primary import OUT,PUBLIC,PLAN,STEM,PRIMARY,NEW_ORDINARY,METHODS,input_path,registration,parent
    from scripts.a100_semantic_pattern_teacher_contract import explicit
    from scripts.a100_joint_teacher_noise_sealed_audit import independent_cpu_risk,text_counts
    from scripts.a100_factor_rsi_primary_quality import subset_metadata

    torch.set_num_threads(2)
    guard=generation_guard()
    plan=registration()
    barrier=json.loads((OUT/'all84_sealed_before_GT.json').read_text())
    assert len(barrier['tasks'])==84 and barrier['GT_open_attempts']==0
    assert barrier['registration_sha256']==file_sha(PLAN)
    seals={}
    for task in barrier['tasks']:
        key=task['pair'],task['method']
        assert key not in seals
        seals[key]=verify_predictions(task['folder'])
    initial_sha={c['pair']:c['observer_sha256'] for c in json.loads((OUT/'contract.json').read_text())['cases']}
    reports=[]
    rounds=self_checks=count_files=0
    for case in plan['cases']:
        pair=case['pair'];folder=OUT/pair
        assert file_sha(folder/'initial_observer.json')==initial_sha[pair]
        initial=json.loads((folder/'initial_observer.json').read_text())
        with np.load(input_path(pair)) as data:
            features,probability=data['features'],data['probability']
        teacher=SemanticPatterns.from_state(initial['state'])
        target=teacher.posterior(probability)
        independent_target,_=explicit(teacher,probability)
        target_error=float(np.abs(target-independent_target).max())
        assert target_error<1e-12
        weight=np.ones(len(target))
        path=folder/'ordinary'/(NEW_ORDINARY+'.pt')
        ordinary=torch.load(path,map_location='cpu',weights_only=False)
        assert ordinary['seed']==0 and ordinary['data_sha256']==case['data_sha256']
        assert ordinary['registration_sha256']==file_sha(PLAN)
        assert ordinary['target_sha256']==hashlib.sha256(target.tobytes()+weight.tobytes()).hexdigest()
        fit=ordinary['fit']
        assert 6000<=fit['steps']<=16000 and fit['adequate_requested'] and fit['adequate_passed']
        assert not fit['target_manual_GT_used']
        model=make_solver('cpu');model.load_state_dict(ordinary['model']);model.eval()
        risk=independent_cpu_risk(model,features,target,weight)
        report_error=max(abs(risk[k]-fit[k]) for k in risk)
        assert report_error<1e-5 and risk['excess_KL']<=.005+1e-6
        final=torch.load(folder/'joint_primary/final.pt',map_location='cpu',weights_only=False)
        bootstrap=torch.load(folder/'joint_primary/bootstrap.pt',map_location='cpu',weights_only=False)
        assert final['seed']==0 and final['registration_sha256']==file_sha(PLAN)
        assert final['data_sha256']==case['data_sha256'] and bootstrap['fit']['steps']==500
        assert len(final['rounds'])==4
        previous=None
        for stage,row in enumerate(final['rounds']):
            r,s=row['observer_update'],row['solver_acceptance']
            assert r['after']>=r['before']-1.01e-9 and s['after']>=s['before']-1.01e-7
            assert abs(r['after']-s['before'])<1e-7
            if previous is not None: assert abs(previous-r['before'])<1e-7
            assert row['solver_fit_proposal']['steps']==(4000 if stage==3 else 500)
            previous=s['after'];rounds+=1
        observer=SemanticPatterns.from_state(final['observer'])
        model=make_solver('cpu');model.load_state_dict(final['model']);model.eval()
        image=predict(model,features,'cpu')
        # Match the observer's specified float64 conversion BEFORE clipping.
        # Clipping float32 scalar priors changes the .05/.95 boundaries.
        _,ll=explicit(observer,probability,np.asarray(image,dtype='float64'))
        independent_objective=ll+10*float(np.sum(observer.base*np.log(observer.table)))
        assert abs(independent_objective-observer.objective(probability,image))<1e-8
        metadata=subset_metadata(case['target'],case['evaluated_video'])
        obs=np.asarray(metadata['videos'][case['evaluated_video']]['observations'])
        for name in METHODS:
            root=(parent.OUT if name in parent.COMPARATORS else OUT)/pair/'predictions'/name
            scores,bits=np.load(root/'scores.npy'),np.load(root/'mask.npy')
            assert scores.shape==bits.shape==(len(metadata['episodes']),len(obs))
            assert np.isfinite(scores).all() and np.array_equal(bits,scores>0)
            for frame in np.unique(obs[:,0]):
                indices=np.flatnonzero(obs[:,0]==frame)
                for qi in range(len(bits)):
                    k=int(bits[qi,indices].sum())
                    chosen=indices[np.lexsort((indices,-scores[qi,indices]))][:k]
                    assert set(chosen)==set(indices[bits[qi,indices]])
            self_checks+=1
        for (p,name),seal in seals.items():
            if p!=pair or 'count_method' not in seal: continue
            by={(r['video'],r['expression']):r for r in seals[p,seal['count_method']]['episodes']}
            for row in seal['episodes']:
                assert text_counts(row['prediction_path'])==text_counts(by[row['video'],row['expression']]['prediction_path'])
                count_files+=1
        reports.append(dict(pair=pair,ordinary_checkpoint_sha256=file_sha(path),
            independent_target_maximum_error=target_error,ordinary_steps=fit['steps'],
            independent_CPU_proper_risk=risk,maximum_fit_report_error=report_error,
            observer_table_parameter_change=float(np.abs(observer.table-teacher.table).max()),
            solver_parameter_change=max(float((v-bootstrap['model'][k]).abs().max()) for k,v in final['model'].items()),
            final_CPU_penalized_objective=independent_objective,CPU_GPU_objective_difference=abs(independent_objective-previous),
            selected_pool_prevalence=observer.prevalence,
            class_conditional_expected_Yes=(observer.table@PATTERNS).tolist(),
            teacher_only_posterior_mean=float(target.mean()),joint_solver_mean=float(image.mean()),
            joint_solver_disagrees_with_initial_observer=int(np.count_nonzero((image>.5)!=(target>.5))),
            agreement_or_objective_change_not_semantic_accuracy=True))
    assert rounds==12 and self_checks==30 and not guard
    result=dict(status='PASS_ALL84_SEALS_THREE_NEW_ORDINARY_CPU_RISKS_AND_PHYSICAL_COUNTS',
        records=84,new_adequate_ordinary_heads_recomputed=3,old18_ordinary_risks_reused_by_exact_SHA=True,
        feedback_rounds_checked=rounds,exact_CountSelf_aliases=self_checks,projected_query_files=count_files,
        cases=reports,manual_GT_reads=0,new_fits_teacher_calls_or_MOT_predictions=0,
        source_sha256=file_sha(Path(__file__)))
    atomic_json(OUT/'independent_sealed_prediction_audit.json',result)
    (PUBLIC/(STEM+'_sealed_audit.json')).write_text(json.dumps(result,separators=(',',':'))+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
