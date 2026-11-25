"""Postseal failure analysis, without new fits or per-item training targets.

Full frozen-response library and canonical GT are opened only after all66
outputs, official evaluations and independent reproduction have completed.
Canonical membership includes box matching/annotation effects and is not
pure teacher semantic truth. Nothing here is fed to adaptation.
"""
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from scripts.a100_residual_rsi_primary import OUT,PUBLIC,STEM,PRIMARY,METHODS,registration


def confusion(truth,decision):
    return dict(TP=int((truth&decision).sum()),FP=int((~truth&decision).sum()),
                FN=int((truth&~decision).sum()),TN=int((~truth&~decision).sum()))


def official(path):
    with path.open() as stream:rows=[r for r in csv.DictReader(stream) if r['seq']!='COMBINED']
    return {key:sum(float(r['HOTA_'+key+'___50']) for r in rows) for key in ('TP','FP','FN')}


def main():
    plan=registration()
    barrier=json.loads((OUT/'all66_sealed_before_GT.json').read_text())
    rep=json.loads((PUBLIC/(STEM+'_reproduction.json')).read_text())
    assert len(barrier['tasks'])==66 and barrier['GT_open_attempts']==0
    assert rep['records']==66 and rep['effects']==63 and rep['status'].startswith('PASS_')
    from flex_uda.a100.mechanism import OUT as DIAGNOSTIC,metadata
    from scripts.a100_observation_evidence_transfer_diagnose import CANONICAL
    from flex_uda.active.residual_rsi_learning import data,broker
    reports=[]
    for case in plan['cases']:
        pair,video=case['pair'],case['evaluated_video'];folder=OUT/pair
        features,query,doc,quotas,auxiliary,ay,_,_=data(case)
        draws=json.loads((folder/'primary/draws.json').read_text())['draws']
        uniform=json.loads((folder/'uniform_draws.json').read_text())['draws']
        # Diagnostic-only reveal, AFTER complete sealed/evaluated predictions.
        b=broker(case,doc,quotas);b._load_private()
        teacher=np.asarray([b._answers[uid]['yes_probability'] for uid in doc['order']])
        md=metadata(case['target']);obs=np.asarray(md['videos'][video]['observations'],dtype=int)
        queries=[q for q in md['episodes'] if q['video_id']==video]
        lookup={tuple(map(int,row)):i for i,row in enumerate(obs)}
        nodes=[]
        for uid in doc['order']:
            action=doc['actions'][uid];r=action['case'];q=action['local_query']
            assert queries[q]['expression_id']==r['query_id']
            nodes.append(q*len(obs)+lookup[int(r['frame_id']),int(r['track_id'])])
        nodes=np.asarray(nodes)
        truth=np.load(DIAGNOSTIC/'offline_GT'/CANONICAL[pair]/video/'labels.npy')>=0
        assert truth.shape==(len(queries),len(obs))
        pool_truth=truth.flatten()[nodes]
        known=np.zeros(len(query),dtype=bool);known[[d['index'] for d in draws]]=True
        uniform_known=np.zeros(len(query),dtype=bool);uniform_known[[d['index'] for d in uniform]]=True
        with np.load(folder/'primary/risk_stage_3.npz') as saved:
            coefficient=saved['risk_coefficients'];weight=saved['population_weights'];proxy=saved['prequential_proxy_mean']
        h=-teacher*np.log(np.clip(teacher,1e-15,1))-(1-teacher)*np.log(np.clip(1-teacher,1e-15,1))
        records={}
        for name in METHODS:
            prediction=folder/'predictions'/name
            score=np.load(prediction/'scores.npy').astype('float64')
            mask=np.load(prediction/'mask.npy');assert score.shape==mask.shape==truth.shape
            z=score.flatten()[nodes];KL=np.logaddexp(0,z)-teacher*z-h
            partitions={}
            for label,select in (('all_pool',np.ones(len(query),bool)),('primary_queried',known),('primary_unqueried',~known)):
                partitions[label]=dict(actions=int(select.sum()),macroquery_weighted_teacher_KL=float(np.sum(weight[select]*KL[select])/max(weight[select].sum(),1e-15)))
            records[name]=dict(official_IoU50_counts=official(prediction/'results/pedestrian_detailed.csv'),
                canonical_membership_counts=confusion(truth,mask),
                positive_queries_zero_output=int((truth.any(1)&~mask.any(1)).sum()),
                full_pool_teacher_fit=partitions,
                teacher_class_agreement_on_primary_queried=int(((z>0)==(teacher>.5))[known].sum()),
                signed_training_risk=float(np.sum(weight*(np.logaddexp(0,z)-coefficient*z))),
                pool_logit_at_cap_count=int((np.abs(z)>=7.999).sum()))
        coverage={}
        for name,asked in (('primary',known),('uniform',uniform_known)):
            coverage[name]=dict(unique_actions=int(asked.sum()),
                positive_pool_queries=int(sum(pool_truth[query==q].any() for q in quotas)),
                positive_pool_queries_never_asked_correct=int(sum(pool_truth[query==q].any() and not (pool_truth&asked)[query==q].any() for q in quotas)),
                positive_queries_outside_declared_action_pool=int(sum(truth[q].any() and not pool_truth[query==q].any() for q in quotas)))
        record=dict(pair=pair,video=video,methods=records,coverage=coverage,
            signed_risk_coefficients=dict(minimum=float(coefficient.min()),maximum=float(coefficient.max()),
                outside_probability_interval=int(((coefficient<0)|(coefficient>1)).sum()),
                queried_coefficient_class_disagrees_with_original_teacher=int((((coefficient>.5)!=(teacher>.5))&known).sum()),
                macroquery_absolute_correction=float(np.sum(weight*np.abs(coefficient-proxy)))),
            original_teacher_canonical_pool_counts=confusion(pool_truth,teacher>.5),
            mean_preanswer_squared_error=float(np.mean([(r['probability']-r['proxy_before'])**2 for r in draws])),
            policy_support='Only declared historical union, not full dense target pool.')
        reports.append(record)
        print(json.dumps(dict(status='POSTSEAL_RESIDUAL_FAILURE_CASE_DIAGNOSED',pair=pair,
            primary=records[PRIMARY],coverage=coverage,signed_coefficients=record['signed_risk_coefficients'])),flush=True)
    result=dict(status='COMPLETE_POSTSEAL_RESIDUAL_RSI_DIAGNOSIS',cases=reports,
        source_sha256=file_sha(__file__),all66_and_reproduction_precede_label_access=True,
        new_models_teacher_or_backbone_calls=0,no_feedback_of_diagnostic_values_to_learners=True,
        canonical_membership_not_pure_semantic_truth=True,
        fixed_model_expected_unbiasedness_not_fitted_model_generalization=True,
        official_IoU50_counts_not_additive_19threshold_HOTA_decomposition=True,
        all_data_DEV=True,goal_status='ACTIVE')
    atomic_json(OUT/'postseal_diagnosis.json',result)
    atomic_json(PUBLIC/(STEM+'_postseal_diagnosis.json'),result)


if __name__=='__main__':main()
