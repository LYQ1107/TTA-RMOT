"""Group exclusion, opposite-model feedback and exact cohort-statistic contract."""
import copy
import json
from pathlib import Path
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from flex_uda.active.crossfit_teacher_feedback import group_folds,outside_predictions,accept_pair
from flex_uda.active.joint_teacher_noise import make_solver
from flex_uda.active.semantic_teacher_patterns import SemanticPatterns
from flex_uda.a100.mechanism import generation_guard
from scripts.a100_crossfit_teacher_primary import OUT,PUBLIC,PLAN,STEM,registration,parent
from scripts.a100_primary_statistics_common import expected_keys,reproduce,write


def main():
    torch.set_num_threads(2);plan=registration();guard=generation_guard()
    reports=[]
    for case,registered in zip(plan['cases'],plan['splits']):
        pair=case['pair'];keys=json.loads((parent.parent.OUT/pair/'row_keys.json').read_text())['keys']
        folds=group_folds(keys)
        assert [int((folds==k).sum()) for k in (0,1)]==registered['fold_sizes']
        groups=[{(r[0],int(r[2])) for r,f in zip(keys,folds) if f==k} for k in (0,1)]
        assert not groups[0]&groups[1] and all(groups)
        reverse=group_folds(list(reversed(keys)))
        assert np.array_equal(reverse[::-1],folds),'Split must be independent of row order'
        reports.append(dict(pair=pair,fold_events=registered['fold_sizes'],fold_track_groups=[len(g) for g in groups],
                            directly_supervised_official_track_overlap=0))
    x=np.zeros((6,5508),dtype='float32');folds=np.array([0,1,0,1,0,1],dtype='int8')
    models=[make_solver('cpu'),make_solver('cpu')]
    for i,model in enumerate(models):
        for parameter in model.parameters(): parameter.data.zero_()
        model.net[-1].bias.data.fill_(float(i))
    image=outside_predictions(models,x,folds,'cpu')
    np.testing.assert_allclose(image[folds==0],1/(1+np.exp(-1)),rtol=1e-7)
    np.testing.assert_allclose(image[folds==1],.5,rtol=1e-7)
    for model in models: model.net[-1].bias.data.zero_()
    previous=[copy.deepcopy(model.state_dict()) for model in models]
    for model in models: model.net[-1].bias.data.fill_(10.)
    rejection=accept_pair(models,previous,SemanticPatterns(),np.zeros((6,3)),x,folds,'cpu')
    assert rejection['accepted_fraction']==0. and rejection['parameter_changes']==[0.,0.]
    for model,old in zip(models,previous):
        assert all(torch.equal(v,old[k]) for k,v in model.state_dict().items())
    # Exercise the actual generic reproduction entry with a local synthetic
    # record set; this does not write fake results into the public directory.
    keys=expected_keys(plan);assert len(keys)==93
    synthetic={key:dict(numerator=[1.]*19,denominator=[4.]*19,HOTA=50.,episodes=1) for key in keys}
    comparisons=[dict(pair=c['pair'],comparator=n,raw_delta_HOTA=0.,
                       primary_at_ordinary_count_delta_HOTA=0.,ordinary_at_primary_count_delta_HOTA=0.)
                 for c in plan['cases'] for n in plan['comparators']]
    with tempfile.TemporaryDirectory(prefix='statistic_contract_',dir=OUT) as name:
        temporary=Path(name)
        write(temporary/(STEM+'_registration.json'),plan)
        write(temporary/(STEM+'_statistics.json'),dict(records=synthetic))
        write(temporary/(STEM+'_analysis.json'),dict(comparisons=comparisons,candidate_eligible_for_larger_screen=False))
        reproduce(temporary,STEM,93,90)
    assert not guard
    result=dict(status='PASS_TRACK_GROUP_EXCLUSION_OPPOSITE_MODEL_ROLLBACK_AND93_COHORT_REPRODUCTION_CONTRACT',
                cases=reports,wrong_direction_pair_proposal_restored=True,
                exact_current_registration_filename_verified=True,synthetic_statistic_records=93,
                real_task_HOTA_or_semantic_quality_not_tested=True,
                GT_open_attempts=0,new_neural_fits_teacher_calls_or_RMOT_predictions=0,
                source_sha256=file_sha(Path(__file__)))
    atomic_json(OUT/'contract.json',result);atomic_json(PUBLIC/(STEM+'_contract.json'),result)
    print(json.dumps(result),flush=True)


if __name__=='__main__': main()
