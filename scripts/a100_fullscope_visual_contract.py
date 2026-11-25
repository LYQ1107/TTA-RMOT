"""Check new fullscope input/remaining-support/LURE contracts, no task GT."""
import json
from pathlib import Path
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from flex_uda.active.fullscope_visual_runtime import Context,TeacherDispatcher,proposal,lure_weights,predicted_student_risk
from scripts.a100_evidence_visual_primary import registration as old_registration
from scripts.a100_full_temporal_support import OUT as SUPPORT


def main():
    from flex_uda.a100.mechanism import generation_guard
    guard=generation_guard();torch.set_num_threads(2)
    parent=old_registration();cases=[]
    with tempfile.TemporaryDirectory(prefix='fullscope-contract-',dir='/var/tmp') as temp:
        for old in parent['cases']:
            pair,video=old['pair'],old['evaluated_video'];case=dict(old,temporal_queries=str(SUPPORT/pair/video/'queries.json'))
            context=Context(case,Path(temp)/pair,TeacherDispatcher(ROOT,Path(temp)/'never_opened.sqlite'),'CPU_NO_TEACHER')
            doc=json.loads((Path(old['pool_folder'])/'actions.json').read_text());physical={tuple(map(int,row)):i for i,row in enumerate(context.video.obs)}
            old_features=np.load(Path(old['pool_folder'])/'features.npy',mmap_mode='r')
            rows=[]
            for uid in doc['order']:
                a=doc['actions'][uid];r=a['case'];rows.append(a['local_query']*context.n+physical[int(r['frame_id']),int(r['track_id'])])
            for begin in range(0,len(rows),512):assert np.array_equal(context.features[rows[begin:begin+512]],old_features[begin:begin+512])
            selected=np.linspace(0,context.n-1,min(16,context.n)).astype(int)
            first=context.features[selected][:,7815:8583]
            last=context.features[(context.nq-1)*context.n+selected][:,7815:8583]
            assert np.array_equal(first,last) and np.array_equal(first,context.images[selected])
            assert context.dispatcher.client is None
            cases.append(dict(pair=pair,old_point_rows_byte_equal=len(rows),all_ROIs=len(context.images),
                old_alignment_ROIs=len({(a['case']['track_id'],a['case']['frame_id']) for a in doc['actions'].values()}),
                full_query_actions=len(context.features),unlabeled_sample_normalization=True,
                query_independent_content_reduction_exact_on_real_rows=True,teacher_calls=0))
    # Exact adaptive WITHOUT-replacement enumeration for new remaining-support
    # proposal, including track-coverage penalty. No neural seeds or GT.
    n,budget=5,3;loss=np.asarray([.1,.8,1.2,.3,2.1]);columns=np.asarray([0,0,1,2,2])
    total_mass=expected=0.;paths=0
    def walk(known,indices,probabilities,mass):
        nonlocal total_mass,expected,paths
        if len(indices)==budget:
            weight=lure_weights(n,probabilities)
            expected+=mass*float((weight*loss[indices]).mean());total_mass+=mass;paths+=1;return
        priority=.1+np.arange(n)+sum(loss[indices])*.2*np.arange(n)[::-1]
        pi=proposal(priority,known,columns)
        assert (pi[known]==0).all() and (pi[~known]>=.2/(~known).sum()-1e-15).all()
        for i in np.flatnonzero(~known):
            next_known=known.copy();next_known[i]=True
            walk(next_known,indices+[int(i)],probabilities+[float(pi[i])],mass*pi[i])
    walk(np.zeros(n,dtype=bool),[],[],1.)
    error=abs(expected-loss.mean());assert abs(total_mass-1)<1e-12 and error<1e-12
    response=np.asarray([.05,.3,.9]);mu=response.mean();var=np.square(response-mu).mean()
    decomposition=max(abs(predicted_student_risk(mu,var,p)-np.square(response-p).mean()) for p in (.1,.5,.9))
    assert decomposition<1e-14
    assert not guard
    result=dict(status='PASS_FULL_SCOPE_INPUT_AND_SAMPLING_CONTRACT_NO_GT',cases=cases,
        exact_adaptive_no_replacement_paths=paths,expected_LURE_error=error,bias_variance_identity_error=float(decomposition),
        full_image_transfer_reduction='Query-independent content objective and batchorder can averagephysicalROIs once; fullsemanticqueryscores are never equated.',
        no_teacher_model_backbone_calls_optimizer_steps_predictions_or_GT=True,
        fixed_model_risk_identity_not_a_fitted_model_or_HOTA_guarantee=True,
        LURE_is_credited_prior_work_not_new_theorem=True,self_source_sha256=file_sha(__file__),goal_status='ACTIVE')
    atomic_json(ROOT/'a100_runs/visual_classical_20261121/supervisor/fullscope_visual_contract_20261126.json',result)
    atomic_json(ROOT/'docs/a100_six_direction_20261122/fullscope_visual_contract_20261126.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
