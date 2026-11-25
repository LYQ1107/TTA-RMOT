"""Teacher-only C2/first-video engineering check, not RMOT quality evidence."""
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from scripts.a100_six_direction import OUT as SIX,PUBLIC
from flex_uda.active.deterministic_observation_runtime import configure_before_cuda

OUT = SIX/'joint_observation_real_fit_pilot_20261123'
STATE = OUT/'state.json'


def main():
    plan=json.loads((PUBLIC/'joint_observation_real_fit_pilot_plan.json').read_text())
    for name,sha in plan['source_sha256'].items():
        assert file_sha(ROOT/name) == sha,name
    execution = configure_before_cuda()
    import torch
    from flex_uda.active.joint_observation_inputs import JointObservationVideo,use_verified_local_features
    from flex_uda.active.joint_observation import JointObservationHead
    from flex_uda.active.joint_observation_fit import fit
    torch.set_num_threads(4)
    atomic_json(STATE,dict(status='PREPARING_TEACHER_ONLY_REAL_FIT_PILOT',pid=os.getpid(),goal_status='ACTIVE'))
    cache = use_verified_local_features()
    raw = json.loads((SIX/'joint_observation_teacher_pilot_20261123/responses.json').read_text())
    indices = [i for i,c in enumerate(raw['cases']) if c['pair'] == 'C2']
    assert len(indices) == 8
    v = JointObservationVideo('C2',raw['cases'][indices[0]]['video'])
    qlookup = {q['expression_id']:i for i,q in enumerate(v.queries)}
    tlookup = {int(t):i for i,t in enumerate(v.tracks)}
    by = {(r['case'],r['mode']):r for r in raw['responses']}
    qi=[];ti=[];ra=[];rb=[];targets=[]
    for i in indices:
        case = raw['cases'][i];qi.append(qlookup[case['query_id']]);ti.append(tlookup[case['track_id']])
        ra.append(v.positions[case['frame_A'],case['track_id']]);rb.append(v.positions[case['frame_B'],case['track_id']])
        ab = np.asarray(by[i,'JOINT_AB']['choice_probabilities'])
        ba = np.asarray(by[i,'JOINT_BA']['choice_probabilities'])[[1,0,2,3]]
        p = (ab+ba)/2;targets.append(p/p.sum())
    qi=np.asarray(qi);ti=np.asarray(ti);ra=np.asarray(ra);rb=np.asarray(rb)
    mask = np.broadcast_to(v.present,(len(v.queries),*v.present.shape))
    unlabelled = v.raw_bags[mask].astype('float64')
    mean = unlabelled.mean(0);std = unlabelled.std(0).clip(min=1e-3)
    anchors = ((v.raw_bags[qi,ti]-mean)/std).astype('float32');present=v.present[ti];anchors[~present]=0.
    first = ((v.dense_rows(qi,ra)-mean)/std).astype('float32')
    second = ((v.dense_rows(qi,rb)-mean)/std).astype('float32')
    # Real row alignment for every context anchor used by these examples.
    alignment=0.
    for row,q,t in zip(anchors,qi,ti):
        actual=v.dense_rows(np.full(int(v.present[t].sum()),q,dtype=int),v.sample_rows[t,v.present[t]])
        expected=((actual-mean)/std).astype('float32')
        alignment=max(alignment,float(np.abs(row[v.present[t]]-expected).max()))
    assert alignment == 0.
    data={k:torch.as_tensor(value,device='cuda:0') for k,value in dict(anchors=anchors,present=present,first=first,second=second,target=np.asarray(targets,dtype='float32')).items()}
    results=[]
    for name,configuration,width in [('STRUCTURED',{},64),('GENERIC_JOINT',{'generic':True},128)]:
        torch.manual_seed(0)
        model=JointObservationHead(v.dimension,width=width,**configuration).cuda()
        atomic_json(STATE,dict(status='FITTING_REAL_TEACHER_ONLY_NO_GT',pid=os.getpid(),model=name,completed=len(results),goal_status='ACTIVE'))
        early,receipt=fit(model,data,minimum_steps=1000,maximum_steps=4000,kind='joint')
        torch.save(dict(model=model.state_dict(),early=early,mean=mean,std=std,receipt=receipt),OUT/(name+'.pt'))
        results.append(dict(name=name,width=width,parameters=sum(p.numel() for p in model.parameters()),**receipt))
        print(json.dumps(results[-1]),flush=True)
        del model;torch.cuda.empty_cache()
    report=dict(status='PASS_REAL_TEACHER_ONLY_JOINT_FIT_ENGINEERING_PILOT',pair='C2',video=v.video,targets=8,
        neural_seeds=[0],feature_dimensions=v.dimension,extra_geometry_dimensions=12,models=results,
        source_sha256={n:file_sha(ROOT/n) for n in ['scripts/a100_joint_observation_real_fit_pilot.py','flex_uda/active/joint_observation.py','flex_uda/active/joint_observation_fit.py','flex_uda/active/joint_observation_inputs.py']},
        execution=execution,exact_context_dense_alignment_error=alignment,physical_GPU=os.environ.get('CUDA_VISIBLE_DEVICES'),
        local_feature_cache_aggregate_SHA=cache['aggregate_sha256'],raw_teacher_source='joint_observation_teacher_pilot_20261123',
        no_new_teacher_or_student_backbone_calls=True,GT_open_attempts=0,no_RMOT_predictions_HOTA_or_accuracy=True,
        engineering_fits_not_full_study_prefixes_or_cold_deployment=True,goal_status='ACTIVE')
    atomic_json(OUT/'analysis.json',report);atomic_json(PUBLIC/'joint_observation_real_fit_pilot.json',report)
    atomic_json(STATE,dict(status=report['status'],pid=os.getpid(),goal_status='ACTIVE'))


if __name__ == '__main__':
    try:main()
    except Exception as error:
        atomic_json(STATE,dict(status='FAILED_REAL_JOINT_FIT_PILOT_REQUIRES_ROOT_CAUSE',error=repr(error),goal_status='ACTIVE'))
        raise
