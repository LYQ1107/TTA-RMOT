"""Teacher-only class slices for a completed condition's current-video fits.

The registered gate is a prefix-average KL. Passing it does not certify a
uniform fit to rare teacher-positive observations. This reads original
current-video scores, never applies a future head backwards or opens GT.
"""
from collections import defaultdict
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from flex_uda.a100.mechanism import PAIRS,metadata,generation_guard
from scripts.a100_six_direction import PUBLIC
from scripts.a100_conditional_six_setting import OUT as QUALITY
from scripts.a100_observation_design_collect import OUT as ANSWERS

ARMS=('FRAME_SOFT','FRAME_HARD','FLAT_SOFT','FLAT_HARD','DIRECT_POINT',
      'DIRECT_BALANCED','DIRECT_PROFILE','CONDITIONAL_BALANCED','FULL')


def run(pair,seed):
    plan=json.loads((PUBLIC/'conditional_teacher_fit_slices_plan.json').read_text())
    assert file_sha(Path(__file__))==plan['source_sha256']
    guard=generation_guard();root=QUALITY/pair/f'seed{seed}'
    completion=json.loads((root/'completion.json').read_text());assert completion['GT_open_attempts']==0
    md=metadata(PAIRS[pair][1]);parts=defaultdict(list);prefix_counts={a:[] for a in ARMS}
    for video,bank in sorted(md['videos'].items()):
        queries=[q for q in md['episodes'] if q['video_id']==video]
        observations=np.asarray(bank['observations'],int)
        positions={tuple(o):i for i,o in enumerate(observations.tolist())}
        doc=json.loads((ANSWERS/'policy_answers'/pair/video/'DESIGN_answers.json').read_text())
        assert doc['GT_open_attempts']==0 and doc['only_this_policy_requested_responses']
        unique={};grouped=defaultdict(list)
        for item in doc['answers']:
            rid=item['request_id'];p=float(item['yes_probability'])
            if rid in unique:
                assert unique[rid]==p
                continue
            unique[rid]=p;qi=int(item['local_query'])
            assert queries[qi]['expression_id']==item['query_id']
            row=positions[int(item['frame_id']),int(item['track_id'])]
            grouped[qi,row].append(p)
        keys=sorted(grouped);qi,rows=np.asarray(keys,int).T
        raw=np.asarray([np.mean(grouped[key]) for key in keys],dtype='float32')
        targets=np.clip(raw,.02,.98);hard=raw>=.5
        parts['raw'].append(raw);parts['soft'].append(targets)
        for arm in ARMS:
            path=root/'videos'/video/(arm+'_FINAL_scores.npy')
            scores=np.load(path,mmap_mode='r');assert scores.shape==(len(queries),len(observations))
            p=np.asarray(scores[qi,rows],dtype='float32');assert np.isfinite(p).all()
            parts[arm].append(p)
            prefix_counts[arm].append(dict(video=video,teacher_examples=len(raw),
                teacher_positive=int((raw>.5).sum()),teacher_negative=int((raw<.5).sum()),teacher_exact_tie=int((raw==.5).sum()),
                teacher_positive_predicted_No=int(((raw>.5)&(p<=.5)).sum()),
                teacher_negative_predicted_Yes=int(((raw<.5)&(p>.5)).sum()),
                high_confidence_teacher_positive_predicted_No=int(((raw>=.9)&(p<=.5)).sum()),
                high_confidence_teacher_negative_predicted_Yes=int(((raw<=.1)&(p>.5)).sum())))
    raw=np.concatenate(parts['raw']);soft=np.concatenate(parts['soft']);positive=raw>=.5
    result={}
    groups=dict(all=np.ones(len(raw),bool),teacher_positive=raw>.5,teacher_negative=raw<.5,teacher_exact_tie=raw==.5,
        high_confidence_teacher_positive=raw>=.9,high_confidence_teacher_negative=raw<=.1)
    for arm in ARMS:
        p=np.concatenate(parts[arm]);target=positive.astype(float) if arm.endswith('HARD') else soft
        row={}
        for name,mask in groups.items():
            n=int(mask.sum());defined=mask&(raw!=.5);errors=((p>.5)!=positive)&defined
            row[name]=dict(observations=n,teacher_sign_disagreements=int(errors.sum()),
                non_tie_observations=int(defined.sum()),
                teacher_sign_disagreement_rate=float(errors.sum()/defined.sum()) if defined.any() else None,
                own_target_probability_MSE=float(np.square(p[mask].astype(float)-target[mask]).mean()) if n else None,
                mean_prediction=float(p[mask].mean()) if n else None)
        result[arm]=dict(slices=row,current_video_counts=prefix_counts[arm])
    assert not guard
    report=dict(status='PASS_CURRENT_VIDEO_TEACHER_FIT_SLICE_DIAGNOSIS_NO_GT',pair=pair,seed=seed,
        policy='DESIGN',teacher_anchor_targets=len(raw),models=result,
        source_scope='Own completed current-video predictions, then each video own paid frame/view-mixture targets. This is not replaying a final head on earlier videos.',
        minority_fit_is_not_certified_by_a_global_KL_gate=True,original_fit_gates_not_redefined=True,
        exact_teacher_half_probability_not_called_a_sign_error=True,hard_training_ties_still_use_original_greater_equal_rule=True,
        soft_and_hard_own_targets_distinguished=True,teacher_sign_not_semantic_GT=True,
        MSE_avoids_recovering_unavailable_logits_from_rounded_probability_files=True,
        no_GT_HOTA_or_new_prediction_or_model_or_question_change=True,manual_GT_open_attempts=0,
        new_training_or_teacher_or_backbone_forward_calls=0,goal_status='ACTIVE')
    path=PUBLIC/(f'conditional_teacher_fit_slices_{pair}_seed{seed}.json');atomic_json(path,report)
    print(json.dumps(dict(status=report['status'],pair=pair,teacher_anchor_targets=len(raw),
        models={a:r['slices'] for a,r in result.items()})))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pair',choices=tuple(PAIRS),required=True);p.add_argument('--seed',type=int,default=0)
    args=p.parse_args();run(args.pair,args.seed)
