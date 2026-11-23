"""Explain completed I1 results without changing any adaptation output.

Explicitly post-hoc, isolated canonical-membership diagnosis. KITTI uses the
previously order-verified numerical membership cache, not a new physical-ID
oracle. CLIP is checked at its nominal anchor only, not its entire scope.
"""
from collections import Counter
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from flex_uda.a100.mechanism import PAIRS,metadata,OUT as DIAGNOSTIC
from scripts.a100_six_direction import PUBLIC,SEEDS
from scripts.a100_observation_design_collect import OUT as ANSWERS
from scripts.a100_observation_design_quality import OUT as QUALITY,POLICIES,conditions


def main():
    plan=json.loads((PUBLIC/'observation_policy_postseal_diagnosis_plan.json').read_text())
    assert file_sha(Path(__file__))==plan['source_sha256']
    assert json.loads((QUALITY/'all_predictions_sealed_before_GT.json').read_text())['methods']==828
    assert json.loads((QUALITY/'official.json').read_text())['methods']==828
    for condition in conditions():
        root=Path(condition['folder'])
        for folder in (root,root/'countmatched'):
            assert json.loads((folder/'completion.json').read_text())['GT_open_attempts']==0
    canonical={'C1':'C1','C2':'C5','C3':'C4','C4':'C4','C5':'C5','C6':'C1'}
    summaries={}
    for pair,(_,target) in PAIRS.items():
        md=metadata(target);acquisition={p:Counter() for p in POLICIES}
        outputs={str(s):{p+'_'+d:Counter() for p in POLICIES for d in ('FINAL','MEAN','TEACHER_FILL')} for s in SEEDS}
        for video,bank in sorted(md['videos'].items()):
            queries=[q for q in md['episodes'] if q['video_id']==video];obs=np.asarray(bank['observations'],int)
            truth=np.load(DIAGNOSTIC/'offline_GT'/canonical[pair]/video/'labels.npy')>=0
            assert truth.shape==(len(queries),len(obs))
            present=truth.any(1);tracks=np.asarray(sorted(set(obs[:,1])))
            positions={tuple(row):i for i,row in enumerate(obs.tolist())}
            bag_truth=np.stack([truth[:,obs[:,1]==track].any(1) for track in tracks],1)
            for policy in POLICIES:
                doc=json.loads((ANSWERS/'policy_answers'/pair/video/(policy+'_answers.json')).read_text())
                assert doc['only_this_policy_requested_responses'] and doc['GT_open_attempts']==0
                bag=np.zeros(len(queries),bool);anchor=bag.copy();yes=bag.copy();seen=set()
                for r in doc['answers']:
                    if r['request_id'] in seen:continue
                    seen.add(r['request_id']);qi,ti=int(r['local_query']),int(r['local_track'])
                    assert queries[qi]['expression_id']==r['query_id'] and int(tracks[ti])==r['track_id']
                    row=positions[int(r['frame_id']),int(r['track_id'])]
                    bag[qi]|=bag_truth[qi,ti];anchor[qi]|=truth[qi,row]
                    yes[qi]|=truth[qi,row] and r['yes_probability']>=.5
                acquisition[policy].update(queries=len(queries),canonical_recoverable_positive_queries=int(present.sum()),
                    logical_requests=doc['logical_requested'],unique_request_ids=len(seen),
                    correct_provided_bag_asked_queries=int(bag.sum()),correct_nominal_anchor_asked_queries=int(anchor.sum()),
                    correct_nominal_anchor_teacher_Yes_queries=int(yes.sum()))
                groups=dict(no_correct_bag_asked=present&~bag,
                    correct_bag_but_no_correct_nominal_anchor=present&bag&~anchor,
                    correct_nominal_anchor_but_no_teacher_Yes=present&anchor&~yes,
                    correct_nominal_anchor_with_teacher_Yes=present&yes)
                assert np.array_equal(sum(g.astype(int) for g in groups.values()),present.astype(int))
                for seed in SEEDS:
                    for decoder in ('FINAL','MEAN','TEACHER_FILL'):
                        name=policy+'_'+decoder
                        bits=np.load(QUALITY/pair/f'seed{seed}'/name/'videos'/video/'mask.npy')
                        assert bits.shape==truth.shape
                        good=(bits&truth).any(1);zero=present&~bits.any(1);miss=present&~good
                        row=outputs[str(seed)][name]
                        row.update(canonical_recoverable_positive_queries=int(present.sum()),
                            selected_rows=int(bits.sum()),selected_canonical_positive_rows=int((bits&truth).sum()),
                            selected_canonical_negative_rows=int((bits&~truth).sum()),
                            recoverable_positive_zero_output_queries=int(zero.sum()),
                            recoverable_positive_without_any_correct_output_queries=int(miss.sum()),
                            correct_output_without_correct_bag_asked_queries=int((present&good&~bag).sum()),
                            correct_output_without_correct_nominal_anchor_teacher_Yes_queries=int((present&good&~yes).sum()))
                        for label,g in groups.items():
                            row['zero_output__'+label]+=int((zero&g).sum())
                            row['no_correct_output__'+label]+=int((miss&g).sum())
        summaries[pair]=dict(acquisition_once_not_tripled_by_neural_seed={k:dict(v) for k,v in acquisition.items()},
            outputs_per_neural_seed={s:{k:dict(v) for k,v in row.items()} for s,row in outputs.items()})
    result=dict(status='PASS_POSTSEAL_I1_CANONICAL_COVERAGE_AND_USAGE_DIAGNOSIS',settings=summaries,
        interpretation='Canonical candidate membership, not per-method official matching or teacher semantic accuracy. A bag can contain a match away from the asked anchor; CLIP may see other frames.',
        KITTI_scope='Original complete-bank numerical membership, observations/query order previously exactly verified; no new physical-ID or one-to-one oracle claim.',
        missing_canonical_candidate_not_called_scene_absence=True,
        source_study_all828_outputs_sealed_before_label_reads=True,
        explicitly_posthoc_after_I1_HOTA_available=True,no_prediction_head_or_question_changed=True,
        new_model_or_teacher_calls=0,no_target_GT_consumed_by_adaptation=True,
        descriptive_counts_not_independent_samples_or_new_HOTA_error_decomposition=True,goal_status='ACTIVE')
    atomic_json(PUBLIC/'observation_policy_postseal_diagnosis.json',result)
    print(json.dumps(dict(status=result['status'],settings=len(summaries),new_fits=0,changed_predictions=0)))


if __name__=='__main__':main()
