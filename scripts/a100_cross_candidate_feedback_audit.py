"""Audit question-channel shifts, without using manual semantic labels."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from scripts.a100_cross_candidate_teacher_pilot import OUT,PLAN,PUBLIC


def main():
    data=json.loads((OUT/'responses.json').read_text())
    by={(r['case'],r['mode']):r for r in data['responses']}
    rows=[]
    for ci,c in enumerate(data['cases']):
        ab=np.asarray(by[ci,'JOINT_AB']['choice_probabilities'])
        ba=np.asarray(by[ci,'JOINT_BA']['choice_probabilities'])
        q=(ab+ba[[1,0,2,3]])/2
        independent=[by[ci,'INDEPENDENT_'+s]['yes_probability_fp32'] for s in ('A','B')]
        shared=[by[ci,'SHARED_'+s]['yes_probability_fp32'] for s in ('A','B')]
        odds_ab=float(np.log(max(ab[0],1e-300)/max(ab[1],1e-300)))
        odds_ba=float(np.log(max(ba[0],1e-300)/max(ba[1],1e-300)))
        rows.append(dict(target=c['target'],view=c['view'],independent_yes=independent,
            shared_yes=shared,joint_marginal_yes=[float(q[0]+q[2]),float(q[1]+q[2])],
            canonical_object_log_odds=(odds_ab-odds_ba)/2,
            display_position_log_odds=(odds_ab+odds_ba)/2,
            joint_top1_choice=[by[ci,'JOINT_'+s]['top1_is_choice'] for s in ('AB','BA')],
            allowed_choice_mass=[by[ci,'JOINT_'+s]['allowed_choice_mass'] for s in ('AB','BA')]))
    def summary(part):
        native=np.asarray([r['independent_yes'] for r in part])
        joint=np.asarray([r['joint_marginal_yes'] for r in part])
        shared=np.asarray([r['shared_yes'] for r in part])
        return dict(cases=len(part),mean_independent_yes=float(native.mean()),
            mean_shared_yes=float(shared.mean()),mean_joint_marginal_yes=float(joint.mean()),
            independent_minus_joint_mean=float((native-joint).mean()),
            mean_absolute_independent_candidate_gap=float(abs(native[:,0]-native[:,1]).mean()),
            mean_absolute_joint_candidate_gap=float(abs(joint[:,0]-joint[:,1]).mean()),
            mean_absolute_object_log_odds=float(np.mean([abs(r['canonical_object_log_odds']) for r in part])),
            mean_absolute_display_position_log_odds=float(np.mean([abs(r['display_position_log_odds']) for r in part])))
    joint_rows=[r for r in data['responses'] if r['mode'].startswith('JOINT')]
    report=dict(status='AUDITED_REAL_CROSS_CANDIDATE_QUESTION_CHANNEL_SHIFT',
        overall=summary(rows),by_target={t:summary([r for r in rows if r['target']==t]) for t in sorted({r['target'] for r in rows})},
        all_joint_top1_is_allowed_choice=all(r['top1_is_choice'] for r in joint_rows),
        allowed_choice_mass_quantiles=np.quantile([r['allowed_choice_mass'] for r in joint_rows],[0,.25,.5,.75,1]).tolist(),
        manual_GT_teacher_accuracy_or_HOTA_not_measured=True,
        teacher_channel_shift_does_not_identify_which_answer_is_correct=True,
        no_new_teacher_training_models_or_task_outputs=True,
        implications=['Four-way formatting succeeded but semantic probability consistency is not established.',
            'Do not equate candidate-relative joint querying with better supervision.',
            'Keep native single-candidate binary teaching as a strong control.',
            'A factorized pair target built from binary outputs is an explicit independence model, never a measured joint teacher answer.'],
        teacher_plan_sha256=file_sha(PLAN),audit_source_sha256=file_sha(Path(__file__)))
    atomic_json(OUT/'question_channel_audit.json',dict(report=report,per_case=rows))
    atomic_json(PUBLIC/'cross_candidate_question_channel_audit.json',report)
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
