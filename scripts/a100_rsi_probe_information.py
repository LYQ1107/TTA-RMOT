"""Explain zero RSI rewards using completed, already-paid probes only.

No new question selection, fitting, predictions, or target GT. The relative
odds diagnostic is an assumption review, never evidence of semantic truth.
"""
from collections import Counter,defaultdict
import json
import math
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from flex_tta.runtime.pf_receipts import atomic_json

SIX=ROOT/'a100_runs/visual_classical_20261121/six_direction_20261122'
PUBLIC=ROOT/'docs/a100_six_direction_20261122'
SCOPE={'native_binary_quality_20261124':('dancetrack0004','dancetrack0005','dancetrack0007'),
       'native_binary_trust_quality_20261124':('dancetrack0004','dancetrack0005')}


def binary_pair_information(a,b):
    mass=a*(1-b)+b*(1-a)
    relative=a*(1-b)/mass if mass else .5
    entropy=-relative*math.log(max(relative,1e-300))-(1-relative)*math.log(max(1-relative,1e-300))
    information=max(math.log(2.)-entropy,0.)
    return mass,relative,information


def main():
    rows=[];cases=Counter();distinct_pairs={};max_relative_shift_error=0.;max_information_error=0.
    for cohort,videos in SCOPE.items():
        for pair in ('C1','C6'):
            for video in videos:
                policies={}
                for policy in ('LEARNED','FROZEN'):
                    local=SIX/cohort/pair/'seed0/videos'/video/'policies'/policy
                    assert(local/'closed.pt').is_file(),('Incomplete policy, do not inspect',cohort,pair,video,policy)
                    report=json.loads((local/'receipt.json').read_text())
                    paid={r['uid']:r for r in json.loads((local/'paid.json').read_text())['paid']}
                    sequence=json.loads((local/'sequence.json').read_text())['sequence']
                    policies[policy]=[r['uid'] for r in sequence]
                    for stage in range(3):
                        probe=[paid[r['uid']] for r in sequence if r['stage']==stage and r['role']=='probe']
                        groups=defaultdict(list);mismatch=0;different_frame=0;boundaries=0;shifts=[]
                        for a,b in zip(probe[::2],probe[1::2]):
                            assert a['local_query']==b['local_query'] and a['case']['view']==b['case']['view']
                            p,q=a['probability'],b['probability'];mass,relative,I=binary_pair_information(p,q)
                            groups[a['local_query']].append((mass,mass*I,I))
                            pair_key=tuple(sorted((a['uid'],b['uid'])))
                            distinct_pairs[pair_key]=dict(mass=mass,weighted_information=mass*I,
                                relative_information=I,both_teacher_opinions_below_half=p<.5 and q<.5)
                            mismatch+=a['effective_view']!=b['effective_view']
                            different_frame+=a['case']['frame_id']!=b['case']['frame_id']
                            if 0<p<1 and 0<q<1:
                                for delta in (-2.,2.):
                                    pp=1/(1+math.exp(-(math.log(p)-math.log1p(-p)+delta)))
                                    qq=1/(1+math.exp(-(math.log(q)-math.log1p(-q)+delta)))
                                    new_mass,new_relative,new_I=binary_pair_information(pp,qq)
                                    max_relative_shift_error=max(max_relative_shift_error,abs(relative-new_relative))
                                    shifts.append(dict(shared_logit_shift=delta,weighted_information=new_mass*new_I))
                            else:boundaries+=1
                        if groups:
                            mass,weighted_I,relative_I=np.mean([np.mean(v,axis=0) for v in groups.values()],axis=0)
                        else:mass=weighted_I=relative_I=0.
                        credit=report['rounds'][stage]['credit'];reward=credit['reward']
                        if credit['information'] is not None:max_information_error=max(max_information_error,abs(weighted_I-credit['information']))
                        if not groups:category='NO_PROBE'
                        elif weighted_I<.005:category='ZERO_BY_INFORMATION_RESOLUTION';assert reward==0.
                        elif reward>0:category='POSITIVE_SKILL_PROGRESS'
                        elif reward<0:category='NEGATIVE_SKILL_PROGRESS'
                        elif credit['skill_before']==credit['skill_after']==0:category='ZERO_BOTH_SKILLS_BELOW_CONSTANT'
                        else:category='ZERO_OTHER'
                        cases[category]+=1
                        rows.append(dict(cohort=cohort,pair=pair,video=video,policy=policy,stage=stage,
                            reward=reward,category=category,conditional_exactly_one_mass=float(mass),
                            current_weighted_information=float(weighted_I),unweighted_relative_information=float(relative_I),
                            relative_information_exceeds_resolution_but_absolute_does_not=bool(weighted_I<.005<=relative_I),
                            teacher_yes_at_half=sum(r['probability']>=.5 for r in probe),paid_probe_nodes=len(probe),
                            effective_view_mismatches=mismatch,different_frame_pairs=different_frame,
                            boundary_probability_pairs_not_shifted=boundaries,
                            shifted_pair_weighted_information=shifts,
                            actual_actor_parameter_change=report['rounds'][stage]['actor_update']['actor_parameter_L2_change']))
                learned=set(policies['LEARNED']);frozen=set(policies['FROZEN'])
                for row in rows:
                    if(row['cohort'],row['pair'],row['video'])==(cohort,pair,video):
                        row['replaced_questions_vs_FROZEN']=len(learned^frozen)//2
                        row['complete_question_order_matches_FROZEN']=policies['LEARNED']==policies['FROZEN']
    assert max_information_error<1e-12 and max_relative_shift_error<1e-7
    summary=dict(policy_rounds=len(rows),categories=dict(cases),distinct_paid_probe_pair_UIDs=len(distinct_pairs),
        distinct_pairs_below_weighted_information_resolution=sum(v['weighted_information']<.005 for v in distinct_pairs.values()),
        distinct_low_absolute_but_nontrivial_relative_pairs=sum(v['weighted_information']<.005<=v['relative_information'] for v in distinct_pairs.values()),
        distinct_both_teacher_opinions_below_half=sum(v['both_teacher_opinions_below_half'] for v in distinct_pairs.values()),
        low_absolute_but_nontrivial_relative_information_rounds=sum(r['relative_information_exceeds_resolution_but_absolute_does_not'] for r in rows),
        teacher_yes_opinions_at_half=sum(r['teacher_yes_at_half'] for r in rows),paid_probe_node_occurrences=sum(r['paid_probe_nodes'] for r in rows),
        effective_view_mismatch_occurrences=sum(r['effective_view_mismatches'] for r in rows),
        max_existing_information_reproduction_error=max_information_error,
        shared_teacher_logit_shift_relative_probability_max_error=max_relative_shift_error)
    result=dict(status='COMPLETED_PREFIX_RSI_PROBE_INFORMATION_DIAGNOSIS_NO_GT',scope=SCOPE,
        summary=summary,rounds=rows,new_teacher_calls=0,new_fits=0,manual_GT_or_RMOT_quality_read=False,
        repeated_cohort_source_policy_probes_are_not_independent_samples=True,
        interpretation='Current credit suppresses tiny exactly-one mass as intended. Relative odds can retain teacher contrast under common logit shifts, but amplifying such contrast may amplify rankings among true negatives. Neither implies semantic truth or recommends changing the live protocol.',
        shared_query_view_intercept_is_modelling_assumption_not_verified_teacher_bias=True,
        all_live_sources_and_reward_resolution_unchanged=True,neural_seeds=[0])
    atomic_json(SIX/'rsi_completed_probe_information_20261124.json',result)
    atomic_json(PUBLIC/'rsi_completed_probe_information_20261124.json',result)
    print(json.dumps(dict(status=result['status'],summary=summary)),flush=True)


if __name__=='__main__':main()
