"""Descriptive paired input effects, separated from semantic correctness.

The five prompt controls were fixed before responses. These contrasts do not
choose a new primary, use GT, fit a model, or claim metadata improves accuracy.
"""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from flex_tta.runtime.pf_receipts import atomic_json
from flex_uda.a100.mechanism import generation_guard
from scripts.a100_query_observability_audit import explicit_location_cue
from scripts.a100_state_teacher_input_pilot import OUT,SUCCESS,question_key
from scripts.a100_six_direction import PUBLIC

CONTRASTS={'CURRENT_REFERENCE_AND_WORDING':('LAST','ORIGINAL'),
    'POSITION_GIVEN_LAST':('POSITION','LAST'),'TIME_GIVEN_LAST':('TIME','LAST'),
    'STATE_GIVEN_LAST':('STATE','LAST'),'POSITION_GIVEN_TIME':('STATE','TIME'),
    'TIME_GIVEN_POSITION':('STATE','POSITION')}


def main():
    guard=generation_guard();complete=json.loads((OUT/'completion.json').read_text());assert complete['status']==SUCCESS
    rows=json.loads((OUT/'answer_checkpoint.json').read_text())['variants'];base=rows['ORIGINAL']
    explicit=np.array([explicit_location_cue(r['query_text']) for r in base]);subsets={'ALL':np.ones(len(base),bool),'EXPLICIT_LOCATION_MOTION':explicit,'OTHER_MOTION':~explicit}
    results={}
    for subset,mask in subsets.items():
        result={}
        for name,(a,b) in CONTRASTS.items():
            assert all(question_key(x)==question_key(y) and x['input']['visual']==y['input']['visual'] for x,y in zip(rows[a],rows[b]))
            x=np.array([r['yes_probability'] for r in rows[a]])[mask];y=np.array([r['yes_probability'] for r in rows[b]])[mask]
            result[name]=dict(questions=len(x),changed_binary_answers=int(((x>.5)!=(y>.5)).sum()),
                yes_to_no=int(((y>.5)&(x<=.5)).sum()),no_to_yes=int(((y<=.5)&(x>.5)).sum()),
                mean_absolute_probability_change=float(np.abs(x-y).mean()),mean_signed_probability_change=float((x-y).mean()))
        results[subset]=result
    assert not guard
    report=dict(status='PAIRED_STATE_INPUT_INCREMENTAL_SENSITIVITY_ANALYZED',subsets=results,
        manual_GT_open_attempts=0,new_teacher_GPU_or_adaptation_calls=0,primary_remains_STATE=True,
        interpretation='LAST alone changes5/12 answers; metadata conditional on LAST changes1/12. Most observed binary change cannot be attributed specifically to added geometry/time. All observed flips are Yes-to-No, which may help or hurt; no accuracy conclusion.',
        unresolved='Semantic use of numeric metadata versus a verbose-prompt/negative-response effect needs a matched placebo or task-compatible counterfactual control before a metadata contribution claim.',
        cost='Fixed order and shared warm image/model cache: do not call later-variant service speed an intrinsic efficiency advantage.',
        all_data_development=True,no_CI_from12_correlated_questions=True,no_posthoc_primary_promotion=True,goal_status='ACTIVE')
    atomic_json(OUT/'incremental_analysis.json',report)
    (PUBLIC/'state_teacher_input_incremental_analysis.json').write_text(json.dumps(report,separators=(',',':'))+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
