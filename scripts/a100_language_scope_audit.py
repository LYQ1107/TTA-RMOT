"""Target-query-only audit of where the current classic cues can operate.

Flags are linguistic diagnostics, not task labels or an accuracy estimate.
This script does not change any running method, threshold, or prediction.
"""
from collections import Counter
import json
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json
from flex_uda.a100.mechanism import generation_guard,metadata
from flex_uda.active.acquisition import expression_types
from flex_uda.active.semantic_colour import appearance_spec
from flex_uda.active.query_consistency import canonical,COLOURS,SUBJECT
from scripts.a100_six_direction import OUT,PUBLIC


def classify(text):
    types=expression_types(text);parsed=canonical(text)
    colours=[a for a in parsed['atoms'] if a.startswith('COLOUR_')]
    appearance=appearance_spec(text)
    # Two subject-like nouns can introduce a reference object. The existing
    # flat pattern parser does not prove which noun owns every later colour.
    # This is a review flag, not a claim that the expression is misparsed.
    nouns=re.findall(r'\b('+SUBJECT+r'|truck|trucks|bus|buses|team|teams)\b',text.lower())
    reference=bool(len(nouns)>1 and re.search(r'\b(behind|beside|near|front of|next to|than|of)\b',text.lower()))
    return dict(types=types,appearance_cue_available=appearance is not None,
        subject_colour_atoms=colours,opaque_qualifier=parsed['opaque'],supported=parsed['supported'],
        compositional_colour_not_in_current_cue=bool(colours and parsed['opaque'] and appearance is None),
        reference_scope_review=reference,
        negation_review=bool(re.search(r'\b(not|no|without|neither|except)\b',text.lower())))


def main():
    guard=generation_guard();targets={}
    for target in ('Dance','KITTI-V2','SportsMOT'):
        queries=metadata(target)['episodes'];counts=Counter();examples={};unique={}
        for query in queries:
            text=query['query_text'];row=classify(text);unique[text]=row
            counts['queries']+=1
            for kind in row['types']:counts['type_'+kind]+=1
            for key in ('appearance_cue_available','supported','compositional_colour_not_in_current_cue',
                        'reference_scope_review','negation_review'):
                counts[key]+=int(row[key])
            for key in ('compositional_colour_not_in_current_cue','reference_scope_review','negation_review'):
                if row[key]:examples.setdefault(key,{})[text]=row
        unique_counts=Counter(queries=len(unique))
        for row in unique.values():
            for key in ('appearance_cue_available','supported','compositional_colour_not_in_current_cue',
                        'reference_scope_review','negation_review'):unique_counts[key]+=int(row[key])
        targets[target]=dict(counts=dict(counts),unique_text_counts=dict(unique_counts),
            examples={key:[dict(text=text,**values) for text,values in sorted(rows.items())[:8]] for key,rows in examples.items()})
    assert not guard
    result=dict(status='TARGET_QUERY_ONLY_CLASSIC_CUE_SCOPE_AUDIT',targets=targets,GT_open_attempts=0,
        teacher_encoder_or_GPU_calls=0,running_numeric_helpers_or_predictions_changed=False,
        meaning='Eligibility and grammar review flags only. They are not semantic GT, coverage of real targets, or proof that an additional cue will help.',
        next_gate='Review completed matched-method outcomes before designing new modules; never apply a car-reference colour to the subject without resolving scope.',
        current_rule_limit='Appearance part_mask deliberately excludes spatial/motion descriptions. This audit tests whether compositional queries fall outside it, not whether the rule is wrong.',
        goal_status='ACTIVE')
    atomic_json(OUT/'language_scope_audit.json',result)
    (PUBLIC/'language_scope_audit.json').write_text(json.dumps(result,ensure_ascii=False,separators=(',',':'))+'\n')
    print(json.dumps(dict(status=result['status'],targets={t:r['counts'] for t,r in targets.items()},GT_open_attempts=0)),flush=True)


if __name__=='__main__':main()
