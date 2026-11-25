"""Teacher-only KITTI pilot: is relational state identifiable from paid QA?

No GT or model outputs are read. In particular CROP already includes position
in its prompt: do not claim that this actual teacher input has no coordinates.
"""
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image
from scipy.special import logit
from flex_tta.runtime.pf_receipts import atomic_json
from flex_uda.a100.mechanism import generation_guard,metadata,history
from flex_uda.active.acquisition import expression_types
from scripts.a100_language_scope_audit import classify
from scripts.a100_six_direction import OUT,PUBLIC


def horizontal_query(text):
    low=text.lower();sides=set(re.findall(r'\b(left|right)\b',low))
    if len(sides)!=1 or 'MOTION' in expression_types(low):return None
    if re.search(r'\b(turn\w*|fac\w*|heading|direction|opposite|towards?|front|behind)\b',low):return None
    if re.search(r'\b(left|right)\s+(?:hand\s+|side\s+)?of\b',low):return None
    if classify(text)['reference_scope_review']:return None
    return 1 if next(iter(sides))=='right' else -1


def main():
    guard=generation_guard();m=metadata('KITTI-V2');results={};details=[]
    for pair in ('C2','C5'):
        total=Counter();groups=[];view=Counter();max_coordinate_error=0.;covariance=variance=0.
        for video,bank in sorted(m['videos'].items()):
            with Image.open(Path(bank['image_root'])/'000000.png') as image:width,height=image.size
            queries={q['expression_id']:q for q in m['episodes'] if q['video_id']==video}
            eligible={k:horizontal_query(q['query_text']) for k,q in queries.items()}
            total['queries']+=len(queries);total['literal_horizontal_eligible_queries']+=sum(v is not None for v in eligible.values())
            records=json.loads((history(pair)/'videos'/video/'questions.json').read_text());paid=defaultdict(list)
            for row in records:
                total['all_actually_requested_questions']+=1
                prompt=row['input']['prompt'];position=re.search(r'normalized position.*?<\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*>',prompt)
                if row['effective_view']=='CROP':
                    total['CROP_questions']+=1;total['CROP_with_explicit_position_in_prompt']+=position is not None
                direction=eligible[row['query_id']]
                if direction is None or row['effective_view'] not in ('CROP','FRAME_BOX'):continue
                key=hashlib.sha256(json.dumps(row['input'],sort_keys=True,separators=(',',':')).encode()).hexdigest();paid[key].append(row)
            local=defaultdict(list)
            for duplicates in paid.values():
                row=duplicates[0];p=float(np.mean([r['yes_probability'] for r in duplicates]));direction=eligible[row['query_id']]
                b=row['input']['bbox'];x=(float(b[0])+float(b[2]))/(2*width)
                prompt=row['input']['prompt'];position=re.search(r'normalized position.*?<\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*>',prompt)
                if position:max_coordinate_error=max(max_coordinate_error,abs(x-float(position.group(1))))
                oriented=x if direction>0 else 1-x
                local[row['query_id'],int(row['track_id']),row['effective_view']].append((int(row['frame_id']),oriented,p))
                total['eligible_unique_requested_inputs']+=1;total['exact_input_records_deduplicated']+=len(duplicates)-1
                view[row['effective_view']]+=1
            for (query,track,kind),values in local.items():
                total['eligible_query_track_view_groups']+=1
                if len({r[0] for r in values})<2:continue
                frame,x,p=map(np.asarray,zip(*values));z=logit(np.clip(p,.02,.98));dx=x-x.mean();dz=z-z.mean()
                gram=float(dx@dx);cross=float(dx@dz)
                row=dict(pair=pair,video=video,query=query,track=track,view=kind,unique_inputs=len(values),
                    distinct_frames=len(set(frame)),x_span=float(np.ptp(x)),within_Gram=gram,within_teacher_logit_cross=cross,
                    binary_teacher_changes=bool((p>.5).any() and (p<=.5).any()))
                groups.append(row);total['groups_with_multiple_observed_frames']+=1
                if gram>1e-12:
                    total['groups_with_identifiable_position_variation']+=1;variance+=gram;covariance+=cross
                    total['excited_groups_with_positive_oriented_covariance']+=cross>0
                    total['excited_groups_with_binary_teacher_changes']+=row['binary_teacher_changes']
        details.extend(groups)
        results[pair]=dict(counts=dict(total),eligible_unique_inputs_by_view=dict(view),within_Gram_sum=variance,
            within_teacher_logit_cross_sum=covariance,pooled_unconstrained_within_slope=covariance/variance if variance else None,
            CROP_prompt_vs_box_normalized_x_max_error=max_coordinate_error,
            interpretation='Excitation and teacher-response covariation only, not semantic correctness or a fitted/passed new module.')
    assert not guard
    result=dict(status='TEACHER_ONLY_RELATIONAL_OBSERVABILITY_PILOT',pairs=results,GT_open_attempts=0,
        new_teacher_encoder_head_or_GPU_calls=0,source='Only actually requested legacy questions; raw input geometry/probabilities, no old posteriors, predictions, identity_vote or unasked cache.',
        critical_input_fact='Actual CROP prompts include normalized position. Cropped image alone and actual image+prompt information are different; do not claim absolute position unobservable in CROP.',
        scope='KITTI C2/C5 pilot, not all-six-setting validation. Conservative literal left/right eligibility is a grammar hypothesis. Within-view grouping prevents view biases being treated as state changes.',
        no_parameters_or_predictions_generated=True,no_GT_for_question_eligibility=True,all_data_development=True,goal_status='ACTIVE')
    atomic_json(OUT/'teacher_relation_observability/full.json',dict(summary=result,groups=details))
    atomic_json(OUT/'teacher_relation_observability/completion.json',result)
    (PUBLIC/'teacher_relation_observability.json').write_text(json.dumps(result,ensure_ascii=False,separators=(',',':'))+'\n')
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
