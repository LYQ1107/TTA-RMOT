"""Audit which task cues the actually paid teacher inputs retain.

CLIP sends cropped images and an ordinal temporal prompt; its raw record's
bounding box is not automatically part of the model prompt. This is an input
audit, not a measurement of semantic accuracy or actual GT error attribution.
"""
from collections import Counter,defaultdict
import json
from pathlib import Path
import re
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from flex_uda.a100.mechanism import PAIRS,metadata,history,generation_guard
from flex_uda.active.acquisition import expression_types
from scripts.a100_six_direction import OUT as SIX,PUBLIC

OUT=SIX/'query_observability_audit'


def explicit_location_cue(text):
    """Conservative textual cue only, not an assertion of annotation scope."""
    low=text.lower()
    for match in re.finditer(r'\b(?:on|at)\s+(?:the\s+)?(?:left|right)(?:\s+(?:side|part))?\b',low):
        after=low[match.end():].strip()
        if not re.match(r'^(?:of|lane|road|street|team|hand)\b',after):return True
    return False


def main():
    start=time.perf_counter();guard=generation_guard();registration=json.loads((PUBLIC/'query_observability_audit_registration.json').read_text())
    assert registration['source_sha256']==file_sha(Path(__file__))
    results={}
    for pair,(_,target) in PAIRS.items():
        m=metadata(target);counts=Counter();views=defaultdict(Counter);query_summary=[]
        for video in sorted(m['videos']):
            queries={q['expression_id']:q for q in m['episodes'] if q['video_id']==video}
            tags={qid:set(expression_types(q['query_text'])) for qid,q in queries.items()}
            paid=defaultdict(list)
            records=json.loads((history(pair)/'videos'/video/'questions.json').read_text())
            for row in records:
                qid=row['query_id'];view=row['effective_view'];prompt=row['input']['prompt'];counts['actual_requests']+=1;views['ALL'][view]+=1
                for tag in tags[qid]:views[tag][view]+=1
                visible_coords='normalized position' in prompt.lower()
                if view=='CROP':counts['CROP_requests']+=1;counts['CROP_with_explicit_coordinate_prompt']+=visible_coords
                if view=='CLIP':
                    counts['CLIP_requests']+=1;counts['CLIP_with_explicit_coordinate_prompt']+=visible_coords
                    info=row.get('clip') or {}
                    if 'frames' not in info:counts['CLIP_missing_frame_metadata']+=1
                    else:
                        unique=len(set(info['frames']));counts['CLIP_with_frame_metadata']+=1
                        counts['CLIP_fewer_than_two_unique_frames']+=unique<2
                    if 'MOTION' in tags[qid] and 'SPATIAL' in tags[qid]:counts['CLIP_on_MOTION_SPATIAL_lexical_composite']+=1
                    if 'MOTION' in tags[qid] and explicit_location_cue(queries[qid]['query_text']):counts['CLIP_on_explicit_location_plus_MOTION_cue']+=1
                paid[qid].append((view,visible_coords))
            for qid,q in queries.items():
                motion='MOTION' in tags[qid];spatial='SPATIAL' in tags[qid];explicit=explicit_location_cue(q['query_text'])
                counts['queries']+=1;counts['MOTION_queries']+=motion;counts['MOTION_SPATIAL_lexical_queries']+=motion and spatial
                counts['explicit_location_MOTION_queries']+=motion and explicit
                rows=paid[qid];only_clip=bool(rows) and all(view=='CLIP' for view,_ in rows)
                counts['MOTION_queries_only_effective_CLIP']+=motion and only_clip
                counts['MOTION_SPATIAL_queries_only_effective_CLIP']+=motion and spatial and only_clip
                counts['explicit_location_MOTION_queries_only_effective_CLIP']+=motion and explicit and only_clip
                if motion and explicit and rows:
                    query_summary.append(dict(video=video,query_id=qid,text=q['query_text'],actual_requests=len(rows),
                        effective_views=dict(Counter(v for v,_ in rows)),any_explicit_coordinate_prompt=any(c for _,c in rows)))
        results[pair]=dict(source=PAIRS[pair][0],target=target,counts=dict(counts),
            actual_views_by_overlapping_lexical_type={k:dict(v) for k,v in views.items()},
            illustrative_explicit_location_motion_queries=query_summary[:4])
    assert not guard
    report=dict(status='TEACHER_INPUT_OBSERVABILITY_AUDIT_COMPLETE',pairs=results,manual_GT_open_attempts=0,
        teacher_probabilities_or_old_beliefs_used=False,new_teacher_or_GPU_or_neural_calls=0,new_quality_predictions=0,
        actual_input_fact='CROP prompt explicitly includes normalized centre; CLIP prompt supplies cropped moments, with no normalized-position or frame-index side channel. Record input.bbox and clip.frames are bookkeeping, not automatically model inputs.',
        theoretical_statement='If a renderer gives identical observations on two worlds where a query predicate differs, no predictor using only that observation can be universally correct. For interior integer translations, crop-only CLIP is invariant; absolute image-side predicates need not be. This is a conditional counterexample, not a claim that every actual CLIP query is ambiguous.',
        interpretation='Counts describe requested inputs and lexical cues, not a count of GT errors. Backgrounds can provide implicit context. Keyword SPATIAL includes ambiguous phrases; even the conservative explicit cue does not certify annotation semantics.',
        next='If missing reference-frame/time channels are tested, compare the same frozen teacher with matched input alternatives and give ordinary distillation the same resulting evidence. Do not alter the running pilot or quality protocols.',
        all_data_development=True,wall_seconds=time.perf_counter()-start,goal_status='ACTIVE')
    atomic_json(OUT/'completion.json',report)
    (PUBLIC/'query_observability_audit.json').write_text(json.dumps(report,separators=(',',':'))+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
