"""Which factor comparisons are actually supported by existing observations?

Wait for the fixed text interpretations. Match only identical visual inputs
and identical non-query prompt context from DESIGN's own answers. This is a
teacher-observation audit, not target semantic truth or a new learned method.
"""
from collections import Counter, defaultdict
import itertools
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from flex_tta.runtime.pf_receipts import atomic_json, file_sha
from flex_uda.a100.mechanism import PAIRS, generation_guard, metadata
from flex_uda.active.query_scope import FIELDS
from scripts.a100_query_scope_teacher import OUT as SCOPES, STATE as SCOPE_STATE, SUCCESS as SCOPE_SUCCESS, text_key
from scripts.a100_observation_design_collect import OUT as ANSWERS, TEACHER_SHA
from scripts.a100_six_direction import OUT as SIX, PUBLIC

OUT = SIX/'scoped_supervision_audit_20261123_v3'
STATE = OUT/'supervisor/state.json'
REGISTRATION = PUBLIC/'scoped_supervision_audit_registration_v3.json'
SUCCESS = 'PASS_SCOPED_SUPERVISION_OBSERVABILITY_AUDIT_NO_GT'


def graph(structure):
    if not structure['full_structure_consensus']:
        return None
    accepted = structure['accepted_structure']
    if accepted['logic'] != 'and' or any(t['kind'] == 'opaque' for t in accepted['terms']):
        return None
    return frozenset(tuple(t[k] for k in FIELDS) for t in accepted['terms'])


def comparison(left, right):
    """Language-conditional relationships, not facts about target objects."""
    if left is None or right is None or left == right:
        return None
    if left < right:
        return 'left_is_subset', None
    if right < left:
        return 'right_is_subset', None
    lo, ro = left-right, right-left
    if len(lo) == len(ro) == 1:
        a, b = next(iter(lo)), next(iter(ro))
        # Only the predicate value may change. Binding/reference/scope and
        # every other clause must remain identical in the accepted parse.
        rest = [i for i, key in enumerate(FIELDS) if key != 'value']
        if all(a[i] == b[i] for i in rest):
            term = dict(zip(FIELDS, a))
            if (term['kind'] == 'appearance' and term['binding'] == 'subject'
                    and term['time'] == 'appearance' and term['part'] in ('whole', 'upper', 'lower')):
                return 'single_subject_appearance_value_contrast', term['name']+'_'+term['part']
    return None


def visual_context(row):
    request = row['input']; prompt = request['prompt']; query = row['query_text']
    if query not in prompt or not request.get('visual'):
        return None
    # Include the actual rendered visual digest and the full residual prompt,
    # not merely TrackID or nominal view. CLIP fallback stays its actual view.
    return (int(row['track_id']), int(row['frame_id']), row['effective_view'],
        request['visual'], tuple(request['bbox']), prompt.replace(query, '<QUERY>', 1))


def quantiles(values):
    if not values:
        return dict(count=0)
    levels = (0, .25, .5, .75, .9, .95, 1)
    return dict(count=len(values), mean=float(np.mean(values)),
        quantiles={str(q):float(v) for q, v in zip(levels, np.quantile(values, levels))})


def run():
    guard = generation_guard(); start = time.perf_counter()
    structures = json.loads((SCOPES/'structures.json').read_text())['structures']
    summary = {}; completed = 0
    for pair, (_, target) in PAIRS.items():
        counts = Counter(); values = defaultdict(list); records = []
        for video, _ in sorted(metadata(target)['videos'].items()):
            queries = [q for q in metadata(target)['episodes'] if q['video_id'] == video]
            texts = sorted({q['query_text'] for q in queries})
            graphs = {text:graph(structures[text_key(text)]) for text in texts}
            counts['query_video_texts'] += len(texts)
            counts['query_video_texts_with_full_nonopaque_consensus'] += sum(g is not None for g in graphs.values())
            pairs = {}
            for left, right in itertools.combinations(texts, 2):
                kind = comparison(graphs[left], graphs[right])
                if kind is not None:
                    pairs[left, right] = kind
                    counts['linguistic_pairs_'+kind[0]] += 1
            document = json.loads((ANSWERS/'policy_answers'/pair/video/'DESIGN_answers.json').read_text())
            assert document['only_this_policy_requested_responses'] and document['GT_open_attempts'] == 0
            counts['logical_requests'] += document['logical_requested']
            seen = set(); contexts = defaultdict(lambda:defaultdict(list))
            for row in document['answers']:
                assert row['frozen_weight_sha256'] == TEACHER_SHA
                if row['request_id'] in seen:
                    counts['repeated_exact_request_records_not_extra_evidence'] += 1
                    continue
                seen.add(row['request_id']); context = visual_context(row)
                if context is None:
                    counts['unmatchable_visual_or_prompt_context'] += 1
                    continue
                contexts[context][row['query_text']].append(row)
            counts['unique_requested_inputs'] += len(seen)
            counts['exact_visual_prompt_contexts'] += len(contexts)
            for context, answered in contexts.items():
                if len(answered) >= 2:
                    counts['contexts_with_multiple_query_texts'] += 1
                for left, right in itertools.combinations(sorted(answered), 2):
                    if (left, right) not in pairs:
                        continue
                    kind, slot = pairs[left, right]
                    pleft = float(np.mean([r['yes_probability'] for r in answered[left]]))
                    pright = float(np.mean([r['yes_probability'] for r in answered[right]]))
                    record = dict(video=video, track=context[0], frame=context[1], effective_view=context[2],
                        left_query=left, right_query=right, relationship=kind, slot=slot,
                        left_probability=pleft, right_probability=pright,
                        left_requests=[r['request_id'] for r in answered[left]],
                        right_requests=[r['request_id'] for r in answered[right]])
                    counts['actually_matched_pairs_'+kind] += 1
                    if kind.endswith('_is_subset'):
                        subset, conjunction = (pleft, pright) if kind == 'left_is_subset' else (pright, pleft)
                        residual = conjunction-subset
                        values['conjunction_probability_minus_subset_probability'].append(residual)
                        counts['matched_subset_conjunction_pairs'] += 1
                        counts['teacher_values_violate_ideal_event_monotonicity_strictly'] += int(residual > 0)
                    else:
                        values['absolute_teacher_difference_'+slot].append(abs(pleft-pright))
                    records.append(record)
            completed += 1
            atomic_json(STATE, dict(status='AUDITING_EXISTING_SCOPE_MATCHED_TEACHER_OBSERVATIONS',
                pid=os.getpid(), completed=completed, total=148, current_pair=pair, video=video,
                manual_GT_open_attempts=len(guard), goal_status='ACTIVE'))
        atomic_json(OUT/pair/'matched_observations.json', dict(records=records, not_to_Git=True,
            only_DESIGN_own_answers=True, teacher_probability_comparisons_not_truth=True))
        summary[pair] = dict(counts=dict(counts), distributions={k:quantiles(v) for k, v in values.items()})
    assert completed == 148 and not guard
    report = dict(status=SUCCESS, settings=summary, completed_setting_videos=completed,
        source_policy='DESIGN fixed before this audit; no other policy answers borrowed',
        matcher='Same rendered visual digest, provided ID, frame, actual view, bbox and complete non-query prompt.',
        structure_scope='Both fixed language prompts agree on an AND structure with no opaque term; consensus remains fallible.',
        interpretation='Ideal event conjunction is a subset event. Raw Qwen answer probabilities need not form a calibrated joint distribution; disagreement does not label target semantic truth.',
        appearance_scope='One accepted appearance predicate value changes; this is not a true positive/negative label, and lexical values need not be mutually exclusive.',
        identified_factors_or_new_learning_claim=False,
        no_new_pseudo_atom_labels_or_model_parameters=True,
        full_corpus_audit_not_future_video_input_to_an_adaptation_head=True,
        worker_wall_seconds=time.perf_counter()-start, manual_GT_open_attempts=0,
        new_teacher_calls=0, new_neural_fits=0, goal_status='ACTIVE', research_goal_complete=False)
    atomic_json(OUT/'completion.json', report)
    (PUBLIC/'scoped_supervision_audit_receipt.json').write_text(json.dumps(report, separators=(',', ':'))+'\n')
    atomic_json(STATE, dict(status=SUCCESS, pid=os.getpid(), completed=completed, goal_status='ACTIVE'))


def main():
    registration = json.loads(REGISTRATION.read_text())
    for name, expected in registration['source_sha256'].items():
        assert file_sha(ROOT/name) == expected, name
    STATE.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(STATE, dict(status='AWAITING_FIXED_QUERY_SCOPE_EXTRACTION', pid=os.getpid(), completed=0))
    while True:
        if not SCOPE_STATE.is_file():
            time.sleep(15)
            continue
        previous = json.loads(SCOPE_STATE.read_text())
        if previous['status'].startswith('FAILED_'):
            raise RuntimeError('Resolve text-scope extraction failure first')
        if previous['status'] == SCOPE_SUCCESS:
            break
        time.sleep(15)
    run()


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        atomic_json(STATE, dict(status='FAILED_SCOPED_SUPERVISION_AUDIT', pid=os.getpid(),
            error=repr(error), goal_status='ACTIVE'))
        raise
