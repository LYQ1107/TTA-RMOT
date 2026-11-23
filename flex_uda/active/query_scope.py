"""Typed, fallible language structure for future weak-supervision models.

These are interpretations of task text, not labels about target objects.
Consensus is not semantic ground truth. Unknown wording remains available
as the original whole query, rather than becoming a negative atom label.
"""
import json
import re

FIELDS = ('kind', 'name', 'value', 'part', 'binding', 'reference', 'time', 'negated')
ENUMS = dict(
    kind={'entity','appearance','spatial','motion','role','activity','opaque'},
    name={'entity','color','color_pattern','garment','gender','number','side','team_side',
          'depth','relative_motion','motion','role','activity','pose','other'},
    part={'upper','lower','whole','none','unspecified'},
    binding={'subject','reference','unspecified'},
    reference={'none','image','ego','team','world','other','unspecified'},
    time={'appearance','instant','interval','role_or_action','unspecified'})
SUBJECTS = {'person','car','vehicle','other','unknown'}


def norm(value):
    return ' '.join(value.lower().strip().split())


def fallback(text, reason):
    return dict(subject='unknown', subject_span=text, logic='opaque',
        terms=[dict(kind='opaque',name='other',value=norm(text),part='unspecified',
            binding='unspecified',reference='unspecified',time='unspecified',negated=False,span=text)],
        uncertain=[reason], valid_format=False, original_query_preserved=text)


def decode(text, response):
    raw = response.strip()
    if raw.startswith('```') and raw.endswith('```'):
        raw = re.sub(r'^```(?:json)?\s*', '', raw, count=1)
        raw = raw[:-3].strip()
    try:
        data = json.loads(raw)
        assert isinstance(data,dict)
        assert data['subject'] in SUBJECTS and data['logic'] in ('and','opaque')
        assert isinstance(data['subject_span'],str) and data['subject_span'].strip()
        assert norm(data['subject_span']) in norm(text)
        assert isinstance(data['terms'],list) and data['terms']
        assert isinstance(data['uncertain'],list) and all(isinstance(s,str) for s in data['uncertain'])
        terms=[]
        for term in data['terms']:
            for key, values in ENUMS.items():assert term[key] in values
            assert type(term['negated']) is bool
            assert isinstance(term['value'],str) and term['value'].strip()
            assert isinstance(term['span'],str) and term['span'].strip()
            assert norm(term['span']) in norm(text)
            terms.append({**{key:term[key] for key in FIELDS},
                'value':norm(term['value']), 'span':term['span']})
        return dict(subject=data['subject'],subject_span=data['subject_span'],logic=data['logic'],
            terms=terms,uncertain=data['uncertain'],valid_format=True,original_query_preserved=text)
    except (ValueError,TypeError,KeyError,AssertionError):
        return fallback(text, 'unresolved_format_or_nonliteral_evidence_span')


def signature(parsed):
    terms = sorted({tuple(term[key] for key in FIELDS) for term in parsed['terms']})
    return parsed['subject'], parsed['logic'], tuple(terms)


def consensus(text, first, second):
    left={tuple(t[k] for k in FIELDS):t for t in first['terms']}
    right={tuple(t[k] for k in FIELDS):t for t in second['terms']}
    shared=[left[k] for k in sorted(left.keys() & right.keys())]
    complete=first['valid_format'] and second['valid_format'] and signature(first)==signature(second)
    # Common terms may be useful diagnostics, but a partial intersection
    # cannot replace the entire description or certify a full factorization.
    return dict(original_query=text,full_structure_consensus=bool(complete),
        accepted_structure=first if complete else fallback(text,'fixed_prompt_structures_disagree'),
        shared_terms_diagnostic_only=shared,
        known_subject_appearance_terms=[t for t in shared if t['kind']=='appearance'
            and t['binding']=='subject' and t['time']=='appearance' and t['part'] in ('upper','lower','whole')],
        all_ambiguity_notes=sorted(set(first['uncertain']+second['uncertain'])),
        teacher_consensus_is_not_semantic_truth=True,
        no_manual_target_annotations_or_image_truth=True)
