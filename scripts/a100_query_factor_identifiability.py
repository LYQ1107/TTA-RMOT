"""Query-only scope and identification audit before a shared-factor model.

The Boolean conjunction model below is a working design model, not semantic
GT. No teacher response, feature, prediction, evaluator or target label enters.
"""
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from flex_uda.a100.mechanism import metadata,generation_guard
from flex_uda.active.query_consistency import canonical
from scripts.a100_language_scope_audit import classify
from scripts.a100_six_direction import OUT as SIX,PUBLIC

OUT=SIX/'query_factor_identifiability'


def literal_factors(text):
    parsed=canonical(text);scope=classify(text)
    unsafe=not parsed['supported'] or scope['reference_scope_review'] or scope['negation_review']
    digest=lambda value:hashlib.sha256(value.encode()).hexdigest()[:16]
    if unsafe:
        return dict(factors=('WHOLE_QUERY_'+digest(' '.join(text.lower().split())),),
            resolved_static=False,conditional_static_after_role_verified=False,
            opaque=True,unsafe_scope=True,temporal_scope_review=False,unspecified_colour=False)
    atoms=set(parsed['atoms'])
    # Remove only the existing grammar's explicit lexical entailment chains;
    # no target detection category or task factor is declared true for free.
    if 'SUBJECT_CAR' in atoms:atoms.discard('SUBJECT_VEHICLE')
    if 'ROLE_libero' in atoms or 'ROLE_goalkeeper' in atoms:atoms.discard('ROLE_PLAYER')
    if any(a.startswith(('ROLE_','GENDER_')) for a in atoms):atoms.discard('SUBJECT_PERSON')
    if 'GARMENT_tshirt' in atoms:atoms.discard('GARMENT_shirt')
    activity=bool(re.search(r'\b(?:dancing|dacning)\b',text.lower()))
    if activity:atoms.add('ACTIVITY_OR_ROLE_DANCING')
    opaque=bool(parsed['opaque'])
    if opaque:atoms.add('UNRESOLVED_QUALIFIER_'+digest(parsed['opaque']))
    unspecified=any(a.startswith('UNSPECIFIED_COLOUR_') for a in atoms)
    base=not opaque and not unspecified
    return dict(factors=tuple(sorted(atoms)),resolved_static=base and not activity,
        conditional_static_after_role_verified=base and activity,opaque=opaque,
        unsafe_scope=False,temporal_scope_review=activity,unspecified_colour=unspecified)


def incidence(rows):
    names=sorted({factor for row in rows for factor in row['factors']})
    index={name:i for i,name in enumerate(names)};matrix=np.zeros((len(rows),len(names)),bool)
    for qi,row in enumerate(rows):
        for factor in row['factors']:matrix[qi,index[factor]]=True
    return names,matrix


def matrix_audit(names,matrix):
    if not len(names) or not len(matrix):
        return dict(queries=len(matrix),factors=len(names),rank=0,duplicate_column_classes=0,
            indistinguishable_single_failure_pairs=0,universal_factors=[],examples=[])
    patterns=defaultdict(list)
    for index,name in enumerate(names):patterns[matrix[:,index].tobytes()].append(name)
    aliases=[values for values in patterns.values() if len(values)>1]
    return dict(queries=len(matrix),factors=len(names),rank=int(np.linalg.matrix_rank(matrix.astype(float))),
        duplicate_column_classes=len(aliases),
        indistinguishable_single_failure_pairs=sum(len(a)*(len(a)-1)//2 for a in aliases),
        universal_factors=[name for i,name in enumerate(names) if matrix[:,i].all()],
        examples=sorted(aliases,key=lambda a:(-len(a),a))[:3])


def check_boolean_quotient():
    # These are artificial states, not target labels. Co-occurring columns
    # can be merged into one conjunction without changing this query family.
    matrix=np.array([[1,1,0,0],[1,1,1,0],[0,0,0,1]],bool)
    reduced=matrix[:,[0,2,3]]
    for bits in range(16):
        state=np.array([(bits>>j)&1 for j in range(4)])
        merged=np.array([state[0]*state[1],state[2],state[3]])
        first=matrix@(1-state)==0;second=reduced@(1-merged)==0
        assert np.array_equal(first,second)
    a=np.array([0,1,1,1]);b=np.array([1,0,1,1])
    assert np.array_equal(matrix@(1-a)==0,matrix@(1-b)==0)
    return dict(all16artificial_assignments_preserved=True,indistinguishable_failure_witness=True,
        claim='Conditional Boolean algebra only; a real semantic feasibility/accuracy claim requires additional assumptions.')


def main():
    registration=json.loads((PUBLIC/'query_factor_identifiability_registration.json').read_text())
    assert registration['source_sha256']==file_sha(Path(__file__))
    started=time.perf_counter();guard=generation_guard();results={}
    for target in ('Dance','KITTI-V2','SportsMOT'):
        m=metadata(target);totals=Counter();videos=[]
        for video in sorted(m['videos']):
            queries=[q for q in m['episodes'] if q['video_id']==video]
            rows=[literal_factors(q['query_text']) for q in queries]
            names,matrix=incidence(rows);all_literals=matrix_audit(names,matrix)
            safe=[r for r in rows if r['resolved_static']];safe_names,safe_matrix=incidence(safe)
            universal=matrix.all(0);conditional=matrix_audit([n for n,u in zip(names,universal) if not u],matrix[:,~universal])
            count=Counter(queries=len(queries),literal_factor_occurrences=int(matrix.sum()),distinct_literal_factors=len(names))
            for key in ('resolved_static','conditional_static_after_role_verified','opaque','unsafe_scope','temporal_scope_review','unspecified_colour'):
                count[key]=sum(r[key] for r in rows)
            count['factors_reused_in_multiple_queries']=int((matrix.sum(0)>1).sum())
            count['Boolean_alias_failure_pairs']=all_literals['indistinguishable_single_failure_pairs']
            totals.update(count)
            videos.append(dict(video=video,counts=dict(count),all_literal_working_model=all_literals,
                resolved_static_only=matrix_audit(safe_names,safe_matrix),
                conditional_if_universal_factors_already_verified_true=conditional))
        results[target]=dict(videos=len(videos),counts=dict(totals),video_diagnostics=videos,
            caution='Summed per-video distinct factor counts are not a global vocabulary size; common factors are not assumed true by the algorithm.')
    assert not guard
    result=dict(status='TASK_QUERY_ONLY_FACTOR_IDENTIFIABILITY_AUDIT_COMPLETE',targets=results,
        Boolean_scope_check=check_boolean_quotient(),manual_target_GT_open_attempts=0,
        teacher_answers_features_predictions_read=0,new_GPU_teacher_or_training_calls=0,
        interpretation='Scope and measurement-design diagnosis, not an accepted model or evidence of improved RMOT. A whole-queryNo only constrains a conjunction; aliases can sometimes be merged instead of queried separately.',
        limitations='Parser assumptions, unresolved temporal/colour/reference scope and semantic restrictions limit Boolean witness interpretation. Matrix rank refers only to a hypothetical linearized factor design, not identification of true semantic probabilities.',
        next='Decide which complementary teacher views or subquestions are useful for the task; give ordinary baselines the same measurements. Do not automatically atomize every description or call textbook identifiability an innovation.',
        all_data_development=True,existing_live_methods_changed=False,wall_seconds=time.perf_counter()-started,goal_status='ACTIVE')
    atomic_json(OUT/'completion.json',result)
    (PUBLIC/'query_factor_identifiability_audit.json').write_text(json.dumps(result,ensure_ascii=False,separators=(',',':'))+'\n')
    print(json.dumps(dict(status=result['status'],targets={t:r['counts'] for t,r in results.items()},
        Boolean_scope_check=result['Boolean_scope_check'],manual_target_GT_open_attempts=0),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
