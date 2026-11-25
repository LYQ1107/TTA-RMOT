"""Teacher-only redundancy/contradiction audit on sealed six-way QA.

Reads only replies actually requested by each stream. No target GT, unasked
alternative views, HOTA or fitted checkpoint selection enters this audit.
"""
from collections import defaultdict
import json
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from flex_tta.runtime.pf_receipts import atomic_json
from flex_uda.a100.mechanism import metadata,generation_guard
from flex_uda.active.query_consistency import relations
from scripts.a100_six_direction import OUT,PUBLIC,PAIRS,source

STATE=OUT/'semantic_budget_audit/state.json'


def audit(pair):
    m=metadata(PAIRS[pair][1]);videos=[]
    for video,bank in sorted(m['videos'].items()):
        texts=[q['query_text'] for q in m['episodes'] if q['video_id']==video]
        _,edges,equivalent=relations(texts);parent=list(range(len(texts)))
        def find(i):
            while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
            return i
        for a,b in equivalent:parent[find(b)]=find(a)
        group=np.asarray([find(i) for i in range(len(texts))]);answers=json.loads((source(pair)/'source'/video/'answers.json').read_text())['answers']
        cells=defaultdict(list);teacher={};per_query=defaultdict(list)
        for row in answers:
            q,t=int(row['local_query']),int(row['local_track']);p=float(row['yes_probability'])
            cells[int(group[q]),t].append((q,p));teacher[q,t]=p;per_query[q].append(p)
        overlap=[(key,rows) for key,rows in cells.items() if len(rows)>1]
        held_loss=[];neutral_loss=[];source_loss=[];implication=[]
        with np.load(source(pair)/'source'/video/'sparse_inputs.npz') as a:prior=a['prior'];tracks=a['tracks']
        for (_,t),rows in overlap:
            for q,p in rows:
                rest=[other for oq,other in rows if oq!=q]
                if not rest:continue
                prediction=float(np.clip(np.mean(rest),.02,.98));raw=float(np.clip(prior[q,t],.02,.98))
                held_loss.append(-p*np.log(prediction)-(1-p)*np.log1p(-prediction));neutral_loss.append(np.log(2.))
                source_loss.append(-p*np.log(raw)-(1-p)*np.log1p(-raw))
        for a,b in edges:
            for t in range(len(tracks)):
                if (int(a),t) in teacher and (int(b),t) in teacher:
                    pa,pb=teacher[int(a),t],teacher[int(b),t]
                    implication.append(dict(first=int(a),second=int(b),local_track=t,positive_order_violation=max(0.,pa-pb),
                        contradictory_hard=pa>.5 and pb<=.5,strong_contradiction=pa>=.8 and pb<=.2))
        row=dict(video=video,queries=len(texts),semantic_classes=len(set(group)),calls=len(answers),
            unique_group_track_cells=len(cells),potential_exact_cell_duplicate_calls=len(answers)-len(cells),
            overlapping_cells=len(overlap),hard_disagreement_cells=sum(min(p for _,p in rows)<=.5<max(p for _,p in rows) for _,rows in overlap),
            strong_disagreement_cells=sum(min(p for _,p in rows)<=.2 and max(p for _,p in rows)>=.8 for _,rows in overlap),
            equivalence_teacher_leave_one_query_NLL=float(np.mean(held_loss)) if held_loss else None,
            same_rows_source_prior_NLL=float(np.mean(source_loss)) if source_loss else None,
            neutral_teacher_NLL=float(np.mean(neutral_loss)) if neutral_loss else None,
            leave_one_query_teacher_observations=len(held_loss),
            queries_without_any_teacher_yes=sum(not any(p>.5 for p in per_query[q]) for q in range(len(texts))),
            queries_with_no_teacher_calls=sum(not per_query[q] for q in range(len(texts))),
            implication_observed_pairs=len(implication),implication_hard_contradictions=sum(r['contradictory_hard'] for r in implication),
            implication_strong_contradictions=sum(r['strong_contradiction'] for r in implication),
            mean_positive_order_violation=float(np.mean([r['positive_order_violation'] for r in implication])) if implication else None)
        videos.append(row)
    counts={key:sum(row[key] for row in videos) for key in ('queries','semantic_classes','calls','unique_group_track_cells',
        'potential_exact_cell_duplicate_calls','overlapping_cells','hard_disagreement_cells','strong_disagreement_cells',
        'leave_one_query_teacher_observations','queries_without_any_teacher_yes','queries_with_no_teacher_calls',
        'implication_observed_pairs','implication_hard_contradictions','implication_strong_contradictions')}
    return dict(pair=pair,counts=counts,videos=videos,GT_open_attempts=0,teacher_agreement_is_not_accuracy=True,
        duplicate_calls_are_only_semantically_redundant_under_text_parser_assumptions=True,
        future_policy_must_still_count_actual_calls_and_no_hidden_answer_selection=True)


def main():
    guard=generation_guard();pending=list(PAIRS);results={};start=time.perf_counter();last=0
    while pending:
        changed=False
        for pair in list(pending):
            if not (source(pair)/'completion.json').is_file():continue
            results[pair]=audit(pair);pending.remove(pair);changed=True
            atomic_json(OUT/'semantic_budget_audit'/f'{pair}.json',results[pair])
        if changed or time.perf_counter()-last>45:
            atomic_json(STATE,dict(status='RUNNING_TEACHER_ONLY_SEMANTIC_BUDGET_AUDIT',pid=os.getpid(),completed={p:r['counts'] for p,r in results.items()},
                pending=pending,wall_seconds=time.perf_counter()-start,goal_status='ACTIVE'));last=time.perf_counter()
        if pending:time.sleep(3)
    assert not guard
    report=dict(status='PASS_SIX_DIRECTION_TEACHER_ONLY_BUDGET_AUDIT',pairs=results,GT_open_attempts=0,new_teacher_encoder_head_calls=0,
        no_manual_target_labels=True,no_HOTA_in_method_design=True,teacher_consistency_not_semantic_GT=True,goal_status='ACTIVE')
    atomic_json(OUT/'semantic_budget_audit/analysis.json',report)
    (PUBLIC/'semantic_budget_teacher_only_audit.json').write_text(json.dumps(report,ensure_ascii=False,separators=(',',':'))+'\n')
    atomic_json(STATE,dict(status=report['status'],pid=os.getpid(),completed={p:r['counts'] for p,r in results.items()},goal_status='ACTIVE'))
    print(json.dumps(dict(status=report['status'],counts={p:r['counts'] for p,r in results.items()})),flush=True)


if __name__=='__main__':
    try:main()
    except Exception as error:
        atomic_json(STATE,dict(status='FAILED_SEMANTIC_BUDGET_AUDIT_REQUIRES_ROOT_CAUSE',error=repr(error),pid=os.getpid(),goal_status='ACTIVE'));raise
