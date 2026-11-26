"""Unlabelled text coverage and independent convex-projection sanity check."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.optimize import minimize,LinearConstraint
from flex_uda.active.query_consistency import relations,project
from flex_uda.a100.mechanism import generation_guard,metadata
from flex_tta.runtime.pf_receipts import atomic_json
from scripts.a100_six_direction import OUT,PUBLIC


def main():
    guard=generation_guard();rng=np.random.default_rng(20261122);error=0.;objective_error=0.;cases=[]
    for n in (2,5,9):
        edges=np.asarray([(a,b) for a in range(n) for b in range(a+1,n) if rng.random()<.5],int).reshape(-1,2)
        if n==5:edges=np.r_[edges,[[1,0],[0,1]]]
        if not len(edges):edges=np.asarray([[0,1]])
        u=rng.uniform(0,1,(n,4));actual,receipt=project(u,edges)
        matrix=np.zeros((len(edges),n))
        for i,(a,b) in enumerate(edges):matrix[i,a]=1.;matrix[i,b]=-1.
        for col in range(4):
            target=u[:,col]
            fitted=minimize(lambda x:.5*np.square(x-target).sum(),target.copy(),jac=lambda x:x-target,
                constraints=[LinearConstraint(matrix,-np.inf,0.)],bounds=[(0.,1.)]*n,method='SLSQP',options=dict(ftol=1e-12,maxiter=500))
            assert fitted.success,fitted.message
            error=max(error,float(np.max(np.abs(actual[:,col]-fitted.x))))
            objective_error=max(objective_error,abs(.5*np.square(actual[:,col]-target).sum()-fitted.fun))
        cases.append(dict(queries=n,edges=len(edges),projection=receipt))
    assert error<1e-6 and objective_error<1e-8,(error,objective_error)
    fixtures=['player','basketball player','basketball player in white','player in white clothes',
        'person with white Tshirt and black pants','person with white shirt and black pants',
        'person with black shirt and white pants','cars','black cars','cars behind black truck','white person','person in white']
    parsed,edges,equivalent=relations(fixtures);edge_set=set(map(tuple,edges.tolist()))
    assert (2,0) in edge_set and (2,3) in edge_set and (3,2) in edge_set
    assert (4,5) in edge_set and (5,4) not in edge_set
    assert (9,8) not in edge_set and (10,11) not in edge_set and (11,10) not in edge_set
    report={}
    for target in ('Dance','KITTI-V2','SportsMOT'):
        m=metadata(target);videos=[]
        for video in sorted(m['videos']):
            texts=[q['query_text'] for q in m['episodes'] if q['video_id']==video]
            parsed,edges,eq=relations(texts)
            videos.append(dict(video=video,queries=len(texts),supported=sum(p['supported'] for p in parsed),
                implications=len(edges),equivalent_pairs=len(eq),
                examples=[dict(first=texts[a],second=texts[b],equivalent=(min(a,b),max(a,b)) in eq) for a,b in edges[:12]],
                all_relations=[list(map(int,e)) for e in edges]))
        report[target]=dict(videos=videos,queries=sum(r['queries'] for r in videos),
            supported_queries=sum(r['supported'] for r in videos),implications=sum(r['implications'] for r in videos),
            equivalent_pairs=sum(r['equivalent_pairs'] for r in videos))
    assert not guard
    result=dict(status='PASS_TEXT_ONLY_QUERY_CONSISTENCY_AUDIT',GT_open_attempts=0,targets=report,
        independent_SLSQP_max_probability_error=error,independent_SLSQP_max_objective_error=objective_error,cases=cases,
        assumptions='Conservative parsed attribute entailment and one sport within a video; unparsed qualifiers preserved. Linguistic assumptions require post-seal error review, not logical GT.',
        planned_decoder_audit='All methods get ordinary per-query >.5 decoding without assumed 17-query mutual exclusivity; identical relaxations for ordinary KD and proposed.',
        no_new_teacher_or_neural_calls=True,goal_status='ACTIVE')
    atomic_json(OUT/'query_consistency/text_audit.json',result)
    (PUBLIC/'query_consistency_text_audit.json').write_text(json.dumps(result,ensure_ascii=False,separators=(',',':'))+'\n')
    print(json.dumps(dict(status=result['status'],targets={k:{n:v[n] for n in ('queries','supported_queries','implications','equivalent_pairs')} for k,v in report.items()},
        independent_solver_max_probability_error=error,GT_open_attempts=0)),flush=True)


if __name__=='__main__':main()
