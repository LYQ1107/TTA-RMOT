"""Selective negative transfer veto from held-TrackID visual evidence.

Use the existing ln(3) appearance-correction cap, not a target-GT-tuned margin.
A negative teacher anchor stays a local rejection. Its edges may be removed
when the cross-fitted visual calibrator supplies its maximal positive odds
correction in a promised-presence task. No network or teacher is updated.
"""
import hashlib
import numpy as np
from scipy.special import logit
from flex_uda.active.topology_selection import graph_matrices,apply_graph


def consensus_graphs(unary,pcd,colours,texts,teacher,protocol,probability,base,active,video):
    assert protocol in ('PRESUPPOSED','CHECKLIST')
    geometry,full,_,valid,supported=graph_matrices(pcd,colours,texts)
    positive_evidence=logit(np.clip(probability,.02,.98))-logit(np.clip(base,.02,.98))
    negative=np.isfinite(teacher)&(teacher<.5)&valid&supported[:,None]
    conflicting=negative&active&(positive_evidence>=np.log(3.))
    selective=full.copy();weak=full.copy();random=full.copy();fallbacks=0
    for q in range(len(texts)):
        selected=conflicting[q];selective[q,selected,:]=0.;selective[q,:,selected]=0.
        if full[q].sum():weak[q]*=selective[q].sum()/full[q].sum()
        number=int(selected.sum());candidates=np.flatnonzero(negative[q])
        rng=np.random.default_rng(int(hashlib.sha256((video+':'+str(q)+':cue_veto_random').encode()).hexdigest()[:8],16))
        removed=rng.choice(candidates,number,replace=False) if number else np.empty(0,dtype=int)
        random[q,removed,:]=0.;random[q,:,removed]=0.
        if random[q].sum():random[q]*=selective[q].sum()/random[q].sum()
        elif selective[q].sum():random[q]=weak[q];fallbacks+=1
    contract=selective if protocol=='PRESUPPOSED' else full
    matched=weak if protocol=='PRESUPPOSED' else full
    randomized=random if protocol=='PRESUPPOSED' else full
    weights=dict(FULL=full,CONSENSUS=contract,ALL_PROTOCOL=selective,WEAK=matched,RANDOM=randomized)
    scores={name:apply_graph(unary,teacher,geometry,w,supported) for name,w in weights.items()}
    errors={name:float(np.max(np.abs(w.sum((1,2))-selective.sum((1,2))))) for name,w in dict(WEAK=weak,RANDOM=random).items()}
    assert max(errors.values())<1e-10
    return scores,dict(protocol=protocol,rule='PRESUPPOSED: vetoedgesonlyforNoopposedbyleaveTrackIDoutvisualcorrection>=existingln3cap;CHECKLISTfull',
        negative_teacher_nodes=int(negative.sum()),visual_conflicting_negative_nodes=int(conflicting.sum()),
        affected_queries=int(conflicting.any(1).sum()),teacher_No_threshold=.5,visual_evidence_threshold=float(np.log(3.)),threshold_inherited_from_existing_odds_cap=True,
        current_track_answers_excluded_from_visual_signal=True,No_anchor_retained_locally=True,
        matched_edge_total_errors=errors,random_No_isolation_count_matched=True,random_zero_edge_fallback_to_weak_queries=fallbacks,
        no_manual_target_GT=True,no_extra_teacher_calls=True,correlated_evidence_regularization_not_independent_Bayes=True)
