"""Bound negative evidence by the externally supplied referring contract.

For promised presence, a queried negative remains a local rejection rather
than a boundary value that diffuses through unqueried people. CHECKLIST keeps
full propagation. A total-edge-weight matched control isolates simple
weakening from the choice of which evidence may travel.
"""
import numpy as np
from flex_uda.active.topology_selection import graph_matrices,apply_graph


def contract_graphs(unary,pcd,colours,texts,teacher,protocol):
    assert protocol in ('PRESUPPOSED','CHECKLIST')
    geometry,full,_,_,supported=graph_matrices(pcd,colours,texts)
    positive=full.copy();weak=full.copy();negative=np.isfinite(teacher)&(teacher<.5)
    for q in range(len(texts)):
        positive[q,negative[q],:]=0.;positive[q,:,negative[q]]=0.
        if full[q].sum():weak[q]*=positive[q].sum()/full[q].sum()
    contract=positive if protocol=='PRESUPPOSED' else full
    matched=weak if protocol=='PRESUPPOSED' else full
    scores={name:apply_graph(unary,teacher,geometry,w,supported) for name,w in dict(FULL=full,CONTRACT=contract,POSITIVE=positive,WEAK=matched).items()}
    error=float(np.max(np.abs(positive.sum((1,2))-weak.sum((1,2)))))
    assert error<1e-10
    return scores,dict(protocol=protocol,rule='No anchors remain local only under PRESUPPOSED; CHECKLIST full',
        negative_teacher_nodes=int(negative.sum()),supported_queries=int(supported.sum()),
        full_edge_weight=float(full.sum()),positive_only_edge_weight=float(positive.sum()),
        contract_edge_weight=float(contract.sum()),matched_edge_total_max_error=error,
        teacher_No_threshold=.5,no_manual_target_GT=True,no_new_teacher_calls=True)
