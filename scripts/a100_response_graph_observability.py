"""Audit which query--visual interactions the paid observations can identify.

No parsing, GT, neural fitting, new teacher requests or RMOT outputs enter.
The additive projection is a diagnostic, not an output calibration method.
"""
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsqr
from flex_tta.runtime.pf_receipts import atomic_json, file_sha
from flex_uda.a100.mechanism import PAIRS, generation_guard, metadata
from scripts.a100_observation_design_collect import OUT as ANSWERS, TEACHER_SHA
from scripts.a100_scoped_supervision_audit import visual_context
from scripts.a100_six_direction import OUT as SIX, PUBLIC

PLAN = PUBLIC/'response_graph_observability_plan.json'
OUT = SIX/'response_graph_observability_20261123'
EPSILON = .02  # Same probability clipping as the active learning study.
POLICIES = ('DESIGN', 'COVERAGE_ONLY', 'PRIOR', 'UNIFORM')


def decompose(query, context, scores):
    """Orthogonal residual to row/column additive functions on paid edges."""
    qn = int(query.max())+1; cn = int(context.max())+1
    m = len(scores); edge = np.arange(m)
    matrix = csr_matrix((np.r_[np.ones(m), -np.ones(m)],
        (np.r_[edge, edge], np.r_[query, qn+context])), shape=(m, qn+cn))
    graph = nx.Graph()
    graph.add_edges_from(zip(query.tolist(), (qn+context).tolist()))
    components = nx.number_connected_components(graph)
    cycle_rank = m-(qn+cn)+components
    assert cycle_rank >= 0 and graph.number_of_edges() == m
    fitted = lsqr(matrix, scores, atol=1e-12, btol=1e-12,
        iter_lim=10*(qn+cn))[0]
    residual = scores-matrix@fitted
    orthogonality = float(np.max(np.abs(matrix.T@residual)))
    assert orthogonality < 1e-7, orthogonality
    bridges = {tuple(sorted(e)) for e in nx.bridges(graph)}
    on_cycle = np.asarray([tuple(sorted(e)) not in bridges
        for e in zip(query.tolist(), (qn+context).tolist())])
    energy = float(residual@residual)
    total_energy = float(scores@scores)
    return dict(query_view_nodes=qn, exact_visual_context_nodes=cn,
        unique_query_context_edges=m, connected_components=components,
        interaction_dimension=cycle_rank,
        edges_on_at_least_one_cycle=int(on_cycle.sum()),
        edges_on_no_cycle=int((~on_cycle).sum()),
        additive_projection_residual_squared_norm=energy,
        original_clipped_logit_squared_norm=total_energy,
        orthogonality_max_absolute_error=orthogonality), residual


def mathematics_checks():
    q = np.asarray([0, 0, 1, 1]); c = np.asarray([0, 1, 0, 1])
    z = np.asarray([1., 0., 0., 1.])
    base, residual = decompose(q, c, z)
    changed = z+np.asarray([8., 8., -3., -3.])+np.asarray([2., -4., 2., -4.])
    _, other = decompose(q, c, changed)
    cancellation = float(np.max(np.abs(residual-other)))
    expected = float(np.max(np.abs(residual-np.asarray([.5, -.5, -.5, .5]))))
    tree, tree_residual = decompose(q[:3], c[:3], np.asarray([3., 8., -2.]))
    assert base['interaction_dimension'] == 1 and tree['interaction_dimension'] == 0
    assert max(cancellation, expected, float(np.max(np.abs(tree_residual)))) < 1e-10
    return dict(two_by_two_additive_shift_cancellation_max_error=cancellation,
        exact_two_by_two_interaction_residual_max_error=expected,
        tree_interaction_dimension=0,
        arbitrary_tree_scores_residual_max_error=float(np.max(np.abs(tree_residual))))


def audit_video(document):
    seen = set(); edges = defaultdict(list); counts = Counter()
    for row in document['answers']:
        counts['answer_records'] += 1
        assert row['frozen_weight_sha256'] == TEACHER_SHA
        if row['request_id'] in seen:
            counts['repeated_exact_request_records'] += 1
            continue
        seen.add(row['request_id'])
        context = visual_context(row)
        if context is None:
            counts['unmatchable_context_records'] += 1
            continue
        # Retain the WHOLE original query. Actual views have separate row
        # effects; no comparison of an atom parser's lossy representations.
        key = (row['query_text'], row['effective_view']), context
        edges[key].append(float(row['yes_probability']))
    counts['unique_requested_inputs'] = len(seen)
    counts['same_query_context_extra_records'] = sum(len(v)-1 for v in edges.values())
    qmap = {q:i for i,q in enumerate(sorted({k[0] for k in edges}))}
    cmap = {c:i for i,c in enumerate(sorted({k[1] for k in edges}))}
    query = np.asarray([qmap[k[0]] for k in edges], dtype=int)
    context = np.asarray([cmap[k[1]] for k in edges], dtype=int)
    probability = np.asarray([np.mean(v) for v in edges.values()])
    probability = probability.clip(EPSILON, 1-EPSILON)
    scores = np.log(probability)-np.log1p(-probability)
    stats, _ = decompose(query, context, scores)
    return dict(counts=dict(counts), **stats)


def main():
    plan = json.loads(PLAN.read_text())
    for name, sha in plan['source_sha256'].items():
        assert file_sha(ROOT/name) == sha, name
    guard = generation_guard(); started = time.perf_counter()
    checks = mathematics_checks(); settings = {}; completed = 0
    for pair, (_, target) in PAIRS.items():
        policies = {}
        for policy in POLICIES:
            videos = []
            for video in sorted(metadata(target)['videos']):
                document = json.loads((ANSWERS/'policy_answers'/pair/video/(policy+'_answers.json')).read_text())
                assert document['GT_open_attempts'] == 0 and document['only_this_policy_requested_responses']
                record = dict(video=video, **audit_video(document)); videos.append(record)
                completed += 1
                atomic_json(OUT/'progress.json', dict(completed_video_policies=completed,
                    total_video_policies=592, pair=pair, policy=policy, video=video,
                    manual_GT_open_attempts=len(guard)))
            keys = ('query_view_nodes', 'exact_visual_context_nodes', 'unique_query_context_edges',
                'connected_components', 'interaction_dimension', 'edges_on_at_least_one_cycle',
                'edges_on_no_cycle', 'additive_projection_residual_squared_norm',
                'original_clipped_logit_squared_norm')
            totals = {key:sum(v[key] for v in videos) for key in keys}
            energy = totals['original_clipped_logit_squared_norm']
            totals['interaction_energy_fraction'] = totals['additive_projection_residual_squared_norm']/energy if energy else 0.
            counts = Counter()
            for v in videos:
                counts.update(v['counts'])
            policies[policy] = dict(totals=totals, counts=dict(counts),
                videos_with_nonzero_interaction_dimension=sum(v['interaction_dimension']>0 for v in videos),
                videos_with_zero_interaction_dimension=sum(v['interaction_dimension']==0 for v in videos),
                maximum_projection_orthogonality_error=max(v['orthogonality_max_absolute_error'] for v in videos))
            atomic_json(OUT/pair/(policy+'_per_video.json'), dict(videos=videos,
                no_raw_GT_or_teacher_arrays=True, not_adaptation_inputs=True))
        settings[pair] = policies
    assert completed == 592 and not guard
    report = dict(status='PASS_PAID_RESPONSE_GRAPH_OBSERVABILITY_NO_GT', settings=settings,
        completed_video_policies=completed, numerical_checks=checks,
        probability_clipping=EPSILON, edge_weighting='one weight per unique exact whole-query/visual-context edge',
        rank_formula='m - number_of_active_query_and_context_nodes + connected_components',
        projection='Orthogonal complement of the signed bipartite incidence column space; standard least squares.',
        interpretation='Residuals are query--visual interactions relative to an additive diagnostic model. They are not automatically teacher errors, calibrated semantic effects or target labels.',
        identifiability_scope='Only interactions supported by paid graph cycles are identifiable after arbitrary row/column additive terms; tree components cannot identify them.',
        no_GT_training_new_teacher_backbone_or_RMOT_outputs=True,
        no_change_to_active_registered_numerical_recipes=True,
        raw_query_and_teacher_rows_remain_local=True,
        future_use_requires_same_information_ordinary_baselines_and_new_registration=True,
        manual_GT_open_attempts=0, wall_seconds=time.perf_counter()-started,
        source_sha256=plan['source_sha256'], goal_status='ACTIVE')
    atomic_json(OUT/'completion.json', report)
    atomic_json(PUBLIC/'response_graph_observability.json', report)
    print(json.dumps(report))


if __name__ == '__main__':
    main()
