"""Post-seal diagnosis of semantic changes within supplied Track IDs.

Physical ID impurity alone does not imply that a clothing predicate changes:
two different dancers may wear the same clothes. This distinction is measured
only after all registered outputs have been sealed and officially evaluated.
"""
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from flex_tta.runtime.pf_receipts import atomic_json
from flex_uda.a100.mechanism import OUT as DIAG, PAIRS, metadata
from flex_uda.active.color_cohort import part_mask
from flex_uda.active.acquisition import expression_types
from scripts.a100_visual_classical_benchmark_scope import FRESH
from scripts.a100_visual_classical_prepare import OUT
from scripts.a100_visual_classical_observation_prepare import OBSERVATION


def diagnose():
    assert json.loads((OUT / 'analysis/sparse_observation_official.json').read_text())['status'] == 'ALL_SPARSE_OBSERVATION_OFFICIAL_COMPLETE'
    result = {}
    for pair in ('C4', 'C1'):
        m = metadata(PAIRS[pair][1])
        totals = {name: Counter() for name in ('all_queries', 'static_clothing')}
        errors = {name: {scope: Counter() for scope in ('stable_membership', 'mixed_membership', 'unmatched_union')}
                  for name in ('POOLED_SOFT', 'ANCHOR_KD', 'ANCHOR_FULL')}
        teacher = Counter()
        physical = Counter()
        query_groups = {name: [] for name in ('static_clothing_branch', 'fallback_spatial', 'fallback_role', 'fallback_other')}
        for query in m['episodes']:
            types = expression_types(query['query_text'])
            group = ('static_clothing_branch' if part_mask(query['query_text']).any() else
                'fallback_spatial' if 'SPATIAL' in types else 'fallback_role' if 'ROLE' in types else 'fallback_other')
            query_groups[group].append(query)
        for video, bank in sorted(m['videos'].items()):
            obs = np.asarray(bank['observations'])
            truth = np.load(DIAG / 'offline_GT' / pair / video / 'labels.npy') >= 0
            union = np.load(DIAG / 'offline_GT' / pair / video / 'union_ids.npy')
            visible = union >= 0
            queries = [q for q in m['episodes'] if q['video_id'] == video]
            tracks = sorted(set(obs[:, 1].tolist()))
            with np.load(OBSERVATION / 'inputs' / pair / video / 'inputs.npz') as a:
                samples, anchors, offsets = a['samples'], a['anchors'], a['offsets']
            mixed_rows = np.zeros(truth.shape, dtype=bool)
            majority, valid_pair = np.zeros((len(queries), len(tracks)), bool), np.zeros((len(queries), len(tracks)), bool)
            for ti, track in enumerate(tracks):
                ids = np.flatnonzero(obs[:, 1] == track)
                valid = ids[visible[ids]]
                if len(valid):
                    identities, counts = np.unique(union[valid], return_counts=True)
                    physical['tracks_with_GT_visible_rows'] += 1
                    physical['tracks_with_multiple_GT_ids'] += len(identities) > 1
                    physical['GT_visible_rows'] += len(valid)
                    physical['majority_identity_rows'] += int(counts.max())
                node_ids = samples[int(offsets[ti]):int(offsets[ti+1])]
                node_valid = node_ids[visible[node_ids]]
                for qi, query in enumerate(queries):
                    if not len(valid):
                        continue
                    bits = truth[qi, valid]
                    mixed = bool(bits.any() and not bits.all())
                    majority[qi, ti] = bits.mean() >= .5
                    valid_pair[qi, ti] = True
                    mixed_rows[qi, ids] = mixed
                    observed = truth[qi, node_valid]
                    mixed_at_nodes = bool(len(observed) >= 2 and observed.any() and not observed.all())
                    for group in ('all_queries', 'static_clothing'):
                        if group == 'static_clothing' and not part_mask(query['query_text']).any():
                            continue
                        total = totals[group]
                        total['query_track_pairs_with_GT_visible_rows'] += 1
                        total['pairs_with_any_positive_row'] += bool(bits.any())
                        total['membership_mixed_pairs'] += mixed
                        total['membership_mixed_pairs_seen_by_three_nodes'] += mixed and mixed_at_nodes
                        total['GT_visible_rows'] += len(valid)
                        total['minority_semantic_membership_rows'] += min(int(bits.sum()), int((~bits).sum()))
                        total['positive_rows_in_mixed_pairs'] += int(bits.sum()) if mixed else 0
            source = FRESH / 'source' / video / 'answers.json' if pair == 'C4' else OUT / 'causal_teacher' / pair / 'COVER' / (video+'.json')
            for row in json.loads(source.read_text())['answers']:
                qi, ti = int(row['local_query']), int(row['local_track'])
                anchor = int(samples[anchors[ti]])
                teacher['asked_pairs'] += 1
                teacher['anchor_GT_union_visible'] += bool(visible[anchor])
                teacher['binary_teacher_agrees_with_anchor_IoU_membership'] += (row['yes_probability'] > .5) == bool(truth[qi, anchor])
                if valid_pair[qi, ti] and visible[anchor]:
                    teacher['visible_anchor_pairs_with_valid_track_majority'] += 1
                    teacher['anchor_truth_differs_from_valid_track_majority'] += int(bool(truth[qi, anchor]) != bool(majority[qi, ti]))
            for name, groups in errors.items():
                mask = np.load(OBSERVATION / 'seed0' / pair / name / 'videos' / video / 'mask.npy')
                selections = dict(stable_membership=visible[None, :] & ~mixed_rows,
                    mixed_membership=visible[None, :] & mixed_rows,
                    unmatched_union=np.broadcast_to(~visible, truth.shape))
                for scope, selected in selections.items():
                    groups[scope].update(TP=int((mask & truth & selected).sum()),
                        FP=int((mask & ~truth & selected).sum()),
                        FN=int((~mask & truth & selected).sum()), rows=int(selected.sum()))
        # Secondary point diagnostics on input-defined query groups. The
        # official HOTA combiner is retained; these are not independent tests
        # or an additive decomposition of the full-dataset HOTA.
        import csv
        from flex_uda.a100.evaluation_reuse import metrics
        from scripts.analyze_pf_video_uncertainty import unpack
        hota = metrics()[0]
        grouped = {}
        for group, queries in query_groups.items():
            wanted = {q['video_id']+'+'+q['expression_id'] for q in queries}
            grouped[group] = dict(queries=len(queries), parents=len({q['video_id'].rsplit('_c', 1)[0] for q in queries}), HOTA={})
            if not wanted:
                continue
            for name in errors:
                path = OBSERVATION / 'seed0' / pair / name / 'results/pedestrian_detailed.csv'
                with path.open() as handle:
                    rows = {r['seq']: unpack(r, hota) for r in csv.DictReader(handle) if r['seq'] in wanted}
                assert len(rows) == len(wanted)
                grouped[group]['HOTA'][name] = float(100*hota.combine_sequences(rows)['HOTA'].mean())
        result[pair] = dict(semantic_membership={k: dict(v) for k, v in totals.items()},
            physical_identity=dict(physical), teacher_anchor=dict(teacher),
            input_defined_query_groups_seed0=grouped,
            seed0_candidate_membership_errors={k: {s: dict(c) for s, c in v.items()} for k, v in errors.items()})
    report = dict(status='POST_SEAL_OBSERVATION_GRANULARITY_DIAGNOSIS', pairs=result,
        target_GT_only_in_this_post_seal_evaluation=True, adaptation_uses_these_labels=False,
        teacher_agreement_is_candidate_IoU_diagnostic_not_pure_semantic_accuracy=True,
        error_counts_not_an_additive_official_HOTA_attribution=True,
        all_development_data=True, goal_status='ACTIVE')
    atomic_json(OUT / 'analysis/observation_granularity_diagnosis.json', report)
    atomic_json(ROOT / 'docs/a100_visual_classical_20261121/observation_granularity_diagnosis.json', report)
    print(json.dumps(dict(event='OBSERVATION_DIAGNOSIS_COMPLETE', pairs=result)), flush=True)


if __name__ == '__main__':
    diagnose()
