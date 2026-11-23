"""Post-seal I1 diagnosis: paid evidence, coarsening, and fitted outputs.

Uses only the completed 828-output observation-policy study. Canonical
candidate membership is an offline diagnostic, not pure teacher semantic
accuracy or official per-method matching. No adaptation consumes this file.
"""
from collections import Counter, defaultdict
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from flex_tta.runtime.pf_receipts import atomic_json, file_sha
from flex_uda.a100.mechanism import PAIRS, metadata, OUT as DIAGNOSTIC
from scripts.a100_six_direction import PUBLIC, SEEDS, OUT as SIX
from scripts.a100_observation_design_collect import OUT as ANSWERS, TEACHER_SHA
from scripts.a100_observation_design_quality import OUT as QUALITY, POLICIES, conditions

NAME = 'observation_evidence_transfer_diagnosis'
PLAN = PUBLIC/(NAME+'_plan.json')
CANONICAL = {'C1':'C1', 'C2':'C5', 'C3':'C4', 'C4':'C4', 'C5':'C5', 'C6':'C1'}
PATTERNS = tuple(f'{i:03b}' for i in range(8))


def confusion(truth, decision):
    truth, decision = np.asarray(truth, bool), np.asarray(decision, bool)
    assert truth.shape == decision.shape
    return Counter(TP=int((truth & decision).sum()), FP=int((~truth & decision).sum()),
        FN=int((truth & ~decision).sum()), TN=int((~truth & ~decision).sum()))


def rates(row):
    def ratio(n, d): return n/d if d else None
    recall = ratio(row['TP'], row['TP']+row['FN'])
    specificity = ratio(row['TN'], row['TN']+row['FP'])
    return dict(positive_recall=recall, negative_specificity=specificity,
        positive_precision=ratio(row['TP'], row['TP']+row['FP']),
        balanced_agreement=(recall+specificity)/2 if recall is not None and specificity is not None else None)


def agreement_record(row):
    return dict(confusion={k:int(row[k]) for k in ('TP', 'FP', 'FN', 'TN')}, rates=rates(row))


def _counts_only(value):
    if isinstance(value, dict):
        return {k:_counts_only(v) for k,v in value.items() if not (k == 'rates' and 'confusion' in value)}
    if isinstance(value, list): return [_counts_only(v) for v in value]
    return value


def _restore_rates(value):
    if isinstance(value, list): return [_restore_rates(v) for v in value]
    if not isinstance(value, dict): return value
    row = {k:_restore_rates(v) for k,v in value.items()}
    if 'confusion' in row and 'rates' not in row: row['rates'] = rates(row['confusion'])
    return row


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def expand_public(payload):
    payload = dict(payload)
    proof = payload.pop('publication_encoding')
    assert proof['kind'] == 'INTEGER_COUNTS_WITH_EXACTLY_DERIVABLE_RATES_V1'
    expanded = _restore_rates(payload)
    assert hashlib.sha256(_canonical(expanded)).hexdigest() == proof['expanded_canonical_sha256']
    return expanded


def publish(result):
    public = _counts_only(result)
    public['publication_encoding'] = dict(kind='INTEGER_COUNTS_WITH_EXACTLY_DERIVABLE_RATES_V1',
        expanded_canonical_sha256=hashlib.sha256(_canonical(result)).hexdigest(),
        note='All confusion counts and paired patterns retained. Recompute omitted redundant rates by this script --expand-public; no raw QA or GT rows published.')
    assert _canonical(expand_public(public)) == _canonical(result)
    (PUBLIC/(NAME+'.json')).write_text(json.dumps(public, separators=(',', ':'))+'\n')


def main():
    start = time.perf_counter()
    plan = json.loads(PLAN.read_text())
    assert plan['posthoc_completed_I1_only'] and file_sha(Path(__file__)) == plan['source_sha256']
    assert json.loads((QUALITY/'all_predictions_sealed_before_GT.json').read_text())['methods'] == 828
    assert json.loads((QUALITY/'official.json').read_text())['methods'] == 828
    assert (QUALITY/'analysis.json').is_file()
    for condition in conditions():
        root = Path(condition['folder'])
        for folder in (root, root/'countmatched'):
            assert json.loads((folder/'completion.json').read_text())['GT_open_attempts'] == 0
    settings = {}
    for pair, (_, target) in PAIRS.items():
        md = metadata(target)
        variability = Counter()
        teacher = {policy:defaultdict(Counter) for policy in POLICIES}
        # Pairwise correctness at the identical nominal anchor: teacher,
        # compressed own-bag mean, and learned prediction, in that order.
        transfer = {policy:{str(seed):{decoder:defaultdict(Counter) for decoder in ('FINAL', 'MEAN')}
            for seed in SEEDS} for policy in POLICIES}
        dense = {policy:{str(seed):{decoder:defaultdict(Counter) for decoder in ('FINAL', 'MEAN')}
            for seed in SEEDS} for policy in POLICIES}
        policy_counts = {policy:Counter() for policy in POLICIES}
        for video, bank in sorted(md['videos'].items()):
            queries = [q for q in md['episodes'] if q['video_id'] == video]
            obs = np.asarray(bank['observations'], int)
            truth = np.load(DIAGNOSTIC/'offline_GT'/CANONICAL[pair]/video/'labels.npy') >= 0
            assert truth.shape == (len(queries), len(obs))
            tracks, columns = np.unique(obs[:, 1], return_inverse=True)
            lengths = np.bincount(columns)
            positive = np.stack([np.bincount(columns, weights=row, minlength=len(tracks)) for row in truth]).astype('int64')
            negative = lengths[None, :]-positive
            mixed = (positive > 0) & (negative > 0)
            variability.update(query_track_bags=int(positive.size), bags_with_positive_rows=int((positive > 0).sum()),
                membership_mixed_bags=int(mixed.sum()), canonical_positive_rows=int(positive.sum()),
                canonical_positive_rows_in_mixed_bags=int(positive[mixed].sum()),
                candidate_query_rows=int(truth.size),
                minimum_row_errors_for_any_bag_constant_binary_rule=int(np.minimum(positive, negative).sum()))
            positions = {tuple(row):i for i, row in enumerate(obs.tolist())}
            saved = {}
            for seed in SEEDS:
                with np.load(QUALITY/pair/f'seed{seed}'/'videos'/video/'coarse_scores.npz') as a:
                    assert np.array_equal(a['tracks'], tracks)
                    saved[seed] = {name:a[name].copy() for name in a.files if name != 'tracks'}
            for policy in POLICIES:
                doc = json.loads((ANSWERS/'policy_answers'/pair/video/(policy+'_answers.json')).read_text())
                assert doc['only_this_policy_requested_responses'] and doc['GT_open_attempts'] == 0
                seen = {}
                for row in doc['answers']:
                    assert row['frozen_weight_sha256'] == TEACHER_SHA
                    if row['request_id'] in seen:
                        assert seen[row['request_id']]['yes_probability'] == row['yes_probability']
                    else:
                        seen[row['request_id']] = row
                records = list(seen.values())
                qi = np.asarray([int(r['local_query']) for r in records])
                ti = np.asarray([int(r['local_track']) for r in records])
                anchors = np.asarray([positions[int(r['frame_id']), int(r['track_id'])] for r in records])
                assert all(queries[q]['expression_id'] == r['query_id'] and int(tracks[t]) == r['track_id']
                    for q, t, r in zip(qi, ti, records))
                raw = np.asarray([float(r['yes_probability']) for r in records])
                label = truth[qi, anchors]
                views = np.asarray([r['effective_view'] for r in records])
                grouped = defaultdict(list)
                for q, t, p in zip(qi, ti, raw): grouped[int(q), int(t)].append(float(p))
                means = {cell:float(np.mean(values)) for cell, values in grouped.items()}
                exact_coarse = np.asarray([means[int(q), int(t)] for q, t in zip(qi, ti)])
                coarse = exact_coarse.astype(saved[SEEDS[0]][policy+'_TEACHER_FILL'].dtype)
                teacher_correct = (raw > .5) == label
                mean_correct = (coarse > .5) == label
                known = np.zeros(positive.shape, bool)
                for q, t in means: known[q, t] = True
                policy_counts[policy].update(logical_requests=doc['logical_requested'], unique_requests=len(records),
                    raw_probability_exact_half=int((raw == .5).sum()), coarse_probability_exact_half=int((coarse == .5).sum()),
                    stored_mean_rounding_changes_decision_requests=int(((coarse > .5) != (exact_coarse > .5)).sum()),
                    asked_query_track_bags=int(known.sum()),
                    asked_canonical_positive_bags=int(((positive > 0) & known).sum()))
                strata = {'ALL_NOMINAL_ANCHORS':np.ones(len(records), bool)}
                strata.update({view:views == view for view in sorted(set(views))})
                for view, selected in strata.items():
                    teacher[policy][view].update(confusion(label[selected], raw[selected] > .5))
                for seed in SEEDS:
                    fill = saved[seed][policy+'_TEACHER_FILL']
                    # The original fill output stores these same means in
                    # its float32 score array; it must not supply new data.
                    for (q, t), p in means.items():
                        assert fill[q, t] == np.asarray(p, dtype=fill.dtype), (pair, video, policy, seed)
                    for decoder in ('FINAL', 'MEAN'):
                        score = saved[seed][policy+'_'+decoder]
                        assert score.shape == positive.shape
                        final_correct = (score[qi, ti] > .5) == label
                        code = teacher_correct.astype(int)*4+mean_correct.astype(int)*2+final_correct.astype(int)
                        for view, selected in strata.items():
                            row = transfer[policy][str(seed)][decoder][view]
                            row.update(confusion(label[selected], score[qi[selected], ti[selected]] > .5))
                            for i, count in enumerate(np.bincount(code[selected], minlength=8)):
                                row['correctness_'+PATTERNS[i]] += int(count)
                        decision = score > .5
                        for name, selected in (('ASKED_BAGS', known), ('UNASKED_BAGS', ~known)):
                            row = dense[policy][str(seed)][decoder][name]
                            row.update(TP=int(positive[selected & decision].sum()), FP=int(negative[selected & decision].sum()),
                                FN=int(positive[selected & ~decision].sum()), TN=int(negative[selected & ~decision].sum()))
                assert all(sum(row['correctness_'+p] for p in PATTERNS) == sum(row[k] for k in ('TP','FP','FN','TN'))
                    for seed in SEEDS for decoder in ('FINAL','MEAN')
                    for row in transfer[policy][str(seed)][decoder].values())
        policies = {}
        for policy in POLICIES:
            policies[policy] = dict(counts=dict(policy_counts[policy]),
                teacher_nominal_anchor={view:agreement_record(row) for view, row in teacher[policy].items()},
                per_seed={seed:{decoder:dict(
                    nominal_anchor={view:dict(**agreement_record(row),
                        correctness_teacher_coarse_learned=[row['correctness_'+p] for p in PATTERNS])
                        for view,row in transfer[policy][seed][decoder].items()},
                    dense_rows={name:agreement_record(row) for name,row in dense[policy][seed][decoder].items()})
                    for decoder in ('FINAL','MEAN')} for seed in map(str, SEEDS)})
        settings[pair] = dict(membership_variability=dict(variability), policies=policies)
        print(json.dumps(dict(event='COMPLETED_I1_EVIDENCE_TRANSFER_SETTING',pair=pair,
            constant_binary_row_error_floor=variability['minimum_row_errors_for_any_bag_constant_binary_rule']/variability['candidate_query_rows'])),flush=True)
    result = dict(status='PASS_POSTSEAL_I1_EVIDENCE_TRANSFER_DIAGNOSIS', settings=settings,
        correctness_pattern_order=list(PATTERNS),
        correctness_pattern_definition='Each bit is correctness against the same nominal-anchor canonical candidate membership: individual raw teacher, original stored float32 own-policy query-track mean, learned saved output. Any threshold changes from mean storage rounding are counted separately.',
        decision_rule='All diagnostic probabilities use >0.5, matching original plain decoder; exact-half counts explicit, not confident Yes labels.',
        interpretation=[
            'Descriptive paired transitions, not independent samples or causal HOTA contributions.',
            'Candidate IoU membership and one-to-one canonical matching are not pure teacher semantic accuracy.',
            'CROP/FRAME_BOX are reported separately; CLIP is nominal-anchor only and can see other frames.',
            'Any bag-constant binary output has at least sum_b min(positive_rows_b,negative_rows_b) canonical row errors. This is not an HOTA upper bound or proof that a state-aware model improves.',
            'Asked/unasked dense groups describe each policy own chosen bags, not randomized comparable groups.',
            'KITTI uses prior order-verified complete-bank numerical membership, not a recovered physical-ID oracle.',
            'No current conditional-study prediction or target evaluation labels read; no policy, model or parameter changed.'
        ],
        source_study_all828_sealed_and_official_complete=True, posthoc_development_diagnosis=True,
        all_GT_access_only_in_this_offline_diagnostic=True, adaptation_consumes_these_labels=False,
        new_teacher_student_or_model_calls=0, wall_seconds=time.perf_counter()-start, goal_status='ACTIVE')
    atomic_json(SIX/'analysis'/(NAME+'.json'), result)
    publish(result)
    print(json.dumps(dict(status=result['status'],wall_seconds=result['wall_seconds'],settings=len(settings))),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expand-public', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.expand_public:
        assert args.output, '--output is required for expansion'
        expanded = expand_public(json.loads(args.expand_public.read_text()))
        atomic_json(args.output, expanded)
        print(json.dumps(dict(status='PASS_EXACT_PUBLIC_COUNTS_EXPANSION', settings=len(expanded['settings']))))
    else:
        assert not args.output
        main()
