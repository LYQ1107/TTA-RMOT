"""Audit actual paid supervision at fixed completed prefixes, without GT.

Reconstruct the exact NativeVisual.record node grouping and round-robin
pairing using metadata only. No feature reads, forward passes, fits, future
unrequested answers or changed predictions. Information is a teacher-target
quantity, not true semantic information or an accuracy measurement.
"""
from collections import Counter, defaultdict
from itertools import zip_longest
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from flex_tta.runtime.pf_receipts import atomic_json, file_sha
from flex_uda.a100.mechanism import metadata, generation_guard
from scripts.a100_native_binary_utility_six_setting import PAIRS, PUBLIC, OUT, source_root


def reconstructed_pairs(paid):
    nodes = defaultdict(list)
    for row in paid:
        case = row['case']
        nodes[int(row['local_query']), int(row['local_track']), int(case['frame_id'])].append(row)
    keys = sorted(nodes); grouped = defaultdict(lambda: defaultdict(list))
    for i, (query, track, _) in enumerate(keys):
        grouped[query][track].append(i)
    pairs = []
    for query, tracks in sorted(grouped.items()):
        order = [x for layer in zip_longest(*(tracks[t] for t in sorted(tracks))) for x in layer if x is not None]
        pairs.extend(zip(order, order[1:]+order[:1]))
    p = np.asarray([np.mean([r['probability'] for r in nodes[k]]) for k in keys], dtype='float64')
    actual_views = [frozenset(r['effective_view'] for r in nodes[k]) for k in keys]
    return keys, grouped, pairs, p, actual_views


def quantities(first, second, same_event=False):
    if same_event:
        # The original record uses [0,0,p,1-p], not two independent copies.
        return 0., 0., 0.
    mass = first*(1-second)+second*(1-first)
    relative = first*(1-second)/mass if mass else .5
    entropy = -(relative*np.log(max(relative, 1e-300))+(1-relative)*np.log(max(1-relative, 1e-300)))
    information = max(mass*(np.log(2.)-entropy), 0.)
    return float(mass), float(information), float(abs(first-second))


def audit_paid(paid):
    keys, groups, pairs, probabilities, views = reconstructed_pairs(paid)
    count = Counter(paid_requests=len(paid), visual_nodes=len(keys), pair_rows=len(pairs), queries=len(groups))
    count['queries_with_only_one_paid_track'] = sum(len(tracks) == 1 for tracks in groups.values())
    exposure = Counter(len(ids) for tracks in groups.values() for ids in tracks.values())
    count['queried_query_track_groups'] = sum(exposure.values())
    count['query_track_groups_with_multiple_requested_moments'] = sum(n for k, n in exposure.items() if k > 1)
    info = Counter(); gaps = []; margins = []; by_query = defaultdict(list)
    for i, j in pairs:
        qa, ta, fa = keys[i]; qb, tb, fb = keys[j]; assert qa == qb
        kind = 'SAME_EVENT' if i == j else ('CROSS_TRACK' if ta != tb else 'SAME_TRACK_DIFFERENT_TIME')
        same_view = len(views[i]) == len(views[j]) == 1 and views[i] == views[j]
        frame_equal = fa == fb
        mass, amount, margin = quantities(probabilities[i], probabilities[j], i == j)
        by_query[qa].append(amount)
        count[kind] += 1
        count[kind+'_same_effective_view'] += same_view
        count[kind+'_same_frame'] += frame_equal
        count[kind+'_same_view_and_frame'] += same_view and frame_equal
        count[kind+'_both_teacher_probabilities_below_half'] += int(probabilities[i] < .5 and probabilities[j] < .5)
        count[kind+'_exactly_zero_teacher_probability_difference'] += int(probabilities[i] == probabilities[j])
        info[kind+'_exactly_one_mass_sum'] += mass
        info[kind+'_weighted_conditional_information_sum'] += amount
        info[kind+'_absolute_probability_difference_sum'] += margin
        if kind == 'CROSS_TRACK':
            gaps.append(abs(fa-fb)); margins.append(margin)
            if not same_view:
                info['cross_track_mismatched_view_information_sum'] += amount
            if not frame_equal:
                info['cross_track_different_frame_information_sum'] += amount
    assert sum(count[k] for k in ('SAME_EVENT', 'CROSS_TRACK', 'SAME_TRACK_DIFFERENT_TIME')) == len(keys)
    info['conditional_information_macro_query_sum'] = sum(float(np.mean(v)) for v in by_query.values())
    return dict(counts=dict(count), sums=dict(info),
        cross_track_frame_gaps=gaps, cross_track_probability_differences=margins,
        requested_moments_per_query_track_histogram=dict(exposure))


def main():
    guard = generation_guard(); settings = {}; details = []; signatures = []
    for pair, (_, target) in PAIRS.items():
        md = metadata(target); videos = sorted(md['videos'])[:10]
        root = source_root(pair); policies = {}
        for policy in ('LEARNED', 'FROZEN'):
            counts = Counter(); sums = Counter(); gaps = []; margins = []; exposure = Counter()
            probe_info = []; progress = []; probed_queries = 0
            for video in videos:
                local = root/'videos'/video
                receipt = json.loads((local/'receipt.json').read_text()); assert receipt['GT_open_attempts'] == 0
                path = local/'policies'/policy/'paid.json'
                doc = json.loads(path.read_text()); assert doc['GT_open_attempts'] == 0
                paid = doc['paid']; budget = receipt['policy_budgets'][policy]
                assert len(paid) == budget['unique_teacher_bundles']
                row = audit_paid(paid); counts.update(row['counts']); sums.update(row['sums'])
                reported = receipt['fits']['FULL_CONDITIONAL_REFERENCE_'+policy]['final_all_paid_fit']['original_risk_rows']
                assert reported == counts['visual_nodes'], (pair, video, policy, reported, counts['visual_nodes'])
                exposure.update(row['requested_moments_per_query_track_histogram'])
                sequence = json.loads((local/'policies'/policy/'sequence.json').read_text())['sequence']
                lookup = {r['uid']: r for r in paid}
                probed_queries += len({lookup[e['uid']]['local_query'] for e in sequence if e['role'] == 'probe'})
                for stage in receipt['fits']['FULL_CONDITIONAL_REFERENCE_'+policy]['rounds']:
                    if stage['credit']['information'] is not None:
                        probe_info.append(stage['credit']['information'])
                    progress.append(stage['credit']['reward'])
                gaps.extend(row.pop('cross_track_frame_gaps')); margins.extend(row.pop('cross_track_probability_differences'))
                details.append(dict(pair=pair, video=video, policy=policy, **row))
                signatures.append(dict(pair=pair, video=video, policy=policy, paid_sha256=file_sha(path),
                    completed_prefix_receipt_sha256=file_sha(local/'receipt.json')))
            cross = counts['CROSS_TRACK']; all_info = sum(sums[k+'_weighted_conditional_information_sum']
                for k in ('CROSS_TRACK', 'SAME_TRACK_DIFFERENT_TIME', 'SAME_EVENT'))
            policies[policy] = dict(counts=dict(counts), sums=dict(sums),
                requested_moments_per_query_track_histogram=dict(exposure),
                all_paid_macro_query_information_mean=sums['conditional_information_macro_query_sum']/counts['queries'],
                logged_probe_nonempty_rounds=len(probe_info),
                logged_probe_information_mean=float(np.mean(probe_info)) if probe_info else None,
                logged_probe_rounds_above_original_resolution=sum(v >= .005 for v in probe_info),
                logged_policy_rounds=len(progress), positive_progress_rounds=sum(v > 0 for v in progress),
                negative_progress_rounds=sum(v < 0 for v in progress),
                query_video_episodes_ever_probed=probed_queries,
                cross_track_fraction=cross/counts['pair_rows'],
                cross_track_same_view_and_frame_fraction=counts['CROSS_TRACK_same_view_and_frame']/cross if cross else None,
                cross_track_information_fraction=sums['CROSS_TRACK_weighted_conditional_information_sum']/all_info if all_info else None,
                cross_track_information_mean=sums['CROSS_TRACK_weighted_conditional_information_sum']/cross if cross else None,
                cross_track_gap_median_and_p90=np.percentile(gaps, [50, 90]).tolist() if gaps else None,
                cross_track_teacher_difference_median_and_p90=np.percentile(margins, [50, 90]).tolist() if margins else None)
        settings[pair] = dict(videos=videos, source_root=str(root), policies=policies)
        print(json.dumps(dict(pair=pair, videos=len(videos), policies={p: dict(
            cross_track_fraction=r['cross_track_fraction'],
            comparable_frame_fraction=r['cross_track_same_view_and_frame_fraction'],
            cross_track_information_mean=r['cross_track_information_mean']) for p, r in policies.items()})), flush=True)
    assert not guard
    destination = OUT.parent/'paid_pair_information_20261124'
    atomic_json(destination/'details.json', dict(rows=details, source_signatures=signatures))
    result = dict(status='COMPLETED_ACTUAL_PAID_PAIR_INFORMATION_PREFIX_AUDIT_NO_GT', settings=settings,
        scope='First min(10,N) sorted completed videos of each of six settings; learned/frozen; corrected C2/C5 and exact Trust other four.',
        pair_rule='Exact metadata reconstruction of NativeVisual.record: mean at identical q,track,frame; round-robin across sorted tracks and time layers.',
        information_definition='m*(ln2-H(r)), m=pA*(1-pB)+pB*(1-pA), r=pA*(1-pB)/m; sum is target discrimination relative to equal candidate utility, not true semantics.',
        stable_identity_gradient_property='With the current conditional=True and independent count encoder, the identity term cancels in same-track utility differences; singleton same-event has zero conditional mass. Cross-track comparisons are the direct relative supervision of stable identity.',
        same_view_and_frame_is_only_metadata_condition_not_teacher_independence_or_correctness=True,
        requested_moment_histogram_does_not_count_the_internal_past_frames_of_one_CLIP=True,
        zero_revisits_is_not_zero_temporal_visual_input_or_proof_revisiting_would_help=True,
        probe_log_risk_is_macro_query_per_round_and_all_paid_mean_uses_all_query_video_episodes=True,
        no_native_quality_read_or_filter_threshold_or_method_change=True,
        manual_GT_open_attempts=0, new_teacher_calls=0, new_fits=0, new_forwards=0,
        new_predictions=0, unpurchased_answers_read=0, current_source_pins_unchanged=True,
        local_details=str(destination/'details.json'), local_details_sha256=file_sha(destination/'details.json'),
        source_sha256=file_sha(Path(__file__)), goal_status='ACTIVE')
    atomic_json(PUBLIC/'native_binary_paid_pair_information_20261124.json', result)
    print(json.dumps(dict(status=result['status'], completed_setting_videos=sum(len(r['videos']) for r in settings.values()))), flush=True)


if __name__ == '__main__':
    main()
