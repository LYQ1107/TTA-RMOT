"""Unlabelled coverage of the finite four-policy RSI action library.

Only original request metadata and supplied candidate anchors are read.
Unavailable-to-question tracks can still receive dense model predictions;
no unavailable track is called a missing true target without GT evaluation.
"""
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from flex_tta.runtime.pf_receipts import atomic_json, file_sha
from flex_uda.a100.mechanism import PAIRS, metadata, generation_guard
from flex_uda.active.native_binary_inputs import inventory
from scripts.a100_anchor_observability import OUT as ANCHORS
from scripts.a100_six_direction import OUT as SIX, PUBLIC


def main():
    guard = generation_guard(); detail = []; settings = {}; units = 0
    for pair, (_, target) in PAIRS.items():
        totals = Counter(); fractions = []; slots = []; videos = 0
        for video in sorted(metadata(target)['videos']):
            actions, quotas = inventory(pair, video)
            with np.load(ANCHORS/target/video/'anchor_histograms.npz') as archive:
                tracks, present = archive['tracks'], archive['present']
            nt, nq = len(tracks), len(quotas)
            assert np.all(present.sum(1) > 0) and present.shape == (nt, 3)
            available = {q: set() for q in quotas}
            states = Counter(); views = {q: set() for q in quotas}
            for action in actions.values():
                q, t = action['local_query'], action['local_track']
                assert int(tracks[t]) == action['case']['track_id']
                available[q].add(t); states[q] += 1; views[q].add(action['case']['view'])
            assert all(len(v) == 1 for v in views.values())
            possible_slots = int(present.sum())
            for q, budget in sorted(quotas.items()):
                count = len(available[q]); fraction = count/nt
                row = dict(pair=pair, video=video, local_query=q, candidate_tracks=nt,
                    question_available_tracks=count, unavailable_to_question_tracks=nt-count,
                    per_query_budget=budget, available_anchor_view_actions=states[q],
                    all_candidate_anchor_actions=possible_slots,
                    question_track_coverage=fraction, action_slot_coverage=states[q]/possible_slots,
                    budget_can_visit_all_tracks=budget >= nt,
                    budget_supports_more_distinct_tracks_than_library=count < min(budget, nt))
                detail.append(row); fractions.append(fraction); slots.append(row['action_slot_coverage'])
                totals.update(query_episodes=1, candidate_query_track_pairs=nt,
                    available_query_track_pairs=count, unavailable_query_track_pairs=nt-count,
                    full_candidate_anchor_actions=possible_slots, library_actions=states[q],
                    full_track_coverage_queries=count == nt,
                    incomplete_track_coverage_queries=count < nt,
                    fewer_than_two_available_tracks_queries=count < 2,
                    budget_covers_all_tracks_queries=budget >= nt,
                    budget_supports_more_distinct_tracks_than_library_queries=count < min(budget, nt),
                    observer_paid_budget=budget)
            videos += 1; units += 1
        settings[pair] = dict(videos=videos, **totals,
            macro_query_track_coverage=float(np.mean(fractions)), minimum_query_track_coverage=float(min(fractions)),
            median_query_track_coverage=float(np.median(fractions)),
            weighted_query_track_coverage=totals['available_query_track_pairs']/totals['candidate_query_track_pairs'],
            macro_query_action_slot_coverage=float(np.mean(slots)))
        print(json.dumps(dict(completed_setting=pair, summary=settings[pair])), flush=True)
    assert units == 148 and not guard
    local = SIX/'native_binary_action_universe_detail_20261124.json'
    atomic_json(local, dict(per_query=detail))
    totals = {key: sum(setting[key] for setting in settings.values()) for key in (
        'query_episodes', 'candidate_query_track_pairs', 'available_query_track_pairs',
        'unavailable_query_track_pairs', 'full_candidate_anchor_actions', 'library_actions',
        'incomplete_track_coverage_queries', 'budget_supports_more_distinct_tracks_than_library_queries',
        'observer_paid_budget')}
    result = dict(status='COMPLETED_UNLABELLED_RSI_FINITE_ACTION_UNIVERSE_AUDIT',
        settings=settings, totals=totals, settings_videos=units,
        full_candidate_anchor_actions_definition='All supplied candidate tracks at their existing up-to-three real anchors, with the same single query-routed requested view.',
        local_detail_path=str(local), local_detail_sha256=file_sha(local),
        source_sha256={name: file_sha(ROOT/name) for name in (
            'scripts/a100_native_binary_action_universe.py', 'flex_uda/active/native_binary_inputs.py')},
        teacher_answers_or_probabilities_read=False, manual_GT_open_attempts=0,
        new_teacher_calls=0, new_fits=0, new_predictions=0, new_HOTA_read=False,
        current_live_sources_or_action_spaces_modified=False,
        unavailable_to_question_does_not_mean_unavailable_to_dense_prediction=True,
        question_coverage_is_not_true_target_recall=True,
        all_settings_share_physical_inputs_not_independent_samples=True,
        implication='A future fresh deployment must account for constructing its action pool from fresh inputs. If broadening the pool, ordinary comparators need the same pool, paid budgets, and fresh quality outputs; library expansion alone is not RSI learning evidence.',
        goal_status='ACTIVE')
    atomic_json(PUBLIC/'native_binary_action_universe_20261124.json', result)
    print(json.dumps(dict(status=result['status'], totals=totals)), flush=True)


if __name__ == '__main__':
    main()
