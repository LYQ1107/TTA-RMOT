"""Audit requested/actual views without changing any live learner.

All six prepaid binary libraries are read in an independent process. Only
request metadata and view/fallback fields are used: no probabilities, new
questions, fits, predictions, task metrics, or manual target annotations.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json, file_sha
from flex_uda.a100.mechanism import PAIRS, metadata, generation_guard
from flex_uda.active.native_binary_inputs import inventory, ANSWERS, POLICIES, TEACHER_SHA
from flex_uda.active.rsi_discrimination_feedback import paired_probe_reservation
from flex_uda.active.rsi_probe_conditioning import comparable_paid_probe
from scripts.a100_observation_teacher_pilot import key as request_key
from scripts.a100_six_direction import OUT as SIX, PUBLIC


def main():
    guard = generation_guard()
    rows, settings = [], {}
    for pair, (_, target) in PAIRS.items():
        total = Counter()
        requested, actual, transitions, pair_views = Counter(), Counter(), Counter(), Counter()
        for video in sorted(metadata(target)['videos']):
            # Reserve BEFORE opening reply files; this is the registered
            # metadata-only reservation, independent of the paid opinions.
            actions, quotas = inventory(pair, video)
            reservation = paired_probe_reservation(actions, quotas)
            answers = {}
            for policy in POLICIES:
                doc = json.loads((ANSWERS/'policy_answers'/pair/video/(policy+'_answers.json')).read_text())
                assert doc['GT_open_attempts'] == 0 and doc['only_this_policy_requested_responses']
                for reply in doc['answers']:
                    uid = hashlib.sha256(request_key(reply).encode()).hexdigest()
                    assert uid in actions and reply['frozen_weight_sha256'] == TEACHER_SHA
                    fields = (reply['effective_view'], bool(reply['clip_fallback']))
                    if uid in answers:
                        assert answers[uid] == fields
                    answers[uid] = fields
            assert set(answers) == set(actions)
            counts = Counter(actions=len(actions), observer_budget=sum(quotas.values()),
                probe_pairs=sum(map(len, reservation['pairs'])),
                fresh_probe_rounds=reservation['fresh_probe_rounds'],
                changed_requested_view=0, clip_fallbacks=0,
                probe_effective_view_mismatches=0, probe_either_changed_view=0,
                probe_different_frames=0)
            for uid, action in actions.items():
                view = action['case']['view']
                effective, fallback = answers[uid]
                requested[view] += 1
                actual[effective] += 1
                transitions[view+' -> '+effective] += 1
                counts['changed_requested_view'] += view != effective
                counts['clip_fallbacks'] += fallback
            for probes in reservation['pairs']:
                paid_metadata = [dict(actions[u], effective_view=answers[u][0])
                                 for pair_uids in probes for u in pair_uids]
                eligible, audit = comparable_paid_probe(paid_metadata)
                assert audit['paid_probe_nodes'] == 2*len(probes)
                assert len(eligible) == 2*(len(probes)-audit['effective_view_mismatch_pairs'])
                if not audit['effective_view_mismatch_pairs']:
                    assert eligible is paid_metadata
                for first, second in probes:
                    a, b = actions[first], actions[second]
                    assert a['local_query'] == b['local_query']
                    assert a['local_track'] != b['local_track']
                    assert a['case']['view'] == b['case']['view']
                    av, bv = answers[first][0], answers[second][0]
                    counts['probe_effective_view_mismatches'] += av != bv
                    counts['probe_either_changed_view'] += (
                        av != a['case']['view'] or bv != b['case']['view'])
                    counts['probe_different_frames'] += a['case']['frame_id'] != b['case']['frame_id']
                    pair_views[av+' / '+bv] += 1
            total.update(counts)
            rows.append(dict(pair=pair, video=video, **counts))
        settings[pair] = dict(videos=sum(r['pair'] == pair for r in rows), **total,
            requested_views=dict(requested), effective_views=dict(actual),
            requested_to_effective=dict(transitions), reserved_probe_effective_pairs=dict(pair_views))
        print(json.dumps(dict(completed_setting=pair, summary=settings[pair])), flush=True)
    assert len(rows) == 148 and not guard
    detail = SIX/'native_binary_effective_views_detail_20261124.json'
    atomic_json(detail, dict(per_setting_video=rows))
    result = dict(status='COMPLETED_ALL_SIX_NATIVE_BINARY_VIEW_METADATA_AUDIT',
        settings=settings, settings_videos=len(rows),
        possible_actions=sum(r['actions'] for r in rows),
        per_observer_budget_all_settings=sum(r['observer_budget'] for r in rows),
        reserved_probe_pairs=sum(r['probe_pairs'] for r in rows),
        probe_effective_view_mismatches=sum(r['probe_effective_view_mismatches'] for r in rows),
        prospective_comparable_probe_pairs=sum(r['probe_pairs']-r['probe_effective_view_mismatches'] for r in rows),
        prospective_conditioning_identity_on_unaffected_pairs_verified=True,
        requested_actual_changes=sum(r['changed_requested_view'] for r in rows),
        local_detail_path=str(detail), local_detail_sha256=file_sha(detail),
        source_sha256={name: file_sha(ROOT/name) for name in (
            'scripts/a100_native_binary_effective_views.py',
            'flex_uda/active/rsi_probe_conditioning.py',
            'flex_uda/active/native_binary_inputs.py',
            'flex_uda/active/rsi_discrimination_feedback.py')},
        probability_fields_used=False, manual_GT_open_attempts=0,
        new_teacher_calls=0, new_fits=0, new_RMOT_predictions=0,
        new_RMOT_quality_read=False, live_sources_or_policy_choices_modified=False,
        interpretation='View consistency audits the conditioning assumption only. Different frames remain possible, and equal views do not establish independent errors, teacher correctness, or RMOT gains.')
    atomic_json(PUBLIC/'native_binary_effective_views_20261124.json', result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
