"""Metadata/full-space and recorded-reply wiring checks, never a GPU trial."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json, file_sha
from flex_uda.a100.mechanism import PAIRS, metadata, generation_guard
from flex_uda.active.native_binary_inputs import inventory
from flex_uda.active.native_binary_live_inputs import full_action_inventory, LiveBinaryBroker
from flex_uda.active.rsi_discrimination_feedback import paired_probe_reservation
from scripts.a100_observation_teacher_pilot import key as request_key
from scripts.a100_six_direction import OUT as SIX, PUBLIC


class RecordedReplyFixture:
    """Replay already-requested replies only; no inference/cache timing claim."""
    def __init__(self, paid, cached_fallback=False):
        self.paid = paid; self.cached_fallback = cached_fallback
        self.stats = dict(logical_calls=0, new_calls=0, cache_hits=0,
            forward_seconds=0., equivalent_forward_seconds=0., batches=0)

    def ask_many(self, requests):
        result = []
        for request in requests:
            uid = hashlib.sha256(request_key(request).encode()).hexdigest()
            row = self.paid[uid]
            result.append(dict(request, yes_probability=row['probability'],
                request_id=row['teacher_request_id'], frozen_weight_sha256=row['frozen_weight_sha256'],
                effective_view=row['effective_view'], requested_view=request['view']))
            if self.cached_fallback:
                assert request['view'] == 'CLIP' and row['effective_view'] == 'CROP'
                result[-1]['view'] = 'CROP'
            self.stats['logical_calls'] += 1; self.stats['cache_hits'] += 1
        return result


def main():
    guard = generation_guard(); opened_answer_libraries = []
    def exclude_library(event, args):
        if event == 'open' and isinstance(args[0], (str, bytes)):
            path = str(args[0])
            if '/policy_answers/' in path:
                opened_answer_libraries.append(path)
                raise PermissionError('Live contract cannot open a historical answer library')
    sys.addaudithook(exclude_library)
    totals = Counter(); settings = {}
    for pair, (_, target) in PAIRS.items():
        counts = Counter()
        for video in sorted(metadata(target)['videos']):
            old, _ = inventory(pair, video)
            routed = full_action_inventory(pair, video, view_mode='routed')
            three = full_action_inventory(pair, video, view_mode='three')
            assert set(old) <= set(routed) <= set(three) and len(three) == 3*len(routed)
            assert all(old[u]['local_query'] == routed[u]['local_query'] and
                old[u]['local_track'] == routed[u]['local_track'] for u in old)
            views = {}
            for action in three.values():
                views.setdefault(action['local_query'], set()).add(action['case']['view'])
            assert all(v == {'CROP', 'CLIP', 'FRAME_BOX'} for v in views.values())
            counts.update(videos=1, old_library_actions=len(old), full_routed_actions=len(routed),
                full_three_view_actions=len(three))
        settings[pair] = dict(counts); totals.update(counts)
        print(json.dumps(dict(completed_setting=pair, counts=dict(counts))), flush=True)
    assert totals['videos'] == 148 and totals['old_library_actions'] == 121629
    assert totals['full_routed_actions'] == 559386 and totals['full_three_view_actions'] == 1678158
    # The already completed first C1 policy supplies an exact API round trip.
    folder = SIX/'native_binary_quality_20261124/C1/seed0/videos/dancetrack0004/policies/LEARNED'
    paid = {r['uid']: r for r in json.loads((folder/'paid.json').read_text())['paid']}
    sequence = json.loads((folder/'sequence.json').read_text())['sequence']
    actions, quotas = inventory('C1', 'dancetrack0004')
    fixture = RecordedReplyFixture(paid)
    broker = LiveBinaryBroker('C1', 'dancetrack0004', actions, quotas,
        paired_probe_reservation(actions, quotas), fixture,
        ledger_path=SIX/'live_broker_contract_service_20261124.jsonl', backend_kind='recorded_contract_fixture')
    for stage in range(3):
        for role in ('train', 'probe'):
            uids = [e['uid'] for e in sequence if e['stage'] == stage and e['role'] == role]
            assert broker.request(uids, role, stage) == [paid[u] for u in uids]
    receipt = broker.receipt(True)
    assert broker.sequence == sequence and len(broker.received) == len(paid) == 56
    before = dict(fixture.stats)
    for invalid in ([sequence[0]['uid']], ['not-a-request']):
        try:
            broker.request(invalid, 'train', 0)
        except AssertionError:
            pass
        else:
            raise AssertionError('Invalid teacher request was not rejected')
    assert fixture.stats == before
    # Real CLIP-fallback metadata from an already completed C2 policy tests
    # the API cache convention: old response.view, current requested_view.
    path = SIX/'native_binary_comparable_quality_20261124/C2/seed0/videos/0005/policies/LEARNED/paid.json'
    fallback = next(r for r in json.loads(path.read_text())['paid']
        if r['case']['view'] == 'CLIP' and r['effective_view'] == 'CROP')
    uid = fallback['uid']; action = {k: fallback[k] for k in ('uid', 'local_query', 'local_track', 'case')}
    client = RecordedReplyFixture({uid: fallback}, cached_fallback=True)
    live = LiveBinaryBroker('C2', '0005', {uid: action}, {action['local_query']: 1},
        dict(groups=set(), probes=[[], [], []]), client, backend_kind='recorded_contract_fixture')
    assert live.request([uid], 'train', 0) == [fallback]
    live.receipt(True)
    assert not opened_answer_libraries and not guard
    report = dict(status='PASS_FULL_METADATA_SPACE_AND_RECORDED_LIVE_BROKER_CONTRACT_NOT_GPU',
        settings=settings, totals=dict(totals), source_policy='Completed fixed first C1/LEARNED',
        roundtrip_paid_requests=56, roundtrip_rows_and_sequence_exact=True,
        invalid_requests_rejected_before_client_call=True,
        cached_CLIP_fallback_current_request_identity_preserved=True,
        historical_answer_libraries_opened=0, source_paid_checkpoints_read=True,
        actual_new_teacher_GPU_calls=0, new_fits=0, manual_GT_open_attempts=0, new_HOTA_read=False,
        full_pool_quality_or_frozen_teacher_GPU_inference_tested=False,
        live_sources_or_action_spaces_modified=False,
        fixture_cost_is_not_physical_inference_or_cold_cost=True,
        source_sha256={name: file_sha(ROOT/name) for name in (
            'flex_uda/active/native_binary_live_inputs.py', 'scripts/a100_native_binary_live_contract.py')},
        remaining='Fresh feature generation, fresh teacher load/cache and final retained learner must be integrated and physically tested in one newly registered matched quality/cost cohort.',
        goal_status='ACTIVE')
    atomic_json(PUBLIC/'native_binary_live_contract_20261124.json', report)
    print(json.dumps(dict(status=report['status'], totals=dict(totals), roundtrip_paid_requests=56)), flush=True)


if __name__ == '__main__':
    main()
