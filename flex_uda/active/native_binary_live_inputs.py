"""Fresh-input action metadata and an on-demand frozen-teacher broker.

These interfaces do not load student features or teacher weights, and do
not constitute a cold experiment. They remove the old four-policy answer
library dependency for a future explicitly registered fresh cohort.
"""
from collections import Counter
import hashlib
import os
import time
import numpy as np
from flex_tta.runtime.pf_receipts import append_json
from flex_uda.a100.mechanism import PAIRS, metadata
from flex_uda.active.acquisition import route
from flex_uda.active.native_binary_inputs import BinaryBroker, TEACHER_SHA
from scripts.a100_observation_teacher_pilot import FIELDS, key as request_key


def full_action_inventory(pair, video, *, view_mode='routed'):
    """Enumerate supplied tracks/real anchors without features or replies.

    Budgets are intentionally supplied separately by the future experiment;
    enumerating a question is not authorization to ask or charge it.
    """
    assert view_mode in ('routed', 'three')
    target = PAIRS[pair][1]; md = metadata(target); bank = md['videos'][video]
    observations = np.asarray(bank['observations'], dtype=int)
    tracks = sorted(set(observations[:, 1].tolist()))
    queries = [q for q in md['episodes'] if q['video_id'] == video]
    anchors = []
    for track in tracks:
        rows = np.flatnonzero(observations[:, 1] == track)
        rows = rows[np.argsort(observations[rows, 0], kind='stable')]
        chosen = np.unique(np.linspace(0, len(rows)-1, min(3, len(rows))).round().astype(int))
        anchors.append([int(observations[rows[i], 0]) for i in chosen])
    actions = {}
    for qi, query in enumerate(queries):
        views = (route(query['query_text']),) if view_mode == 'routed' else ('CROP', 'CLIP', 'FRAME_BOX')
        for ti, (track, frames) in enumerate(zip(tracks, anchors)):
            for frame in frames:
                for view in views:
                    case = dict(target=target, video_id=video, query_id=query['expression_id'],
                        query_text=query['query_text'], track_id=int(track), frame_id=frame, view=view)
                    uid = hashlib.sha256(request_key(case).encode()).hexdigest()
                    assert uid not in actions
                    actions[uid] = dict(uid=uid, local_query=qi, local_track=ti, case=case)
    assert len(actions) == len(queries)*sum(map(len, anchors))*(3 if view_mode == 'three' else 1)
    return actions


class LiveBinaryBroker(BinaryBroker):
    """Same paid-request interface, backed only by the supplied teacher API."""
    def __init__(self, pair, video, actions, quotas, reservation, teacher_client,
                 *, ledger_path=None, backend_kind='physical_gpu'):
        super().__init__(pair, video, actions, quotas, reservation)
        assert backend_kind in ('physical_gpu', 'recorded_contract_fixture')
        self.client = teacher_client; self.ledger_path = ledger_path; self.backend_kind = backend_kind
        self.start_stats = dict(teacher_client.stats); self.service_seconds = 0.; self.attempts = 0

    def _load_private(self):
        raise RuntimeError('Live broker has no historical answer-library reader')

    def request(self, uids, role, stage):
        assert role in ('train', 'probe') and stage in (0, 1, 2)
        assert len(uids) == len(set(uids))
        counts = Counter(r['local_query'] for r in self.received.values())
        for uid in uids:
            assert uid in self.actions and uid not in self.received
            action = self.actions[uid]; q, t = action['local_query'], action['local_track']
            if role == 'train':
                assert (q, t) not in self.reservation['groups']
            else:
                assert uid in self.reservation['probes'][stage]
            counts[q] += 1
            assert counts[q] <= self.quotas[q]
        if not uids:
            return []
        requests = [{k: self.actions[u]['case'][k] for k in FIELDS} for u in uids]
        started = time.perf_counter(); before = dict(self.client.stats); success = False
        try:
            answers = self.client.ask_many(requests)
            assert len(answers) == len(requests)
            result = []
            for uid, case, answer in zip(uids, requests, answers):
                # A CLIP fallback and CROP can legitimately share a cache
                # entry. Preserve the CURRENT requested view, and validate
                # the returned requested_view plus all other identity fields.
                assert all(answer[k] == case[k] for k in FIELDS if k != 'view')
                assert answer.get('requested_view', answer['view']) == case['view']
                assert answer['frozen_weight_sha256'] == TEACHER_SHA
                p = float(answer['yes_probability'])
                assert np.isfinite(p) and 0 <= p <= 1
                result.append(dict(self.actions[uid], probability=p,
                    teacher_request_id=answer['request_id'], frozen_weight_sha256=TEACHER_SHA,
                    effective_view=answer['effective_view']))
            for uid, row in zip(uids, result):
                self.received[uid] = row
                self.sequence.append(dict(uid=uid, role=role, stage=stage))
            success = True
            return result
        finally:
            elapsed = time.perf_counter()-started; self.service_seconds += elapsed; self.attempts += 1
            delta = {k: self.client.stats.get(k, 0)-v for k, v in before.items()}
            if self.ledger_path is not None:
                append_json(self.ledger_path, dict(phase='teacher_request_service', seconds=elapsed,
                    requested_UIDs=len(uids), role=role, stage=stage, successful_return=success,
                    teacher_client_cost_delta=delta, backend_kind=self.backend_kind,
                    physical_GPU=os.environ.get('CUDA_VISIBLE_DEVICES'),
                    failed_attempt_costs_retained=True, includes_model_load_or_features=False))

    def receipt(self, complete=False):
        row = super().receipt(complete)
        delta = {k: self.client.stats.get(k, 0)-v for k, v in self.start_stats.items()}
        row.update(actual_new_teacher_GPU_calls=delta.get('new_calls', 0),
            teacher_cache_hits=delta.get('cache_hits', 0),
            teacher_actual_forward_seconds=delta.get('forward_seconds', 0.),
            teacher_service_wall_seconds=self.service_seconds, teacher_service_attempts=self.attempts,
            teacher_client_cost_delta=delta, backend_kind=self.backend_kind,
            prepaid_library_replay_not_deployment_cost=self.backend_kind != 'physical_gpu',
            historical_answer_library_read=False,
            teacher_service_component_is_not_full_cold_cost=True)
        return row
