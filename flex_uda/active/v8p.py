"""Post-D3 query-protocol switch; the CHECKLIST numerical path is v8 itself."""
import hashlib
import json
import numpy as np
from flex_uda.active.v8 import BeliefStateV8, PresenceEM, questions_v8
from flex_uda.active.v7 import questions_v7

CHECKLIST = 'CHECKLIST'
PRESUPPOSED = 'PRESUPPOSED'


def classify_protocol(video_queries):
    """Exact sets of raw runtime sentence texts; no labels, scores or GT."""
    if not video_queries:
        raise ValueError('Protocol requires at least one video')
    sets = {str(video): tuple(sorted(set(texts)))
            for video, texts in sorted(video_queries.items())}
    unique = set(sets.values())
    protocol = CHECKLIST if len(unique) == 1 else PRESUPPOSED
    return dict(protocol=protocol, presence_inference=protocol == CHECKLIST,
                videos=len(sets), unique_query_sets=len(unique),
                per_video={v: dict(query_count=len(texts),
                    query_set_sha256=hashlib.sha256(json.dumps(texts, ensure_ascii=False,
                        separators=(',', ':')).encode()).hexdigest())
                           for v, texts in sets.items()})


class PresupposedBeliefState(BeliefStateV8):
    """Presence is a protocol assumption, never inferred from sparse No votes."""
    def recompute_presence(self, reliability=None, queries=None):
        if reliability is not None:
            self.presence_reliability = dict(reliability)
        n = len(self.a)
        self.presence_probability = np.ones(n)
        self.presence_probability_recomputed = np.ones(n)
        self.absent = np.zeros(n, dtype=bool)
        self.confirmed_present = np.ones(n, dtype=bool)
        self.presence_stopped = np.zeros(n, dtype=bool)
        self.stop_probability = np.full(n, np.nan)
        self.low = np.zeros(n, dtype=bool)
        self.low_tentative = np.zeros(n, dtype=bool)
        self.presence_logLR_unclipped_sum = np.zeros(n)
        self.presence_logLR_clipped_sum = np.zeros(n)
        self.presence_logLR_clip_count = np.zeros(n, dtype=int)
        self.needed_strata = np.asarray([[np.any(self.strata[q] == k) for k in range(3)]
                                        for q in range(n)], dtype=bool)
        self.covered_strata = np.zeros((n, 3), dtype=bool)
        # Inherited fast-step still needs actual teacher support. Identity
        # answers are support in v8 too; an unasked sentence cannot fit T/beta.
        for (q, track), answers in self.answer_history.items():
            k = int(self.strata[q, track])
            if answers and k >= 0:
                self.covered_strata[q, k] = True
        if getattr(self, 'replay_presence_snapshot', None) is not None:
            self.apply_presence_snapshot(self.replay_presence_snapshot)
        assert self.presence_probability.min() == 1
        assert not self.absent.any() and not self.low_tentative.any()


class DisabledPresenceEstimator:
    """Checkpoint-compatible no-op, not a PresenceEM instance or invocation."""
    rho = .5
    pi = np.full(3, .5)
    converged = True

    def receipt(self):
        return dict(version='v8P', protocol=PRESUPPOSED,
                    presence_inference=False, status='DISABLED_BY_QUERY_PROTOCOL',
                    rho_hat=1., pi_strata=[.5, .5, .5], iterations=0,
                    converged=True, records=0, tracks_per_stratum=[0, 0, 0],
                    identification_status=['NOT_APPLICABLE']*3,
                    presence_likelihood_evaluations_per_stratum=[0, 0, 0])

    def add_video(self, video, state):
        assert not state.absent.any() and np.all(state.presence_probability == 1)
        return 0

    def fit(self, reliability):
        return self.receipt()


def belief_state(protocol, *args, **kwargs):
    if protocol == CHECKLIST:
        return BeliefStateV8(*args, **kwargs)
    if protocol != PRESUPPOSED:
        raise ValueError(protocol)
    return PresupposedBeliefState(*args, **kwargs)


def presence_estimator(protocol):
    return PresenceEM() if protocol == CHECKLIST else DisabledPresenceEstimator()


def questions_v8p(protocol, state, texts, lengths, reliability, round_index):
    if protocol == CHECKLIST:
        return questions_v8(state, texts, lengths, reliability, round_index)
    assert protocol == PRESUPPOSED
    assert np.all(state.presence_probability == 1) and not state.absent.any()
    # v7 supplies exactly the inherited RECHECK then IDENTITY ordering when
    # presence=1. Unlike filtering v8's output, this never constructs GLANCE.
    result = questions_v7(state, texts, lengths, reliability, round_index)
    assert all(q.kind in ('IDENTITY', 'POSITIVE_RECHECK') for q in result)
    return result
