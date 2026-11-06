"""Preregistered v8 hierarchical presence, coverage, and acquisition.

The registered MAR rationale assumes the supplied three-stratum response
model and conditional independence are correctly specified. Adaptive questions
depend only on observed scores/answers; no outcome-filtered sentence subset is
used. The capped E-step below is the specified robust approximation, not a
general proof of consistency; the registered recovery test is kept explicit.
"""
from collections import defaultdict
import numpy as np
from scipy.special import expit, logit
from flex_uda.active.v7 import BeliefStateV7, questions_v7, fast_step_v7
from flex_uda.active.acquisition import Question, VIEWS, information_gain, route


class PresenceEM:
    def __init__(self, n_strata=3, a=2., b=2., clip=(.05, .95),
                 iters=200, tol=1e-7, track_cap=2.):
        self.n_strata = n_strata
        self.a, self.b, self.clip = a, b, clip
        self.iters, self.tol, self.track_cap = iters, tol, track_cap
        self.records = []
        self.rho = .5
        self.pi = np.full(n_strata, .5)
        self.iterations = 0
        self.converged = True
        self.last_delta = 0.

    def add_video(self, video, state):
        added = 0
        for q in range(len(state.a)):
            tracks = []
            for (qi, t), answers in sorted(state.answer_history.items()):
                k = int(state.strata[qi, t])
                if qi == q and k >= 0 and answers:
                    tracks.append((k, [(v, bool(y)) for v, y in answers]))
            if tracks:
                self.records.append(dict(video=str(video), query=int(q), tracks=tracks))
                added += 1
        return added

    def track_terms(self, k, answers, rel):
        l_pos = sum(np.log(rel[v][0] if y else 1-rel[v][0]) for v, y in answers)
        l_neg = sum(np.log(1-rel[v][1] if y else rel[v][1]) for v, y in answers)
        m1 = np.logaddexp(np.log(self.pi[k])+l_pos,
                         np.log1p(-self.pi[k])+l_neg)
        return (float(np.clip(m1-l_neg, -self.track_cap, self.track_cap)),
                float(np.exp(np.log(self.pi[k])+l_pos-m1)))

    def fit(self, rel):
        if not self.records:
            self.iterations, self.converged, self.last_delta = 0, True, 0.
            return self.receipt()
        qs, strata, lp, ln = [], [], [], []
        for q, rec in enumerate(self.records):
            for k, answers in rec['tracks']:
                qs.append(q); strata.append(k)
                lp.append(sum(np.log(rel[v][0] if y else 1-rel[v][0]) for v, y in answers))
                ln.append(sum(np.log(1-rel[v][1] if y else rel[v][1]) for v, y in answers))
        qs = np.asarray(qs, dtype=int); strata = np.asarray(strata, dtype=int)
        lp, ln = np.asarray(lp), np.asarray(ln)
        n = len(self.records)
        self.converged = False
        for iteration in range(self.iters):
            m1 = np.logaddexp(np.log(self.pi[strata])+lp,
                             np.log1p(-self.pi[strata])+ln)
            llr = np.clip(m1-ln, -self.track_cap, self.track_cap)
            r = np.exp(np.log(self.pi[strata])+lp-m1)
            g = expit(logit(self.rho)+np.bincount(qs, weights=llr, minlength=n))
            num = np.bincount(strata, weights=g[qs]*r, minlength=self.n_strata)
            den = np.bincount(strata, weights=g[qs], minlength=self.n_strata)
            rho = float(np.clip((g.sum()+self.a-1)/(n+self.a+self.b-2), *self.clip))
            pi = np.clip((num+self.a-1)/(den+self.a+self.b-2), *self.clip)
            self.last_delta = float(max(abs(rho-self.rho), np.abs(pi-self.pi).max()))
            self.rho, self.pi, self.iterations = rho, pi, iteration+1
            if self.last_delta < self.tol:
                self.converged = True
                break
        return self.receipt()

    def receipt(self, iterations=None):
        counts = np.zeros(self.n_strata, dtype=int)
        for rec in self.records:
            for k, _ in rec['tracks']: counts[k] += 1
        return dict(version='v8', rho_hat=float(self.rho), pi_strata=self.pi.tolist(),
                    iterations=self.iterations if iterations is None else iterations,
                    converged=bool(self.converged), last_delta=self.last_delta,
                    records=len(self.records), tracks_per_stratum=counts.tolist(),
                    Beta_prior=[self.a, self.b], clip=list(self.clip),
                    tolerance=self.tol, maximum_iterations=self.iters,
                    track_logLR_cap=self.track_cap,
                    outcome_filtered_sentence_selection=False)


class BeliefStateV8(BeliefStateV7):
    snapshot_fields = ('presence_probability', 'absent', 'confirmed_present',
                       'presence_stopped', 'stop_probability', 'low_tentative')

    def __init__(self, track_scores, fractions, presence_max, *, track_lengths,
                 rho_hat=.5, pi_strata=None, slope=1.):
        self.pi_strata_start = np.full(3, .5) if pi_strata is None else np.asarray(pi_strata, dtype=float).copy()
        self.low_tentative = np.zeros(len(track_scores), dtype=bool)
        self.covered_strata = np.zeros((len(track_scores), 3), dtype=bool)
        self.needed_strata = np.zeros_like(self.covered_strata)
        super().__init__(track_scores, fractions, presence_max, track_lengths=track_lengths,
                         rho_hat=rho_hat, pi_hat=.5, slope=slope)

    def recompute_presence(self, reliability=None, queries=None):
        if reliability is not None: self.presence_reliability = dict(reliability)
        if not hasattr(self, 'presence_probability'):
            self.presence_probability = np.full(len(self.a), .5)
            self.presence_probability_recomputed = np.full(len(self.a), .5)
            self.presence_logLR_unclipped_sum = np.zeros(len(self.a))
            self.presence_logLR_clipped_sum = np.zeros(len(self.a))
            self.presence_logLR_clip_count = np.zeros(len(self.a), dtype=int)
        estimator = PresenceEM()
        estimator.pi = self.pi_strata_start.copy()
        for q in range(len(self.a)) if queries is None else queries:
            needed = {int(k) for k in self.strata[q] if k >= 0}
            covered = set(); raw_sum = clipped_sum = 0.; capped = 0
            for (qi, t), answers in sorted(self.answer_history.items()):
                k = int(self.strata[qi, t])
                if qi != q or k < 0 or not answers: continue
                covered.add(k)
                lp = sum(np.log(self.presence_reliability[v][0] if y else 1-self.presence_reliability[v][0]) for v, y in answers)
                ln = sum(np.log(1-self.presence_reliability[v][1] if y else self.presence_reliability[v][1]) for v, y in answers)
                m1 = np.logaddexp(np.log(self.pi_strata_start[k])+lp,
                                 np.log1p(-self.pi_strata_start[k])+ln)
                raw = float(m1-ln)
                clipped, _ = estimator.track_terms(k, answers, self.presence_reliability)
                raw_sum += raw; clipped_sum += clipped; capped += int(abs(raw) > 2.)
            value = float(expit(logit(np.clip(self.rho_start, .05, .95)) +
                                self.weak_presence_logits[q]+clipped_sum))
            self.needed_strata[q] = [k in needed for k in range(3)]
            self.covered_strata[q] = [k in covered for k in range(3)]
            self.presence_probability_recomputed[q] = value
            self.presence_logLR_unclipped_sum[q] = raw_sum
            self.presence_logLR_clipped_sum[q] = clipped_sum
            self.presence_logLR_clip_count[q] = capped
            # Require actual evidence even if the eligible set is empty.
            if not self.presence_stopped[q] and value < .1 and covered and covered >= needed:
                self.presence_stopped[q] = True
                self.stop_probability[q] = value
            self.presence_probability[q] = self.stop_probability[q] if self.presence_stopped[q] else value
            self.low_tentative[q] = bool(not self.presence_stopped[q] and value < .1 and not covered >= needed)
        self.absent = self.presence_stopped.copy()
        self.confirmed_present = (self.presence_probability > .9) & ~self.absent
        self.low = self.presence_probability < .1
        if getattr(self, 'replay_presence_snapshot', None) is not None:
            self.apply_presence_snapshot(self.replay_presence_snapshot)

    @property
    def presence_supported(self):
        return self.covered_strata.any(axis=1)

    def presence_snapshot(self):
        return {key: np.asarray(getattr(self, key)).copy() for key in self.snapshot_fields}

    def apply_presence_snapshot(self, snap):
        for key in self.snapshot_fields:
            setattr(self, key, np.asarray(snap[key]).copy())
        self.low = self.presence_probability < .1
        assert all(getattr(self, k).tobytes() == np.asarray(snap[k]).tobytes()
                   for k in self.snapshot_fields)

    def freeze_presence_snapshot(self, snap):
        self.replay_presence_snapshot = {key: np.asarray(snap[key]).copy()
                                         for key in self.snapshot_fields}
        self.apply_presence_snapshot(self.replay_presence_snapshot)

    def output_absence(self):
        return self.absent | (self.low_tentative & (self.presence_probability < .1))


def questions_v8(state, texts, lengths, reliability, round_index):
    eligible = [t for t, n in enumerate(lengths) if n >= 3]
    glance, cover = [], []
    original = questions_v7(state, texts, lengths, reliability, round_index)
    for q, text in enumerate(texts):
        if state.absent[q]: continue
        def make(t, kind):
            gains = {v: information_gain(state.posterior[q, t], *reliability[v]) for v in VIEWS}
            view = route(text) if round_index < 2 else max(VIEWS, key=lambda v: gains[v])
            return Question(q, t, view, kind, gains[view])
        if not state.asked_mask[q].any():
            available = [t for t in eligible if state.strata[q, t] == 0]
            if available:
                t = min(available, key=lambda t: (-state.base_logits[q, t], t))
                glance.append(make(t, 'GLANCE'))
        if state.low_tentative[q]:
            for k in range(3):
                if state.covered_strata[q, k]: continue
                available = [t for t in eligible if state.strata[q, t] == k and not state.asked_mask[q, t]]
                if available:
                    t = min(available, key=lambda t: (-state.base_logits[q, t], t))
                    cover.append(make(t, 'PRESENCE_COVER'))
    order = lambda x: (-x.information_gain, x.kind != 'PRESENCE', -state.z[x.query, x.track], x.query, x.track, x.view)
    rechecks = [x for x in original if x.kind == 'POSITIVE_RECHECK']
    ordinary = [x for x in original if x.kind != 'POSITIVE_RECHECK' and
                not (state.low_tentative[x.query] and x.kind == 'IDENTITY')]
    # GLANCE has the registered query-index order; the remaining phases use
    # the inherited v7 ordering, with each pair asked at most once per round.
    return sorted(glance, key=lambda x: x.query)+sorted(rechecks, key=order)+sorted(cover, key=order)+sorted(ordinary, key=order)


def fast_step_v8(state, presence_logits, **kwargs):
    result = fast_step_v7(state, presence_logits, **kwargs)
    result.update(version='v8', presence_support_all_answer_kinds=True)
    return result
