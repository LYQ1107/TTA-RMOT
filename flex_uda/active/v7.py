"""Registered online empirical-Bayes presence and evidence-only calibration."""
from collections import defaultdict
import numpy as np
from scipy.special import expit, logit
from scipy.stats import rankdata
from flex_uda.active.v6 import BeliefStateV6, fast_step_v6
from flex_uda.active.acquisition import Question, VIEWS, information_gain, route


def weak_rank_logits(maximum):
    value = np.asarray(maximum, dtype=float)
    return np.zeros_like(value) if len(value) == 1 else 2*(rankdata(value, method='average')-1)/(len(value)-1)-1


class OnlinePresenceEB:
    def __init__(self):
        self.rho_alpha = self.rho_beta = self.pi_alpha = self.pi_beta = 1.

    @property
    def rho(self): return self.rho_alpha/(self.rho_alpha+self.rho_beta)

    @property
    def pi(self): return self.pi_alpha/(self.pi_alpha+self.pi_beta)

    def receipt(self):
        return dict(rho_hat=self.rho, pi_hat=self.pi, rho_alpha=self.rho_alpha,
                    rho_beta=self.rho_beta, pi_alpha=self.pi_alpha, pi_beta=self.pi_beta)

    def finish_video(self, state, reliability):
        before = self.receipt()
        present = (state.presence_probability > .9) & ~state.absent
        absent = state.absent
        counts = []
        for (q, track), answers in sorted(state.presence_answers.items()):
            if not present[q]: continue
            log1, log0 = np.log(state.pi_start), np.log1p(-state.pi_start)
            for view, yes in answers:
                s, c = reliability[view]
                log1 += np.log(s if yes else 1-s)
                log0 += np.log(1-c if yes else c)
            p = float(np.exp(log1-np.logaddexp(log1, log0)))
            counts.append(dict(query=int(q), track=int(track), answers=len(answers), probability=p))
        self.rho_alpha += int(present.sum()); self.rho_beta += int(absent.sum())
        self.pi_alpha += sum(x['probability'] for x in counts)
        self.pi_beta += sum(1-x['probability'] for x in counts)
        return dict(before=before, after=self.receipt(), confirmed_present=int(present.sum()),
                    confirmed_absent=int(absent.sum()), unknown=int((~present & ~absent).sum()),
                    pi_pair_soft_counts=counts, pi_video_start=state.pi_start,
                    identity_answers_excluded=True, each_stratified_pair_counted_once=True)


class BeliefStateV7(BeliefStateV6):
    def __init__(self, track_scores, fractions, presence_max, *, track_lengths,
                 rho_hat=.5, pi_hat=.5, slope=1., presence_logLR_cap=2.):
        self.rho_start, self.pi_start = float(rho_hat), float(pi_hat)
        self.presence_logLR_cap = presence_logLR_cap
        self.presence_answers = defaultdict(list)
        self.presence_by_query = defaultdict(list)
        self.answer_history = defaultdict(list)
        self.presence_reliability = {v: (.8, .8) for v in VIEWS}
        self.presence_stopped = np.zeros(len(track_scores), dtype=bool)
        self.stop_probability = np.full(len(track_scores), np.nan)
        super().__init__(track_scores, fractions, presence_max, track_lengths=track_lengths, slope=slope)

    @property
    def prior(self):
        return np.clip(super().prior, -2., 2.)

    def refresh_presence(self, presence_max):
        self.presence_max = np.asarray(presence_max, dtype=float)
        self.weak_presence_logits = weak_rank_logits(self.presence_max)
        self.recompute_presence()

    def recompute_presence(self, reliability=None, queries=None):
        if reliability is not None: self.presence_reliability = dict(reliability)
        if not hasattr(self, 'presence_probability'):
            self.presence_probability = np.full(len(self.a), .5)
            self.presence_probability_recomputed = np.full(len(self.a), .5)
            self.presence_logLR_unclipped_sum = np.zeros(len(self.a))
            self.presence_logLR_clipped_sum = np.zeros(len(self.a))
            self.presence_logLR_clip_count = np.zeros(len(self.a), dtype=int)
        if queries is None: queries = range(len(self.a))
        for q in queries:
            total = logit(self.rho_start)+self.weak_presence_logits[q]
            raw_sum = clipped_sum = 0.; capped = 0
            for view, yes in self.presence_by_query[q]:
                s, c = self.presence_reliability[view]
                under_present = self.pi_start*s+(1-self.pi_start)*(1-c) if yes else self.pi_start*(1-s)+(1-self.pi_start)*c
                under_empty = 1-c if yes else c
                delta = np.log(under_present)-np.log(under_empty)
                raw_sum += delta
                if self.presence_logLR_cap is not None:
                    clipped = np.clip(delta, -self.presence_logLR_cap, self.presence_logLR_cap)
                    capped += int(clipped != delta); delta = clipped
                clipped_sum += delta
                total += delta
            value = float(expit(total))
            self.presence_probability_recomputed[q] = value
            self.presence_logLR_unclipped_sum[q] = raw_sum
            self.presence_logLR_clipped_sum[q] = clipped_sum
            self.presence_logLR_clip_count[q] = capped
            if not self.presence_stopped[q] and value < .1:
                self.presence_stopped[q] = True; self.stop_probability[q] = value
            self.presence_probability[q] = self.stop_probability[q] if self.presence_stopped[q] else value
        self.absent = self.presence_stopped.copy()
        self.confirmed_present = (self.presence_probability > .9) & ~self.absent
        self.low = self.presence_probability < .1

    def observe(self, query, track, answer, view, reliability, *, presence_question=False):
        from flex_uda.active.belief import evidence_increment
        self.answer_history[query, track].append((view, bool(answer)))
        if presence_question:
            self.presence_answers[query, track].append((view, bool(answer)))
            self.presence_by_query[query].append((view, bool(answer)))
        raw = evidence_increment(answer, *reliability)
        delta = float(np.clip(raw, -2., 2.))
        self.evidence[query, track] += delta; self.asked_mask[query, track] = True
        self.events.append(dict(query=int(query), track=int(track), answer=bool(answer), view=view,
            delta=delta, unclipped_delta=raw, sensitivity=float(reliability[0]),
            specificity=float(reliability[1]), presence_question=presence_question))
        (self.yes_tracks if answer else self.no_tracks)[query].add(int(track))
        self.recompute_presence(queries=[query])

    def pseudo_labels(self):
        p = self.posterior
        value = np.where(p > .85, 1, np.where(p < .15, 0, -1)).astype('int8')
        value[~self.asked_mask] = -1; value[:, self.frozen_tracks] = -1
        return value

    @property
    def presence_supported(self):
        return np.asarray([bool(self.presence_by_query[q]) for q in range(len(self.a))])


class _TeacherSupportedFastView:
    def __init__(self, state): self.state = state
    def __getattr__(self, name): return getattr(self.state, name)
    @property
    def confirmed_present(self): return self.state.confirmed_present & self.state.presence_supported
    @property
    def absent(self): return self.state.absent & self.state.presence_supported


def fast_step_v7(state, presence_logits, **kwargs):
    # The v6 objective, [0,1] box, a L2=1, and T/beta fit are unchanged.
    result = fast_step_v6(_TeacherSupportedFastView(state), presence_logits, **kwargs)
    result['version'] = 'v7'; result['presence_support_only_stratified_answers'] = True
    return result


def questions_v7(state, texts, lengths, reliability, round_index):
    eligible = [t for t, n in enumerate(lengths) if n >= 3]
    probability = state.posterior
    rechecks, ordinary = [], []
    for q, text in enumerate(texts):
        if state.absent[q]: continue
        routed = route(text)
        for t in eligible:
            history = state.answer_history[q, t]
            gains = {v: information_gain(probability[q, t], *reliability[v]) for v in VIEWS}
            if len(history) == 1 and history[0][1] and .5 <= probability[q, t] < .85:
                rechecks.append(Question(q, t, history[0][0], 'POSITIVE_RECHECK', gains[history[0][0]]))
            else:
                view = routed if round_index < 2 else max(VIEWS, key=lambda v: gains[v])
                ordinary.append(Question(q, t, view, 'IDENTITY', max(gains.values())))
        if .1 <= state.presence_probability[q] <= .9:
            used = {t for qi, t in state.presence_answers if qi == q}
            covered = {int(state.strata[q, t]) for t in used}
            missing = [s for s in range(3) if s not in covered]
            for stratum in missing or range(3):
                available = [t for t in eligible if state.strata[q, t] == stratum and t not in used]
                if not available: continue
                t = min(available, key=lambda t: (-max(information_gain(probability[q, t], *reliability[v]) for v in VIEWS), -state.z[q, t], t))
                gains = {v: information_gain(state.presence_probability[q], *reliability[v]) for v in VIEWS}
                view = routed if round_index < 2 else max(VIEWS, key=lambda v: gains[v])
                ordinary.append(Question(q, t, view, 'PRESENCE', max(gains.values())))
    order = lambda x: (-x.information_gain, x.kind != 'PRESENCE', -state.z[x.query, x.track], x.query, x.track, x.view)
    return sorted(rechecks, key=order)+sorted(ordinary, key=order)


def balanced_weights(positive, negative, *, exclusive):
    if exclusive:
        return dict(binary_positive_weight=1., none_class_weight=max(.2, min(1., positive/negative)) if negative else 1.)
    return dict(binary_positive_weight=min(5., negative/positive) if positive else 1., none_class_weight=1.)
