"""Registered v6 repairs; the v5 belief/calibration path remains unchanged."""
import numpy as np
from scipy.special import expit
from flex_uda.active.belief import normalize_track_scores, evidence_increment
from flex_uda.active.reliability import Reliability


def selection_fractions(observations, track_ids, frame_index, margins, utilities,
                        presence_logits, *, temperature=1., bias=0., raw_k=None):
    """Stage-A expected-IoU top-k, with calibrated query-frame presence.

    Keep the conditional nonempty-set utility and replace only its presence
    mass. At the source calibration (T=1,beta=0), use the sealed raw_k to
    avoid changing float64 argmax ties through stored float32 utilities.
    """
    from scripts.diag_pf_countmatched import topk_ids
    from collections import defaultdict
    observations = np.asarray(observations)
    by_frame = defaultdict(list)
    by_track = defaultdict(list)
    for j, (frame, track) in enumerate(observations):
        by_frame[int(frame)].append(j); by_track[int(track)].append(j)
    mask = np.zeros_like(margins, dtype=bool)
    calibrated = expit(np.asarray(presence_logits)/temperature+bias)
    if temperature == 1. and bias == 0. and raw_k is not None:
        counts = np.asarray(raw_k)
    else:
        old_p = expit(np.asarray(presence_logits))
        values = np.asarray(utilities, dtype=float).copy()
        values[..., 1:] *= (calibrated/np.maximum(old_p, np.finfo(float).tiny))[..., None]
        values[..., 0] = 1-calibrated
        counts = values.argmax(axis=-1)
    for frame, ids in by_frame.items():
        column = frame_index[frame]
        tracks = observations[ids, 1].tolist()
        for q in range(len(margins)):
            scores = dict(zip(tracks, margins[q, ids]))
            chosen = topk_ids(tracks, scores, int(counts[q, column]))
            mask[q, ids] = [t in chosen for t in tracks]
    fractions = np.stack([mask[:, by_track[t]].mean(axis=1) for t in track_ids], axis=1)
    return fractions, calibrated, mask


class BeliefStateV6:
    def __init__(self, track_scores, fractions, presence_max, *, track_lengths, slope=1.):
        self.z = normalize_track_scores(track_scores)  # Acquisition tie-breaking only.
        self.frozen_tracks = np.asarray(track_lengths) < 3
        self.a = np.zeros(len(track_scores), dtype=float)
        self.slope = float(slope)
        self.evidence = np.zeros_like(np.asarray(fractions, dtype=float))
        self.asked_mask = np.zeros_like(self.evidence, dtype=bool)
        self.yes_tracks = [set() for _ in self.a]
        self.no_tracks = [set() for _ in self.a]
        self.events = []
        self.confirmed_present = np.zeros(len(self.a), dtype=bool)
        self.absent = np.zeros(len(self.a), dtype=bool)
        self.presence_max = np.asarray(presence_max, dtype=float)
        self.low = self.presence_max < .2
        self.replace_fractions(fractions)
        self.initial_prior = self.prior.copy()
        # Freeze each video's initial prior-ranking thirds, including a
        # deterministic track-index tie. The z values never define a prior.
        self.strata = np.full(self.evidence.shape, -1, dtype=int)
        eligible = np.flatnonzero(~self.frozen_tracks)
        for q in range(len(self.a)):
            ranked = sorted(eligible, key=lambda t: (-self.base_logits[q, t], int(t)))
            for k, part in enumerate(np.array_split(ranked, 3)):
                self.strata[q, part] = k
        self.refresh_presence(self.presence_max)

    def replace_fractions(self, fractions):
        self.fractions = np.asarray(fractions, dtype=float).copy()
        p = np.clip(self.fractions, .05, .95)
        self.base_logits = np.clip(np.log(p/(1-p)), -2., 2.)
        self.expected = self.fractions.sum(axis=1)  # Diagnostic only, never an intercept.

    @property
    def prior(self):
        result = self.a[:, None] + self.slope*self.base_logits
        if hasattr(self, 'initial_prior'):
            result[:, self.frozen_tracks] = self.initial_prior[:, self.frozen_tracks]
        return result

    @property
    def logits(self):
        result = self.prior + self.evidence
        result[self.absent] = -np.inf
        return result

    @property
    def posterior(self):
        return expit(self.logits)

    def observe(self, query, track, answer, view, reliability, *, presence_question=False):
        raw = evidence_increment(answer, *reliability)
        delta = float(np.clip(raw, -2., 2.))
        self.evidence[query, track] += delta
        self.asked_mask[query, track] = True
        self.events.append(dict(query=int(query), track=int(track), answer=bool(answer), view=view,
            delta=delta, unclipped_delta=raw, sensitivity=float(reliability[0]),
            specificity=float(reliability[1]), presence_question=presence_question))
        (self.yes_tracks if answer else self.no_tracks)[query].add(int(track))
        self.confirmed_present[query] = bool(self.yes_tracks[query])
        self.refresh_presence(self.presence_max)

    def refresh_presence(self, presence_max):
        self.presence_max = np.asarray(presence_max, dtype=float)
        self.low = self.presence_max < .2
        for q in range(len(self.a)):
            covered = {int(self.strata[q, t]) for t in self.no_tracks[q]
                       if self.strata[q, t] >= 0}
            self.absent[q] = (self.low[q] and not self.yes_tracks[q]
                              and len(self.no_tracks[q]) >= 3 and covered == {0, 1, 2})

    def replace_prior(self, *, intercept=None, slope=None, scores=None):
        if intercept is not None:
            self.a = np.asarray(intercept, dtype=float).copy()
        if slope is not None:
            if not 0 <= slope <= 1:
                raise ValueError('Registered v6 w must be in [0,1]')
            self.slope = float(slope)
        if scores is not None:
            self.z = normalize_track_scores(scores)
        return self.posterior

    def pseudo_labels(self):
        p = self.posterior
        labels = np.where(p > .95, 1, np.where(p < .05, 0, -1)).astype('int8')
        labels[~self.asked_mask] = -1
        labels[:, self.frozen_tracks] = -1
        return labels


class ReliabilityV6(Reliability):
    """Same EM/update count, Beta(2,2); no correlated identity annotator."""
    workers = ('CROP', 'CLIP', 'FRAME_BOX')

    def __init__(self):
        super().__init__(alpha=2., beta=2., iterations=20)
        # The user changed the Beta prior, not v5's initial EM values (.8,.8).
        self.values = {w: self.values[w] for w in self.workers}

    def add(self, key, view, answer, identity_answer=None):
        self.answers[key][view].append(int(answer))

    def fit(self):
        if not self.answers:
            return self.values
        keys = sorted(self.answers, key=str)
        yes = np.zeros((len(keys), len(self.workers))); no = np.zeros_like(yes)
        for i, key in enumerate(keys):
            for j, worker in enumerate(self.workers):
                observed = self.answers[key].get(worker, [])
                yes[i, j] = sum(observed); no[i, j] = len(observed)-sum(observed)
        s = np.array([self.values[w][0] for w in self.workers])
        c = np.array([self.values[w][1] for w in self.workers])
        prevalence = self.prevalence
        for _ in range(self.iterations):
            log1 = np.log(prevalence)+yes@np.log(s)+no@np.log1p(-s)
            log0 = np.log1p(-prevalence)+yes@np.log1p(-c)+no@np.log(c)
            positive = np.exp(log1-np.logaddexp(log1, log0))
            trials = yes+no
            s = (positive@yes+self.alpha)/(positive@trials+self.alpha+self.beta)
            c = ((1-positive)@no+self.alpha)/((1-positive)@trials+self.alpha+self.beta)
            prevalence = float((positive.sum()+1)/(len(positive)+2))
        self.values = {w: (float(s[j]), float(c[j])) for j, w in enumerate(self.workers)}
        self.prevalence = prevalence
        self.posteriors = dict(zip(keys, map(float, positive)))
        return self.values

    def receipt(self):
        return dict(super().receipt(), workers=list(self.workers), identity_annotator=False,
                    initial_sensitivity_specificity=[.8, .8], per_answer_log_odds_cap=2.)


def questions_v6(state, texts, lengths, reliability, round_index):
    from flex_uda.active.acquisition import Question, VIEWS, route, information_gain
    result = []
    eligible = [t for t, n in enumerate(lengths) if n >= 3]
    for q, text in enumerate(texts):
        if state.absent[q]:
            continue
        routed = route(text)
        for t in eligible:
            ig = {v: information_gain(state.posterior[q, t], *reliability[v]) for v in VIEWS}
            view = routed if round_index < 2 else max(VIEWS, key=lambda v: ig[v])
            result.append(Question(q, t, view, 'IDENTITY', max(ig.values())))
        if (state.low[q] or .1 < state.presence_max[q] < .9) and not state.confirmed_present[q]:
            covered = {state.strata[q, t] for t in state.no_tracks[q]}
            for stratum in range(3):
                if stratum in covered:
                    continue
                candidates = [t for t in eligible if state.strata[q, t] == stratum
                              and t not in state.no_tracks[q]]
                if not candidates:
                    continue
                # The section4.3 information-gain value is unchanged; z only
                # orders equal-value candidates. Never replace it by a prior.
                t = min(candidates, key=lambda t: (-max(information_gain(state.posterior[q, t],
                        *reliability[v]) for v in VIEWS), -state.z[q, t], t))
                ig = {v: information_gain(state.presence_max[q], *reliability[v]) for v in VIEWS}
                view = routed if round_index < 2 else max(VIEWS, key=lambda v: ig[v])
                result.append(Question(q, t, view, 'PRESENCE', max(ig.values())))
    return sorted(result, key=lambda x: (-x.information_gain, x.kind != 'PRESENCE',
                                        -state.z[x.query, x.track], x.query, x.track, x.view))


def fast_step_v6(state, presence_logits, *, temperature=1., bias=0., **unused):
    """Box-constrained w and zero-anchored a; unchanged presence objective."""
    from scipy.optimize import minimize
    import torch
    import torch.nn.functional as F
    from flex_uda.active.map_calibration import minimize_map
    pseudo = state.pseudo_labels()
    agrees = ((pseudo == 1) & (state.evidence > 0)) | ((pseudo == 0) & (state.evidence < 0))
    valid = (pseudo >= 0) & state.asked_mask & agrees
    positive = int(((pseudo == 1) & valid).sum())
    negative = int(((pseudo == 0) & valid).sum())
    present = state.confirmed_present & ~state.absent
    absent = state.absent & ~state.confirmed_present
    confirmed = present | absent
    identity = dict(status='SKIP_INSUFFICIENT_TEACHER_SUPPORTED_CLASSES', positive=positive,
        negative=negative, unqueried_confident_excluded=0,
        queried_confident_without_agreeing_evidence=int(((pseudo >= 0) & ~agrees).sum()))
    existence = dict(status='SKIP_INSUFFICIENT_CONFIRMED_CLASSES', positive=int(present.sum()),
        negative=int(absent.sum()), conflicting_queries=int((state.confirmed_present & state.absent).sum()))
    if positive and negative:
        indices = np.flatnonzero(valid.any(1))
        mask = valid[indices]; x = state.base_logits[indices]; y = pseudo[indices].clip(0)
        def objective(parameters):
            a, w = parameters[:-1], parameters[-1]
            logits = a[:, None] + w*x
            data = np.logaddexp(0., logits[mask]).sum() - (y[mask]*logits[mask]).sum()
            prior = .5*np.square(a).sum()  # lambda_a=1, no new w penalty.
            residual = np.where(mask, expit(logits)-y, 0.)
            gradient = np.r_[residual.sum(1)+a, (residual*x).sum()]
            return float(data+prior), gradient
        initial = np.r_[state.a[indices], state.slope]
        before = objective(initial)[0]
        result = minimize(objective, initial, method='L-BFGS-B', jac=True,
                          bounds=[(None, None)]*len(indices)+[(0., 1.)], options={'maxiter': 50})
        if not np.isfinite(result.x).all() or not np.isfinite(result.fun):
            raise FloatingPointError('Nonfinite v6 constrained calibration')
        intercept = state.a.copy(); intercept[indices] = result.x[:-1]
        state.replace_prior(intercept=intercept, slope=float(result.x[-1]))
        identity.update(status='MAP_UPDATED', queries=indices.tolist(), objective_before=before,
            objective_after=float(result.fun), optimizer_iterations=int(result.nit),
            optimizer='FP64 L-BFGS-B for the registered exact [0,1] box',
            optimizer_success=bool(result.success), optimizer_message=str(result.message),
            a_L2_coefficient=1., w_bounds=[0., 1.])
    if present.any() and absent.any():
        log_t = torch.tensor(np.log(temperature), dtype=torch.float64, requires_grad=True)
        beta = torch.tensor(bias, dtype=torch.float64, requires_grad=True)
        value = torch.tensor(np.asarray(presence_logits).max(axis=1)[confirmed], dtype=torch.float64)
        target = torch.tensor(present[confirmed], dtype=torch.float64)
        def objective_presence():
            data = F.binary_cross_entropy_with_logits(value/log_t.exp()+beta, target, reduction='sum')
            return data, .5*(log_t.square()+beta.square())
        existence.update(minimize_map([log_t, beta], objective_presence), status='MAP_UPDATED')
        temperature = float(log_t.detach().exp()); bias = float(beta.detach())
    state.refresh_presence(expit(np.asarray(presence_logits)/temperature+bias).max(axis=1))
    updated = identity['status'] == 'MAP_UPDATED' or existence['status'] == 'MAP_UPDATED'
    return dict(status='V6_MAP_UPDATED' if updated else 'SKIP_INSUFFICIENT_TEACHER_SUPPORTED_CLASSES',
        identity=identity, presence=existence, temperature=temperature, bias=bias, slope=state.slope,
        w=state.slope, teacher_evidence_not_reapplied=True, support_uses_target_GT=False)


def exclusive_training_logits(margin, pseudo, track_indices):
    import torch
    known = torch.tensor(np.stack([pseudo[:, ti] >= 0 for ti in track_indices]),
                         dtype=torch.bool, device=margin.device)
    return margin.masked_fill(~known, -torch.inf)
