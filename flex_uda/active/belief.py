"""Track beliefs keep the replaceable prior separate from frozen evidence."""
import numpy as np
from scipy.special import expit


def solve_intercept(z, count, slope=1.):
    z = np.asarray(z, dtype=float)
    if not len(z): return 0.
    count = np.clip(count, 1e-9, len(z)-1e-9)
    lo, hi = -80., 80.
    for _ in range(80):
        mid = (lo+hi)/2
        if expit(mid+slope*z).sum() < count: lo = mid
        else: hi = mid
    return (lo+hi)/2


def normalize_track_scores(scores):
    scores = np.asarray(scores, dtype=float)
    std = scores.std(axis=1, keepdims=True)
    return np.divide(scores-scores.mean(axis=1, keepdims=True), std,
                     out=np.zeros_like(scores), where=std > 0)


def evidence_increment(answer, sensitivity, specificity):
    s, c = np.clip([sensitivity, specificity], 1e-8, 1-1e-8)
    return float(np.log(s/(1-c)) if answer else np.log((1-s)/c))


class BeliefState:
    def __init__(self, track_scores, expected_counts, presence_max, *, slope=1., use_presence=True, track_lengths=None):
        self.z = normalize_track_scores(track_scores)
        self.expected = np.asarray(expected_counts, dtype=float)
        self.presence_max = np.asarray(presence_max, dtype=float)
        self.low = self.expected < .5 if use_presence else np.zeros(len(self.z), dtype=bool)
        self.slope = float(slope)
        self.a = (np.asarray([solve_intercept(z, max(.5, k), slope) for z, k in zip(self.z, self.expected)])
                  if use_presence else np.zeros(len(self.z)))
        self.evidence = np.zeros_like(self.z)
        self.frozen_tracks = (np.asarray(track_lengths) < 3 if track_lengths is not None else np.zeros(self.z.shape[1], dtype=bool))
        self.initial_prior = self.a[:, None] + self.slope*self.z
        self.absent = np.zeros(len(self.z), dtype=bool)
        self.confirmed_present = np.zeros(len(self.z), dtype=bool)
        self.no_tracks = [set() for _ in self.z]
        self.events = []

    @property
    def prior(self):
        value = self.a[:, None] + self.slope*self.z
        value[:, self.frozen_tracks] = self.initial_prior[:, self.frozen_tracks]
        return value

    @property
    def logits(self):
        result = self.prior + self.evidence
        result[self.absent] = -np.inf
        return result

    @property
    def posterior(self): return expit(self.logits)

    def observe(self, query, track, answer, view, reliability, *, presence_question=False):
        s, c = reliability
        delta = evidence_increment(answer, s, c)
        self.evidence[query, track] += delta
        self.events.append(dict(query=int(query), track=int(track), answer=bool(answer), view=view,
            delta=delta, sensitivity=float(s), specificity=float(c), presence_question=presence_question))
        if presence_question:
            if answer:
                self.low[query] = False
                self.confirmed_present[query] = True
                self.no_tracks[query].clear()
                # A Yes certifies at least one target; it does not force the
                # candidate posterior positive or fabricate another target.
                self.a[query] = solve_intercept(self.z[query], max(1., self.expected[query]), self.slope)
            else:
                self.no_tracks[query].add(int(track))
                if len(self.no_tracks[query]) >= 2:
                    self.absent[query] = True

    def replace_prior(self, *, intercept=None, slope=None, scores=None):
        if intercept is not None: self.a = np.asarray(intercept, dtype=float).copy()
        if slope is not None: self.slope = float(slope)
        if scores is not None: self.z = normalize_track_scores(scores)
        # No teacher increment is recomputed here. Each saved delta is added
        # exactly once, even after multiple fast/slow refits.
        return self.posterior

    def pseudo_labels(self):
        p = self.posterior
        labels = np.where(p > .95, 1, np.where(p < .05, 0, -1)).astype('int8')
        labels[:, self.frozen_tracks] = -1
        return labels
