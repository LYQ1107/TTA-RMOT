"""Teacher-observed conditional state learning; no evaluation-label access.

The clean referring state is latent. A transition either keeps the preceding
state or draws a new state from the neural solver. Teacher decisions are noisy
observations, not clean-state targets. A clip has ONE latent observation time
among its actual input observations; it does not create four independent votes.

Forward/backward is exact for fixed node likelihoods. Clip-time responsibilities
use a structured variational bound, not an exact clip marginal. Neural EM and
selective state updates are prior ideas, not by themselves new contributions or
evidence of RSI. Semantic orientation and temporal continuity are assumptions.
"""
from dataclasses import dataclass
import numpy as np
from scipy.special import expit, logsumexp
import torch
from torch import nn


VIEWS = ('CROP', 'CLIP', 'FRAME_BOX')


@dataclass(frozen=True)
class ScopeEvent:
    query: int
    nodes: tuple
    answer: int
    view: int
    uid: str


def validate_events(events, queries, observations):
    uids = set()
    for event in events:
        if event.uid in uids:
            raise ValueError('Repeated teacher event: ' + event.uid)
        uids.add(event.uid)
        if not 0 <= event.query < queries or event.answer not in (0, 1):
            raise ValueError('Invalid query/decision')
        if event.view not in range(3) or not event.nodes:
            raise ValueError('Invalid view/empty observation scope')
        if len(set(event.nodes)) != len(event.nodes):
            raise ValueError('Repeated physical observation inside one event')
        if min(event.nodes) < 0 or max(event.nodes) >= observations:
            raise ValueError('Observation outside provided candidate bank')
        if event.view != 1 and len(event.nodes) != 1:
            raise ValueError('A frame/crop decision must have one observation')


def sequence_layout(observations):
    """All provided observations, retaining physical IDs and actual time gaps."""
    obs = np.asarray(observations, dtype=np.int64)
    if obs.ndim != 2 or obs.shape[1] != 2 or len(obs) == 0:
        raise ValueError('Expected nonempty [frame, candidate-ID] observations')
    if len(np.unique(obs, axis=0)) != len(obs):
        raise ValueError('Duplicate candidate/frame observation')
    previous = np.full(len(obs), -1, dtype=np.int64)
    tracks = []
    for identity in np.unique(obs[:, 1]):
        rows = np.flatnonzero(obs[:, 1] == identity)
        rows = rows[np.argsort(obs[rows, 0], kind='stable')]
        if np.any(np.diff(obs[rows, 0]) <= 0):
            raise ValueError('Non-increasing time within candidate')
        previous[rows[1:]] = rows[:-1]
        tracks.append(rows)
    levels = tuple(np.array([rows[k] for rows in tracks if k < len(rows)],
                            dtype=np.int64)
                   for k in range(max(map(len, tracks))))
    return dict(previous=previous, levels=levels, tracks=tuple(tracks))


def _probability(value, name, shape=None):
    value = np.asarray(value, dtype=np.float64)
    if shape is not None and value.shape != shape:
        raise ValueError(name + ' shape')
    if not np.isfinite(value).all() or np.any((value < 0) | (value > 1)):
        raise ValueError(name + ' range')
    return value


def forward_backward(prior, keep, log0, log1, layout):
    """Exact binary copy/reset HMM and EM sufficient statistics.

    P(z_t=b|z_prev=a,X) = r_t I[a=b] + (1-r_t) Ber(b;p_t).
    At a sequence start the state is drawn from Ber(p_t). The copy/reset
    indicators are part of the latent model, so reset masses must NOT be
    replaced with ordinary semantic posteriors in the solver M step.
    """
    p = _probability(prior, 'prior')
    if p.ndim != 2:
        raise ValueError('prior must be [query, observation]')
    r = _probability(keep, 'keep', p.shape)
    l0, l1 = (np.asarray(x, dtype=np.float64) for x in (log0, log1))
    if l0.shape != p.shape or l1.shape != p.shape:
        raise ValueError('likelihood shape')
    if not np.isfinite(l0).all() or not np.isfinite(l1).all():
        raise ValueError('finite observation log likelihoods required')
    previous, levels = layout['previous'], layout['levels']
    if len(previous) != p.shape[1]:
        raise ValueError('sequence bank mismatch')
    alpha = np.empty_like(p)
    predictive = np.empty_like(p)
    normalizers = np.empty_like(p)
    for depth, rows in enumerate(levels):
        pred = p[:, rows] if depth == 0 else (
            r[:, rows] * alpha[:, previous[rows]] + (1-r[:, rows])*p[:, rows])
        predictive[:, rows] = pred
        a0 = np.log(np.maximum(1-pred, 1e-300)) + l0[:, rows]
        a1 = np.log(np.maximum(pred, 1e-300)) + l1[:, rows]
        # Keep exact deterministic branches for the enumeration contract.
        a0 = np.where(pred == 1, -np.inf, a0)
        a1 = np.where(pred == 0, -np.inf, a1)
        norm = np.logaddexp(a0, a1)
        if not np.isfinite(norm).all():
            raise FloatingPointError('Impossible observation under the model')
        alpha[:, rows] = np.exp(a1-norm)
        normalizers[:, rows] = norm
    beta = np.full_like(p, .5)
    posterior, reset_positive, reset_negative = (np.empty_like(p) for _ in range(3))
    copy = np.zeros_like(p)
    for depth in range(len(levels)-1, -1, -1):
        rows = levels[depth]
        b = beta[:, rows]
        a = alpha[:, rows]
        den = a*b+(1-a)*(1-b)
        posterior[:, rows] = a*b/den
        if depth == 0:
            reset_positive[:, rows] = posterior[:, rows]
            reset_negative[:, rows] = 1-posterior[:, rows]
            continue
        scale = np.maximum(l0[:, rows], l1[:, rows])
        u0 = np.exp(l0[:, rows]-scale)*(1-b)
        u1 = np.exp(l1[:, rows]-scale)*b
        pv, rv, av = p[:, rows], r[:, rows], alpha[:, previous[rows]]
        pred = predictive[:, rows]
        denominator = (1-pred)*u0+pred*u1
        reset_positive[:, rows] = (1-rv)*pv*u1/denominator
        reset_negative[:, rows] = (1-rv)*(1-pv)*u0/denominator
        copy[:, rows] = rv*((1-av)*u0+av*u1)/denominator
        reset_mix = (1-rv)*((1-pv)*u0+pv*u1)
        b0, b1 = rv*u0+reset_mix, rv*u1+reset_mix
        beta[:, previous[rows]] = b1/(b0+b1)
    result = dict(posterior=posterior, reset_positive=reset_positive,
                  reset_negative=reset_negative, copy=copy,
                  log_evidence=normalizers.sum(1), alpha=alpha,
                  log0=l0, log1=l1)
    mass = reset_positive+reset_negative+copy
    if np.max(np.abs(mass-1)) > 2e-10:
        raise FloatingPointError('Copy/reset posterior masses do not sum to one')
    if not all(np.isfinite(result[k]).all() for k in
               ('posterior', 'reset_positive', 'reset_negative', 'copy', 'log_evidence')):
        raise FloatingPointError('Nonfinite state inference')
    return result


def event_log_likelihood(events, reliability):
    """Bernoulli teacher DECISIONS, not a density for continuous probabilities.

    Original soft probabilities stay immutable and remain available to all
    ordinary controls. This latent model deliberately uses their hard decision.
    Sensitivity/specificity are oriented above chance as an explicit assumption;
    that assumption does not identify semantic truth from this one teacher.
    """
    rel = _probability(reliability, 'reliability', (3, 2))
    if np.any((rel <= 0) | (rel >= 1)):
        raise ValueError('Strictly interior channel probabilities required')
    l0, l1 = [], []
    for event in events:
        sensitivity, specificity = rel[event.view]
        l0.append(np.log(1-specificity if event.answer else specificity))
        l1.append(np.log(sensitivity if event.answer else 1-sensitivity))
    return np.array(l0), np.array(l1)


def scoped_bound(prior, keep, observations, events, reliability, *,
                 protocol='PRESUPPOSED', existence_prior=.5,
                 iterations=8, responsibilities=None):
    """Structured ELBO with one latent uniform clip-time per teacher event.

    E=0 is the checklist hypothesis that all provided referring states are 0.
    E=1 uses the HMM and still permits no candidate match. PRESUPPOSED fixes E=1
    at VIDEO scope. No frame is forced positive, including missing-candidate
    cases. Unqueried candidates are retained and no acquisition is stopped.
    """
    p = _probability(prior, 'prior')
    if p.ndim != 2:
        raise ValueError('prior shape')
    nq, n = p.shape
    validate_events(events, nq, n)
    layout = sequence_layout(observations)
    if len(layout['previous']) != n or protocol not in ('CHECKLIST', 'PRESUPPOSED'):
        raise ValueError('Invalid protocol/observation bank')
    obs = np.asarray(observations)
    for event in events:
        local = obs[list(event.nodes)]
        if len(np.unique(local[:, 1])) != 1 or np.any(np.diff(local[:, 0]) <= 0):
            raise ValueError('An event scope must be chronological and keep its candidate ID')
    rho = np.broadcast_to(_probability(existence_prior, 'existence'), (nq,))
    if protocol == 'CHECKLIST' and np.any((rho <= 0) | (rho >= 1)):
        raise ValueError('Checklist existence prior must be interior')
    if iterations < 1:
        raise ValueError('At least one variational iteration')
    ll0, ll1 = event_log_likelihood(events, reliability)
    scope = [np.asarray(e.nodes, dtype=np.int64) for e in events]
    if responsibilities is None:
        weights = [np.full(len(nodes), 1/len(nodes)) for nodes in scope]
    else:
        if len(responsibilities) != len(scope):
            raise ValueError('responsibility length')
        weights = [np.asarray(w, dtype=float).copy() for w in responsibilities]
        if any(w.shape != ids.shape or (w < 0).any() or not np.isclose(w.sum(), 1)
               for w, ids in zip(weights, scope)):
            raise ValueError('Invalid event responsibilities')
    absent_log = np.zeros(nq)
    for event, value in zip(events, ll0):
        absent_log[event.query] += value
    history = []
    for iteration in range(iterations):
        l0, l1, entropy = np.zeros_like(p), np.zeros_like(p), np.zeros(nq)
        for j, (event, nodes, w) in enumerate(zip(events, scope, weights)):
            l0[event.query, nodes] += w*ll0[j]
            l1[event.query, nodes] += w*ll1[j]
            entropy[event.query] -= np.sum(w*np.log(np.maximum(w, 1e-300)*len(w)))
        chain = forward_backward(p, keep, l0, l1, layout)
        present_bound = chain['log_evidence']+entropy
        if protocol == 'PRESUPPOSED':
            eta, bound = np.ones(nq), present_bound
        else:
            log_yes = np.log(rho)+present_bound
            log_no = np.log1p(-rho)+absent_log
            bound = np.logaddexp(log_yes, log_no)
            eta = np.exp(log_yes-bound)
        history.append(bound.copy())
        if iteration+1 == iterations:
            break
        updated = []
        for j, (event, nodes) in enumerate(zip(events, scope)):
            q = chain['posterior'][event.query, nodes]
            # Optimize J | E=1. For E=0 all times have the same emission,
            # hence J can analytically remain uniform and has zero KL cost.
            logw = (1-q)*ll0[j]+q*ll1[j]
            updated.append(np.exp(logw-logsumexp(logw)))
        weights = updated
    trace = np.stack(history)
    if len(trace) > 1 and np.min(np.diff(trace, axis=0)) < -2e-9:
        raise FloatingPointError('Structured coordinate bound decreased')
    result = dict(chain)
    result.update(posterior=eta[:, None]*chain['posterior'],
                  conditional_posterior=chain['posterior'],
                  reset_positive=eta[:, None]*chain['reset_positive'],
                  reset_negative=eta[:, None]*chain['reset_negative'],
                  copy=eta[:, None]*chain['copy'], existence=eta,
                  bound=bound, present_bound=present_bound,
                  absent_log_evidence=absent_log, responsibilities=weights,
                  coordinate_bound_trace=trace, layout=layout,
                  exact_observation_marginal=all(len(e.nodes) == 1 for e in events))
    return result


def complete_state_objective(prior, keep, expected, *, query_weight=None):
    """Expected state log likelihood to audit the neural M step.

    The full observed-data bound must still be recomputed before accepting an
    approximate sampled neural update. This helper does not certify that step.
    """
    p = np.clip(_probability(prior, 'prior'), 1e-12, 1-1e-12)
    r = np.clip(_probability(keep, 'keep', p.shape), 1e-12, 1-1e-12)
    qw = np.ones(p.shape[0]) if query_weight is None else np.asarray(query_weight)
    if qw.shape != (p.shape[0],) or np.any(qw < 0):
        raise ValueError('query weight')
    start = expected['layout']['previous'] < 0
    value = expected['reset_positive']*np.log(p)+expected['reset_negative']*np.log1p(-p)
    remaining = ~start
    value[:, remaining] += (expected['copy'][:, remaining]*np.log(r[:, remaining])
                            + (expected['reset_positive'][:, remaining]
                               + expected['reset_negative'][:, remaining])*np.log1p(-r[:, remaining]))
    return float(np.sum(qw[:, None]*value))


def accepted_fraction(before, evaluate, *, tolerance=1e-10):
    """Registered objective backtracking, never GT/checkpoint selection."""
    trials = []
    for fraction in (1., .5, .25, .125, .0625, .03125):
        after = float(evaluate(fraction))
        trials.append(dict(fraction=fraction, objective=after))
        if np.isfinite(after) and after <= before+tolerance:
            return fraction, trials
    return 0., trials


class StateObserver(nn.Module):
    """Learn input-conditioned copy probability and oriented teacher channels.

    Inputs must be observed image/state changes and time gaps. Neither a GT
    label nor R/S agreement may be inserted as a true copy/reset target.
    Expected latent counts are model-dependent EM statistics, not true labels.
    """
    def __init__(self, dimension=8):
        super().__init__()
        self.copy_network = nn.Sequential(nn.Linear(dimension, 32), nn.SiLU(),
                                          nn.Linear(32, 1))
        nn.init.zeros_(self.copy_network[-1].weight)
        nn.init.zeros_(self.copy_network[-1].bias)
        initial = torch.logit(torch.tensor((.8-.5)/.49))
        self.channel_raw = nn.Parameter(torch.full((3, 2), float(initial)))

    def forward(self, change, base_keep):
        base = base_keep.clamp(.001, .999)
        return torch.sigmoid(torch.logit(base)+self.copy_network(change).flatten())

    def reliability(self):
        return .5+.49*self.channel_raw.sigmoid()


def observable_changes(observations, appearance, query_state, boxes, dynamic):
    """Eight observed change covariates, shared with ordinary controls.

    Distances and median scales are descriptive, not identity/semantic truth.
    The initial 0.02/0.05 hazards are fixed modeling priors, not GT-tuned
    constants. They are measured per median observed time gap. The neural
    writer may learn to retain or reset instead of enforcing constant IDs.
    """
    obs = np.asarray(observations)
    layout = sequence_layout(obs)
    appearance = np.asarray(appearance, dtype=float)
    state = np.asarray(query_state, dtype=float)
    boxes = np.asarray(boxes, dtype=float)
    dynamic = np.asarray(dynamic, dtype=bool)
    nq, n = state.shape[:2]
    if appearance.shape[0] != n or boxes.shape != (n, 4) or dynamic.shape != (nq,):
        raise ValueError('Observed writer inputs disagree')
    previous = layout['previous']
    has_previous = previous >= 0
    before = np.maximum(previous, 0)
    gap = np.where(has_previous, obs[:, 0]-obs[before, 0], 0).astype(float)
    median_gap = max(float(np.median(gap[has_previous])) if has_previous.any() else 1., 1.)

    def cosine(x, y):
        return np.clip(1-(x*y).sum(-1)/np.maximum(
            np.linalg.norm(x, axis=-1)*np.linalg.norm(y, axis=-1), 1e-12), 0, 2)

    def scaled(x):
        positive = x[..., has_previous]
        positive = positive[positive > 1e-12]
        scale = max(float(np.median(positive)) if positive.size else 1., 1e-12)
        return np.minimum(x/scale, 10)

    app = scaled(cosine(appearance, appearance[before]))
    semantic = scaled(cosine(state, state[:, before]))
    size = np.maximum(boxes[:, 2:], 1e-6)
    centre = scaled(np.linalg.norm((boxes[:, :2]-boxes[before, :2])/
                                  np.maximum(size, size[before]), axis=-1))
    resizing = scaled(np.abs(np.log(size)-np.log(size[before])).mean(-1))
    changes = np.empty((nq, n, 8), dtype=np.float32)
    changes[:, :, 0] = np.minimum(gap/median_gap, 10)
    changes[:, :, 1] = app
    changes[:, :, 2] = semantic
    changes[:, :, 3] = centre
    changes[:, :, 4] = resizing
    changes[:, :, 5] = dynamic[:, None]
    changes[:, :, 6] = gap > median_gap
    changes[:, :, 7] = has_previous
    changes[:, ~has_previous] = 0
    hazard = np.where(dynamic, .05, .02)
    base_keep = np.exp(np.log1p(-hazard[:, None])*gap[None, :]/median_gap)
    base_keep[:, ~has_previous] = 0
    return changes, base_keep.astype('float32'), layout


def channel_expected_counts(events, expected):
    """Soft truth/decision counts from one observation likelihood per event."""
    counts = np.zeros((3, 2, 2), dtype=float)
    for event, w in zip(events, expected['responsibilities']):
        q = event.query
        # J|E=0 is uniform: all states are zero there. The E=1 contribution
        # uses its separately optimized responsibility, avoiding a spurious
        # existence-dependent multiplicity of a clip observation.
        positive = float(np.dot(w, expected['conditional_posterior'][q, list(event.nodes)]))
        positive *= expected['existence'][q]
        counts[event.view, 1, event.answer] += positive
        counts[event.view, 0, event.answer] += 1-positive
    return counts
