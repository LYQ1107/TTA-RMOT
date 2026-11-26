"""Joint image solver and correlated, possibly missing teacher observations.

The two Gaussian components are a modeling hypothesis, not verified semantic
classes. Their orientation is fixed by the teacher Yes convention. Generalized
EM checks the actual observed-teacher likelihood, never target annotations.
"""
import copy
import math
import time

import numpy as np
from scipy.special import expit, logsumexp
import torch
from torch import nn


class CorrelatedTeacher:
    def __init__(self, mean, covariance):
        self.mean = np.asarray(mean, dtype='float64').copy()
        self.covariance = np.asarray(covariance, dtype='float64').copy()
        assert self.mean.shape == (2, 3) and self.covariance.shape == (2, 3, 3)

    @staticmethod
    def observations(probability):
        p = np.asarray(probability, dtype='float64')
        assert p.ndim == 2 and p.shape[1] == 3 and np.isfinite(p).any(1).all()
        known = np.isfinite(p)
        assert ((p[known] >= 0) & (p[known] <= 1)).all()
        p = np.clip(p, .001, .999)
        return np.log(p)-np.log1p(-p), known

    @classmethod
    def initialize(cls, probability):
        z, mask = cls.observations(probability)
        assert mask.any(0).all(), 'Each channel must have actual calibration evidence'
        low = np.minimum(np.nanquantile(z, .2, axis=0), -1.)
        high = np.maximum(np.nanquantile(z, .8, axis=0), 1.)
        diagonal = np.maximum(np.nanvar(z, axis=0), 1.)
        return cls(np.stack((low, high)), np.tile(np.diag(diagonal), (2, 1, 1)))

    def state(self):
        return dict(mean=self.mean.tolist(), covariance=self.covariance.tolist(),
                    component_orientation='Mean teacher Yes logit of component1 exceeds component0; not GT orientation',
                    covariance_eigenvalue_floor=.01, probability_logit_clip=[.001, .999])

    @classmethod
    def from_state(cls, state):
        return cls(state['mean'], state['covariance'])

    def log_density(self, probability):
        z, known = self.observations(probability)
        result = np.empty((len(z), 2))
        for pattern in np.unique(known, axis=0):
            rows = np.flatnonzero(np.all(known == pattern, axis=1))
            axes = np.flatnonzero(pattern)
            for component in range(2):
                covariance = self.covariance[component][np.ix_(axes, axes)]
                cholesky = np.linalg.cholesky(covariance)
                delta = z[np.ix_(rows, axes)]-self.mean[component, axes]
                scaled = np.linalg.solve(cholesky, delta.T).T
                result[rows, component] = -.5*(len(axes)*math.log(2*math.pi)
                    +2*np.log(np.diag(cholesky)).sum()+np.square(scaled).sum(1))
        return result

    def joint_logs(self, probability, prior=None):
        if prior is None:
            prior = np.full(len(probability), .5)
        prior = np.clip(np.asarray(prior, dtype='float64'), .05, .95)
        assert prior.shape == (len(probability),)
        return self.log_density(probability)+np.column_stack((np.log1p(-prior), np.log(prior)))

    def posterior(self, probability, prior=None):
        joint = self.joint_logs(probability, prior)
        return np.exp(joint[:, 1]-logsumexp(joint, axis=1))

    def likelihood(self, probability, prior=None):
        return float(logsumexp(self.joint_logs(probability, prior), axis=1).sum())

    def update(self, probability, prior=None):
        """Missing-view conditional moments, followed by an observed-LL check."""
        z, known = self.observations(probability)
        posterior = self.posterior(probability, prior)
        responsibilities = np.column_stack((1-posterior, posterior))
        new_mean, new_covariance = [], []
        for component in range(2):
            total = float(responsibilities[:, component].sum())
            if total < 1e-8:
                new_mean.append(self.mean[component].copy())
                new_covariance.append(self.covariance[component].copy())
                continue
            first, second = np.zeros(3), np.zeros((3, 3))
            for pattern in np.unique(known, axis=0):
                rows = np.flatnonzero(np.all(known == pattern, axis=1))
                observed, missing = np.flatnonzero(pattern), np.flatnonzero(~pattern)
                expectation = np.tile(self.mean[component], (len(rows), 1))
                expectation[:, observed] = z[np.ix_(rows, observed)]
                residual = np.zeros((3, 3))
                if len(missing):
                    covariance = self.covariance[component]
                    cross = covariance[np.ix_(missing, observed)]
                    transform = np.linalg.solve(covariance[np.ix_(observed, observed)], cross.T).T
                    expectation[:, missing] += (z[np.ix_(rows, observed)]-self.mean[component, observed])@transform.T
                    residual[np.ix_(missing, missing)] = covariance[np.ix_(missing, missing)]-transform@cross.T
                weight = responsibilities[rows, component]
                first += weight@expectation
                second += (expectation*weight[:, None]).T@expectation+weight.sum()*residual
            mean = first/total
            covariance = second/total-np.outer(mean, mean)
            covariance = (covariance+covariance.T)/2
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            covariance = (eigenvectors*np.maximum(eigenvalues, .01))@eigenvectors.T
            new_mean.append(mean)
            new_covariance.append(covariance)
        before = self.likelihood(probability, prior)
        old_mean, old_covariance = self.mean.copy(), self.covariance.copy()
        accepted = 0.
        after = before
        for fraction in (1., .5, .25, .125, .0625, .03125, .015625, .0078125, .00390625):
            self.mean = old_mean+fraction*(np.asarray(new_mean)-old_mean)
            self.covariance = old_covariance+fraction*(np.asarray(new_covariance)-old_covariance)
            if self.mean[1].mean() <= self.mean[0].mean():
                continue
            value = self.likelihood(probability, prior)
            if np.isfinite(value) and value >= before-1e-9:
                accepted, after = fraction, value
                break
        if not accepted:
            self.mean, self.covariance = old_mean, old_covariance
        return dict(before=before, after=after, accepted_fraction=accepted,
                    maximum_parameter_change=max(float(np.abs(self.mean-old_mean).max()),
                                                 float(np.abs(self.covariance-old_covariance).max())),
                    observed_teacher_likelihood_not_semantic_accuracy=True)

    def fit(self, probability, iterations=20):
        return [self.update(probability) for _ in range(iterations)]


class Solver(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(5508, 128), nn.SiLU(), nn.Linear(128, 128),
                                 nn.SiLU(), nn.Linear(128, 1))

    def forward(self, features):
        return self.net(features).flatten()


def make_solver(device):
    torch.manual_seed(0)
    return Solver().to(device)


def predict(model, features, device):
    model.eval()
    with torch.no_grad():
        return np.concatenate([model(torch.as_tensor(features[b:b+512], device=device)).sigmoid().cpu().numpy()
                               for b in range(0, len(features), 512)])


def entropy(probability):
    p = np.clip(np.asarray(probability, dtype='float64'), 1e-12, 1-1e-12)
    return -p*np.log(p)-(1-p)*np.log1p(-p)


def risk_floor(features, target, weight):
    x = np.ascontiguousarray(features, dtype='float32')
    _, inverse = np.unique(x.view(np.dtype((np.void, x.shape[1]*4))).ravel(), return_inverse=True)
    count = np.bincount(inverse, weights=weight)
    mean = np.bincount(inverse, weights=weight*target)/count
    return float((weight*entropy(mean)[inverse]).sum()/weight.sum())


def fit_solver(model, features, target, device, *, steps, weight=None, adequate=False):
    start = time.perf_counter()
    target = np.asarray(target, dtype='float64')
    weight = np.ones(len(target)) if weight is None else np.asarray(weight, dtype='float64')
    assert len(features) == len(target) and np.isfinite(target).all() and ((target >= 0) & (target <= 1)).all()
    assert np.isfinite(weight).all() and (weight > 0).all()
    floor = risk_floor(features, target, weight)
    x = torch.as_tensor(features, device=device)
    y = torch.as_tensor(target, dtype=torch.float32, device=device)
    w = torch.as_tensor(weight, dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    model.train()
    history = []
    maximum = max(steps, 16000) if adequate else steps
    excess = math.inf
    for iteration in range(maximum):
        optimizer.zero_grad(set_to_none=True)
        for begin in range(0, len(target), 512):
            loss = nn.functional.binary_cross_entropy_with_logits(model(x[begin:begin+512]), y[begin:begin+512], reduction='none')
            (loss*w[begin:begin+512]/w.sum()).sum().backward()
        optimizer.step()
        done = iteration+1
        if done == steps or done % 1000 == 0 or done == maximum:
            with torch.no_grad():
                z = torch.cat([model(x[b:b+512]).double() for b in range(0, len(target), 512)])
                ce = nn.functional.binary_cross_entropy_with_logits(z, torch.as_tensor(target, device=device), reduction='none')
                risk = float((ce*torch.as_tensor(weight, device=device)).sum()/weight.sum())
            excess = risk-floor
            history.append(dict(steps=done, proper_CE=risk, attainable_floor=floor, excess_KL=excess))
            if done >= steps and (not adequate or excess <= .005):
                break
    model.eval()
    if adequate:
        assert excess <= .005, ('Preserve ordinary proper fitting threshold', excess)
    return dict(steps=done, labels=len(target), risk=risk, floor=floor, excess_KL=excess,
                adequate_requested=adequate, adequate_passed=excess <= .005, history=history,
                wall_seconds=time.perf_counter()-start, seed=0, target_manual_GT_used=False)


def accept_solver(model, previous_state, observer, probability, features, device):
    proposed = copy.deepcopy(model.state_dict())
    model.load_state_dict(previous_state)
    before = observer.likelihood(probability, predict(model, features, device))
    accepted, after = 0., before
    for fraction in (1., .5, .25, .125, .0625, .03125, .015625, .0078125, .00390625):
        model.load_state_dict({k: previous_state[k]+fraction*(proposed[k]-previous_state[k]) for k in proposed})
        value = observer.likelihood(probability, predict(model, features, device))
        if np.isfinite(value) and value >= before-1e-7:
            accepted, after = fraction, value
            break
    if not accepted:
        model.load_state_dict(previous_state)
    return dict(before=before, after=after, accepted_fraction=accepted,
                maximum_parameter_change=max(float((v-previous_state[k]).abs().max()) for k, v in model.state_dict().items()),
                likelihood_only_not_GT_or_HOTA_selection=True)
