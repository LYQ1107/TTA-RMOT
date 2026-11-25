"""Semantically oriented, correlated teacher-opinion emission patterns.

Every binary3-view pattern remains possible for both latent task classes.
The Dirichlet prior is weak guidance, not certified labels or majority truth.
Learned calibration prevalence belongs to the selected evidence pool only.
"""
import copy
import itertools
import numpy as np

from flex_uda.active.joint_teacher_noise import predict

PATTERNS = np.asarray(list(itertools.product((0, 1), repeat=3)), dtype=int)


class SemanticPatterns:
    def __init__(self, table=None, prevalence=.5):
        self.base = np.asarray([np.prod(np.where(PATTERNS == y, .8, .2), axis=1) for y in (0, 1)])
        self.table = self.base.copy() if table is None else np.asarray(table, dtype='float64').copy()
        self.prevalence = float(prevalence)
        assert self.table.shape == (2, 8) and (self.table > 0).all()
        assert np.allclose(self.table.sum(1), 1.) and 0 < self.prevalence < 1

    def state(self):
        return dict(table=self.table.tolist(), prevalence=self.prevalence,
                    prior_strength=10., prior_view_correct_probability=.8,
                    prevalence_is_selected_evidence_not_video_absence=True,
                    every_pattern_possible_for_both_classes=True)

    @classmethod
    def from_state(cls, state):
        return cls(state['table'], state['prevalence'])

    @staticmethod
    def compatible(probability):
        p = np.asarray(probability, dtype='float64')
        assert p.ndim == 2 and p.shape[1] == 3
        known = np.isfinite(p) & (p != .5)
        yes = p > .5
        return np.all((yes[:, None, :] == PATTERNS[None, :, :]) | ~known[:, None, :], axis=2).astype(float)

    def factors(self, probability):
        compatible = self.compatible(probability)
        evidence = compatible@self.table.T
        assert (evidence > 0).all()
        return compatible, evidence

    def prior(self, probability, image_prior):
        if image_prior is None:
            return np.full(len(probability), self.prevalence)
        prior = np.asarray(image_prior, dtype='float64')
        assert prior.shape == (len(probability),)
        return np.clip(prior, .05, .95)

    def posterior(self, probability, image_prior=None):
        _, evidence = self.factors(probability)
        prior = self.prior(probability, image_prior)
        mass = evidence*np.column_stack((1-prior, prior))
        return mass[:, 1]/mass.sum(1)

    def log_likelihood(self, probability, image_prior=None):
        _, evidence = self.factors(probability)
        prior = self.prior(probability, image_prior)
        return float(np.log((1-prior)*evidence[:, 0]+prior*evidence[:, 1]).sum())

    def objective(self, probability, image_prior=None):
        regularizer = float(10*np.sum(self.base*np.log(self.table)))
        if image_prior is None:
            regularizer += np.log(self.prevalence)+np.log1p(-self.prevalence)
        return self.log_likelihood(probability, image_prior)+regularizer

    def update(self, probability, image_prior=None):
        compatible, evidence = self.factors(probability)
        positive = self.posterior(probability, image_prior)
        responsibilities = np.column_stack((1-positive, positive))
        counts = self.table*((responsibilities/evidence).T@compatible)
        target = counts+10*self.base
        target /= target.sum(1, keepdims=True)
        prevalence = ((positive.sum()+1)/(len(positive)+2)) if image_prior is None else self.prevalence
        before = self.objective(probability, image_prior)
        raw_before = self.log_likelihood(probability, image_prior)
        old_table, old_prevalence = self.table.copy(), self.prevalence
        accepted, after = 0., before
        expected_yes = PATTERNS.sum(1)/3
        for fraction in (1., .5, .25, .125, .0625, .03125, .015625, .0078125, .00390625):
            self.table = old_table+fraction*(target-old_table)
            self.prevalence = old_prevalence+fraction*(prevalence-old_prevalence)
            if self.table[1]@expected_yes <= self.table[0]@expected_yes:
                continue
            value = self.objective(probability, image_prior)
            if np.isfinite(value) and value >= before-1e-9:
                accepted, after = fraction, value
                break
        if not accepted:
            self.table, self.prevalence = old_table, old_prevalence
        return dict(before=before, after=after, raw_log_likelihood_before=raw_before,
                    raw_log_likelihood_after=self.log_likelihood(probability, image_prior),
                    accepted_fraction=accepted, selected_prevalence=self.prevalence,
                    maximum_parameter_change=max(float(np.abs(self.table-old_table).max()), abs(self.prevalence-old_prevalence)),
                    objective_includes_Dirichlet_prior_not_GT=True)

    def fit(self, probability, maximum=200):
        history = []
        for _ in range(maximum):
            row = self.update(probability)
            history.append(row)
            if row['after']-row['before'] <= 1e-8*(1+abs(row['before'])):
                break
        return history


def accept_solver(model, previous_state, observer, probability, features, device):
    proposed = copy.deepcopy(model.state_dict())
    model.load_state_dict(previous_state)
    before = observer.objective(probability, predict(model, features, device))
    accepted, after = 0., before
    for fraction in (1., .5, .25, .125, .0625, .03125, .015625, .0078125, .00390625):
        model.load_state_dict({k: previous_state[k]+fraction*(proposed[k]-previous_state[k]) for k in proposed})
        value = observer.objective(probability, predict(model, features, device))
        if np.isfinite(value) and value >= before-1e-7:
            accepted, after = fraction, value
            break
    if not accepted:
        model.load_state_dict(previous_state)
    return dict(before=before, after=after, accepted_fraction=accepted,
                maximum_parameter_change=max(float((v-previous_state[k]).abs().max()) for k, v in model.state_dict().items()),
                posterior_and_prior_not_ground_truth=True)
