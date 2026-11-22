"""Independent conditional identities, rather than a HOTA guarantee."""
import numpy as np
from flex_uda.active.relational_transport import fit, center_groups
from scripts.a100_relation_theory_audit import quadratic_reference


def test_excitation_changes_ridge_bias_under_the_exact_local_model():
    truth = .8
    mu = .1
    previous_bias = float('inf')
    for spread in (0., .05, .2, .4):
        rows = [dict(group=('v', str(g), g, 'CROP'),
            x=float(.5 + spread * offset), teacher_logit=float(g * .2 + truth * (.5 + spread * offset)),
            weight=1.) for g in range(3) for offset in (-1, 0, 1)]
        x, y, w, _ = center_groups(rows)
        reference, gram = quadratic_reference(x, y, w, mu)
        beta, receipt = fit(rows, mu=mu)
        assert np.all(np.abs(beta*x-y) < 1.)
        assert abs(beta-reference) < 1e-10
        assert abs(beta - gram/(gram+mu)*truth) < 1e-10
        bias = abs(beta - truth)
        assert bias <= previous_bias
        previous_bias = bias
        if spread == 0:
            assert receipt['excited_groups'] == 0 and beta == 0.


def test_without_within_variation_distinct_slopes_have_the_same_observations():
    position = np.array([.1, .4, .8])
    intercept = np.array([.2, -.3, 1.])
    first, second = .7, 4.1
    teacher_logit = intercept + first * position
    alternative_intercept = intercept + (first-second) * position
    np.testing.assert_allclose(teacher_logit, alternative_intercept + second*position, atol=1e-12)
    # Unique regularized numerical output would not distinguish these two
    # exactly observationally equivalent explanations.
