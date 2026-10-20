import unittest

import torch

from rmot_tta.adaptation.losses import (
    bernoulli_entropy_from_logits,
    equivalent_query_consistency,
    mutual_exclusion_penalty,
    source_anchor_loss,
    weighted_bce_with_logits,
)


class SetLossTest(unittest.TestCase):
    def test_multiple_positive_candidates_are_not_mutually_excluded(self):
        logits = torch.tensor([4.0, 4.0])
        target = torch.ones(2)
        self.assertLess(float(weighted_bce_with_logits(logits, target)), 0.1)
        self.assertEqual(float(mutual_exclusion_penalty(torch.tensor([0.9]), torch.tensor([0.0]))), 0.0)

    def test_empty_and_masked_batches_are_finite(self):
        empty = torch.empty(0)
        self.assertEqual(float(weighted_bce_with_logits(empty, empty)), 0.0)
        self.assertEqual(float(equivalent_query_consistency(empty, empty)), 0.0)
        masked = torch.tensor([1.0, -1.0])
        self.assertEqual(float(weighted_bce_with_logits(masked, torch.ones(2), mask=torch.zeros(2, dtype=torch.bool))), 0.0)
        self.assertTrue(torch.isfinite(bernoulli_entropy_from_logits(torch.tensor([-1000.0, 1000.0]))).all())
        self.assertTrue(torch.isfinite(source_anchor_loss(torch.tensor([0.0]), torch.tensor([0.0]))))

    def test_double_low_counterfactual_is_not_forced(self):
        self.assertEqual(float(mutual_exclusion_penalty(torch.tensor([0.0]), torch.tensor([0.0]))), 0.0)


if __name__ == "__main__":
    unittest.main()
