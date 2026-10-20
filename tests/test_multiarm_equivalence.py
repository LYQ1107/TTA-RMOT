import unittest

import torch

from rmot_tta.adaptation.controller import AdaptationController


class MultiarmEquivalenceTest(unittest.TestCase):
    def test_adaptation_state_is_not_shared(self):
        weight = torch.zeros(1, 256)
        bias = torch.zeros(1)
        left = AdaptationController("QCS_TTA", weight, bias)
        right = AdaptationController("NO_STATE", weight, bias)
        self.assertIsNot(left.adapter, right.adapter)
        self.assertIsNot(left.optimizer, right.optimizer)
        self.assertIsNot(left.memory, right.memory)


if __name__ == "__main__":
    unittest.main()
