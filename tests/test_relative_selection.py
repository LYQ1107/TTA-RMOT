import unittest

import torch

from rmot_tta.runtime.runner import _prediction_keep_mask


class RelativeSelectionTests(unittest.TestCase):
    def test_relative_gate_keeps_a_set_and_caps_without_absolute_threshold(self):
        ids = torch.tensor([1, 2, 3, -1])
        objectness = torch.ones(4)
        boxes = torch.tensor([
            [0.0, 0.0, 10.0, 10.0],
            [0.0, 0.0, 10.0, 10.0],
            [0.0, 0.0, 10.0, 10.0],
            [0.0, 0.0, 10.0, 10.0],
        ])
        logits = torch.tensor([-3.0, -3.2, -3.7, 4.0])
        config = {
            "objectness_threshold": 0.0,
            "area_threshold": 0.0,
            "referring_threshold": 0.99,
            "selection": {
                "mode": "relative_margin",
                "relative_logit_margin": 0.75,
                "max_candidates_per_frame": 2,
            },
        }
        keep = _prediction_keep_mask(ids, objectness, boxes, logits, config)
        self.assertEqual(keep.tolist(), [True, True, False, False])

    def test_absolute_gate_is_unchanged(self):
        ids = torch.tensor([1, 2])
        objectness = torch.ones(2)
        boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 10.0, 10.0]])
        logits = torch.tensor([0.0, -2.0])
        config = {"objectness_threshold": 0.0, "area_threshold": 0.0,
                  "referring_threshold": 0.4}
        keep = _prediction_keep_mask(ids, objectness, boxes, logits, config)
        self.assertEqual(keep.tolist(), [True, False])


if __name__ == "__main__":
    unittest.main()
