import unittest

import torch

from rmot_tta.probes.alignment import align_probe_results
from rmot_tta.types import FrameSnapshot, ProbeCandidate, ProbeResult


class SnapshotAlignmentTest(unittest.TestCase):
    def test_source_head_and_box_alignment(self):
        features = torch.zeros(2, 256)
        features[0, 0] = 2.0
        features[1, 0] = -2.0
        weight = torch.zeros(1, 256)
        weight[0, 0] = 1.5
        bias = torch.tensor([0.25])
        logits = torch.nn.functional.linear(features, weight, bias).reshape(-1)
        snapshot = FrameSnapshot(
            "ep", "0001", "expr", "cars", 3, ("u0", "u1"), features, logits, logits.sigmoid(),
            torch.tensor([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]]),
            torch.ones(2), torch.tensor([1, 2]), torch.zeros(2, dtype=torch.long), (40, 40), torch.ones(2, dtype=torch.bool),
        )
        self.assertTrue(torch.allclose(torch.nn.functional.linear(snapshot.features, weight, bias).reshape(-1), snapshot.source_logits))
        probe = ProbeResult("q0", 3, "cars", (
            ProbeCandidate("shadow0", (0.0, 0.0, 10.0, 10.0), 0.9, 2.2, 1),
            ProbeCandidate("shadow1", (20.0, 20.0, 30.0, 30.0), 0.1, -2.2, 2),
        ), 0.0, (3,))
        aligned = align_probe_results(snapshot, [probe], min_iou=0.6)
        self.assertEqual([item.main_index for item in aligned], [0, 1])
        self.assertAlmostEqual(aligned[0].r0, 0.9)
        self.assertAlmostEqual(aligned[1].r0, 0.1)


if __name__ == "__main__":
    unittest.main()
