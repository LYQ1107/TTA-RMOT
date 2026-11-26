import unittest

import torch

from rmot_tta.adaptation.controller import AdaptationController
from rmot_tta.evidence.records import CandidateEvidence, EvidenceLabel
from rmot_tta.types import FrameSnapshot


def snapshot(frame: int) -> FrameSnapshot:
    features = torch.randn(2, 256)
    logits = torch.tensor([-1.0, 1.0])
    return FrameSnapshot("ep", "v", "expr", "cars", frame, (f"u{frame}a", f"u{frame}b"), features,
                         logits, logits.sigmoid(), torch.tensor([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]]),
                         torch.ones(2), torch.tensor([1, 2]), torch.zeros(2, dtype=torch.long), (40, 40), torch.ones(2, dtype=torch.bool))


class FrozenParameterTest(unittest.TestCase):
    def test_only_external_adapter_changes_and_real_step_occurs(self):
        weight = torch.zeros(1, 256)
        bias = torch.zeros(1)
        controller = AdaptationController("QCS_TTA", weight, bias, update_every_n_frames=1)
        source_weight = controller.head257.linear.weight.detach().clone()
        current = snapshot(0)
        evidence = [CandidateEvidence(uid, EvidenceLabel.SUPPORTED_POS, 0.9, 1.0, "EQ_SELF_PROBE", "q0", 1.0, "test", 0, int(track))
                    for uid, track in zip(current.candidate_uid, current.pred_track_id.tolist())]
        controller.observe(current, evidence)
        result = controller.maybe_adapt(current, evidence)
        self.assertTrue(result.stepped)
        self.assertGreater(controller.update_steps, 0)
        self.assertTrue(torch.equal(source_weight, controller.head257.linear.weight.detach()))
        self.assertTrue(any(float(parameter.detach().abs().sum()) > 0 for parameter in controller.adapter.parameters()))


if __name__ == "__main__":
    unittest.main()
