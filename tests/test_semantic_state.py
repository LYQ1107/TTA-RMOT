import unittest

import torch

from rmot_tta.evidence.records import EvidenceLabel
from rmot_tta.memory.semantic_memory import SemanticMemory


class SemanticStateTest(unittest.TestCase):
    def test_reliable_state_changes_preserve_old_samples(self):
        memory = SemanticMemory(capacity=16, per_track_capacity=16)
        feature = torch.ones(256)
        memory.observe("ep", "expr", 7, EvidenceLabel.SUPPORTED_POS, 0, feature, 0.0, 0.9, 1.0, "EQ_SELF_PROBE", "q0", "sig0", 1.0)
        memory.observe("ep", "expr", 7, EvidenceLabel.SUPPORTED_NEG, 4, feature * 2, 0.0, 0.0, 1.0, "CONTRAST_SELF_PROBE", "q_cf", "sig1", 1.0)
        memory.observe("ep", "expr", 7, EvidenceLabel.SUPPORTED_POS, 8, feature * 3, 0.0, 0.9, 1.0, "EQ_SELF_PROBE", "q0", "sig2", 1.0)
        support = memory.state("ep", "expr", 7)
        self.assertEqual(support.label, EvidenceLabel.SUPPORTED_POS)
        self.assertEqual(support.state_version, 2)
        self.assertEqual(support.pred_track_id, 7)
        samples = memory.replay.samples()
        self.assertEqual([sample.soft_target for sample in samples], [0.9, 0.0, 0.9])

    def test_unknown_does_not_flip_active_state(self):
        memory = SemanticMemory()
        feature = torch.zeros(256)
        memory.observe("ep", "expr", 1, EvidenceLabel.SUPPORTED_POS, 0, feature, 0.0, 1.0, 1.0, "EQ_SELF_PROBE", "q0", "s", 1.0)
        memory.observe("ep", "expr", 1, EvidenceLabel.UNKNOWN, 1, feature, 0.0, None, 0.0, "ALIGNMENT", "q0", "s", 0.0)
        self.assertEqual(memory.active_label("ep", "expr", 1), EvidenceLabel.SUPPORTED_POS)


if __name__ == "__main__":
    unittest.main()
