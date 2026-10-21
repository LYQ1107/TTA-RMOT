import unittest

import torch

from rmot_tta.adaptation.independent import (
    CacheTDARetriever,
    IndependentAdaptationController,
)
from rmot_tta.evidence.records import CandidateEvidence, EvidenceLabel
from rmot_tta.types import FrameSnapshot


def _snapshot(frame_index=0, with_query=True):
    return FrameSnapshot(
        episode_id="0005/e0", video_id="0005", expression_id="e0",
        query_text="the car", frame_index=frame_index,
        candidate_uid=("u0", "u1", "u2"),
        features=torch.randn(3, 256),
        source_logits=torch.tensor([2.0, -2.0, 0.2]),
        source_probs=torch.tensor([0.8808, 0.1192, 0.5498]),
        boxes_xyxy=torch.tensor([[0., 0., 20., 20.], [5., 5., 30., 30.], [1., 2., 12., 14.]]),
        objectness=torch.tensor([.9, .8, .7]),
        pred_track_id=torch.tensor([10, 11, 12]),
        disappear_time=torch.tensor([0, 1, 0]), image_size=(100, 100),
        valid_mask=torch.tensor([True, True, True]),
        query_embedding=torch.randn(256) if with_query else None,
    )


EVIDENCE = [
    CandidateEvidence("u0", EvidenceLabel.SUPPORTED_POS, 1.0, 1.0, "synthetic", "q0", 1.0, "test", 0, 10),
    CandidateEvidence("u1", EvidenceLabel.SUPPORTED_NEG, 0.0, 1.0, "synthetic", "q0", 1.0, "test", 0, 11),
]


class IndependentTtaTest(unittest.TestCase):
    def test_each_adapter_starts_source_equivalent_then_updates(self):
        methods = (
            "VISUAL_TTA", "LANGUAGE_TTA", "FUSION_TTA",
            "VISUAL_LANGUAGE_TTA", "VISUAL_FUSION_TTA",
            "LANGUAGE_FUSION_TTA", "ALL_TTA",
        )
        for method in methods:
            with self.subTest(method=method):
                controller = IndependentAdaptationController(
                    method, lr=1e-2, update_every_n_frames=1,
                    batch_size=8, minimum_distinct_track_time_samples=2,
                )
                snapshot = _snapshot()
                before = controller.predict_logits(snapshot).detach().clone()
                controller.observe(snapshot, EVIDENCE)
                update = controller.maybe_adapt(snapshot, EVIDENCE)
                after = controller.predict_logits(snapshot).detach().clone()
                self.assertTrue(update.stepped)
                self.assertEqual(controller.nonzero_gradient_steps, 1)
                self.assertGreater(controller.parameter_delta_sum, 0.0)
                self.assertTrue(torch.equal(before, snapshot.source_logits))
                self.assertFalse(torch.equal(after, before))

    def test_tda_is_nonparametric_and_uses_cache(self):
        snapshot = _snapshot(with_query=False)
        controller = CacheTDARetriever(capacity=4, similarity_threshold=.5)
        controller.observe(snapshot, EVIDENCE)
        update = controller.maybe_adapt(snapshot, EVIDENCE)
        adapted = controller.predict_logits(snapshot)
        self.assertFalse(update.stepped)
        self.assertEqual(controller.cache_updates, 2)
        self.assertEqual(controller.cache_hits, 2)
        self.assertFalse(torch.equal(adapted, snapshot.source_logits))


if __name__ == "__main__":
    unittest.main()
