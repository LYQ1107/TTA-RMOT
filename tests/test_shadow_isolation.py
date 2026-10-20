import unittest

import torch

from rmot_tta.probes.shadow_state import ShadowRuntime


class FakeWrapper:
    def __init__(self):
        self.calls = []

    def reset_runtime(self):
        self.calls.append("reset")

    def infer_frame(self, **kwargs):
        self.calls.append((kwargs["frame_index"], kwargs["track_instances"], kwargs["sentence"]))
        return type("Result", (), {"track_instances": object(), "snapshot": None})()


class ShadowIsolationTest(unittest.TestCase):
    def test_shadow_state_is_private_and_resettable(self):
        left = FakeWrapper()
        right = FakeWrapper()
        a = ShadowRuntime(left, "q0")
        b = ShadowRuntime(right, "q_cf")
        a.step(torch.zeros(3, 2, 2), "q0", (2, 2), 0)
        b.step(torch.zeros(3, 2, 2), "qcf", (2, 2), 0)
        self.assertIsNot(a.state.track_instances, b.state.track_instances)
        a.reset()
        self.assertIsNone(a.state.track_instances)
        self.assertIn("qcf", right.calls[-1][2])


if __name__ == "__main__":
    unittest.main()
