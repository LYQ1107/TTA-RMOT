import unittest

from pathlib import Path


class CausalityContractTest(unittest.TestCase):
    def test_runner_orders_prediction_before_observe_and_update(self):
        source = Path("rmot_tta/runtime/runner.py").read_text(encoding="utf-8")
        prediction = source.index("prediction = self._prediction")
        observe = source.index("controller.observe", prediction)
        adapt = source.index("controller.maybe_adapt", observe)
        self.assertLess(prediction, observe)
        self.assertLess(observe, adapt)
        self.assertIn('"future_frames_used": False', source)


if __name__ == "__main__":
    unittest.main()
