import unittest

from pathlib import Path


class ResumeSafetyTest(unittest.TestCase):
    def test_existing_prediction_directory_is_not_appended(self):
        source = Path("rmot_tta/runtime/runner.py").read_text(encoding="utf-8")
        self.assertIn("prediction files already exist", source)
        self.assertIn("fresh run directory", source)


if __name__ == "__main__":
    unittest.main()
