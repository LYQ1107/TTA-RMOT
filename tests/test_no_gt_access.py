import json
import tempfile
import unittest
from pathlib import Path

from rmot_tta.runtime.runner import OnlineEpisodeRunner


class NoGtAccessTest(unittest.TestCase):
    def test_runtime_manifest_contract_rejects_gt_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.jsonl"
            path.write_text(json.dumps({"video_id": "v", "expression_id": "e", "frame_index": 0,
                                       "query_text": "cars", "image_path": "/does/not/exist", "gt_id": 1}) + "\n")
            runner = OnlineEpisodeRunner(Path("configs/tta/qcs_tta.yaml"), path, Path(directory) / "out")
            with self.assertRaises(AssertionError):
                runner._load_rows()

    def test_runner_source_has_no_evaluation_manifest_access(self):
        source = Path("rmot_tta/runtime/runner.py").read_text(encoding="utf-8")
        self.assertIn("evaluation_manifest_opened", source)
        self.assertIn('forbidden = {"label_path", "expression_path"', source)
        self.assertNotIn("TrackEval", source)


if __name__ == "__main__":
    unittest.main()
