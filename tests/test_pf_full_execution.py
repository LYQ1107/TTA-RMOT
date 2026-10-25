"""Focused regressions for scientific coverage and whole-episode resume."""
import gzip
import json

import pytest

from test_pf_runner import fixture_files, runner, preflight
from flex_tta.runtime.pf_shards import weighted_manifest, idle_gpus
from flex_tta.runtime.pf_receipts import EpisodeStore
from flex_tta.dance_io import merge_source_supplement


def test_lpt_whole_video_coverage(tmp_path):
    data, tracks, _ = fixture_files(tmp_path, videos=25)
    manifest = weighted_manifest(tracks, data)
    assigned = [v for s in manifest["shards"] for v in s["videos"]]
    assert len(assigned) == len(set(assigned)) == 25
    assert sum(s["episodes"] for s in manifest["shards"]) == 425
    assert [s["episodes"] for s in manifest["shards"]] == [119, 102, 102, 102]
    assert all(s["max_candidates_per_frame"] == 5 for s in manifest["videos"])


def test_dynamic_gpu_selection_never_shares():
    snapshot = [dict(index=i, free_mib=50000, processes=([{"pid": 99}] if i != 2 else []), utilization=0)
                for i in range(8)]
    assert [g["index"] for g in idle_gpus(snapshot)] == [2]
    assert idle_gpus(snapshot, requested=[0, 1]) == []
    snapshot[2]["utilization"] = 80
    assert idle_gpus(snapshot) == []


def test_atomic_receipt_integrity_and_partial_restart(tmp_path):
    store = EpisodeStore(tmp_path, "v", "e", {"seed": 0})
    with pytest.raises(RuntimeError):
        with store.attempt() as (pred, margins):
            pred.write("unfinished\n")
            raise RuntimeError("simulated interrupted episode")
    assert not store.receipt.exists()
    resumed = EpisodeStore(tmp_path, "v", "e", {"seed": 0}, resume=True)
    with resumed.attempt() as (pred, margins):
        pred.write("complete\n")
        margins.write(json.dumps({"frame": 1})+"\n")
    resumed.finish({"stats": {"frames": 1}})
    assert resumed.completed()["status"] == "complete"
    assert list(resumed.root.glob("failed_attempts/*/predict.txt.partial"))
    assert resumed.prediction.read_text() == "complete\n"
    changed = EpisodeStore(tmp_path, "v", "e", {"seed": 1}, resume=True)
    with pytest.raises(ValueError, match="configuration"):
        changed.completed()
    resumed.prediction.write_text("damaged\n")
    with pytest.raises(ValueError, match="integrity"):
        resumed.completed()


def test_replay_resume_does_not_repeat_or_change_episode(tmp_path):
    data, tracks, source = fixture_files(tmp_path)
    argv = ["--scope", "smoke", "--method", "RELATIVE_ONLY", "--max-frames", "3",
            "--videos", "dancetrack0000", "--dataset-root", str(data), "--tracker-root", str(tracks),
            "--source-result", str(source), "--output-root", str(tmp_path / "out")]
    first = runner.run(runner.parse_args(argv))
    root = tmp_path / "out/RELATIVE_ONLY"
    before = (root / "completed_episodes.jsonl").read_text()
    second = runner.run(runner.parse_args(argv + ["--resume"]))
    assert first["stats"] == second["stats"]
    assert before == (root / "completed_episodes.jsonl").read_text()


def test_full_frame_coverage_rejects_missing_scores(tmp_path):
    args = type("Args", (), {"scope": "full"})()
    with pytest.raises(ValueError, match="every fixed MOT"):
        runner.validate_frames(args, {1: {1: "raw"}}, {1: {}})


def test_source_supplement_cannot_change_existing_frozen_scores():
    source = {"v": {"1": {"1": {"s": [2., -1.]}}}}
    merge_source_supplement(source, {"v": {"1": {"2": {"s": [1., -2.]}}}})
    assert source["v"]["1"]["1"]["s"] == [2., -1.]
    with pytest.raises(ValueError, match="must not replace"):
        merge_source_supplement(source, {"v": {"1": {"1": {"s": [-5., 2.]}}}})


def test_preflight_checks_tracker_observation_not_only_expression_coverage(tmp_path):
    data, tracks, path = fixture_files(tmp_path)
    source = json.loads(path.read_text())
    for frames in source["dancetrack0000"].values():
        for expressions in frames.values():
            logits = next(iter(expressions.values()))
            expressions.update({f"describe person {i}": logits for i in range(17)})
    del source["dancetrack0000"]["1"]["4"]
    path.write_text(json.dumps(source))
    report = preflight.inventory(data, tracks, tmp_path / "refer-kitti-v2-best.pth")
    preflight.verify_server_assets(report, data, tracks, None, None, path, None)
    assert report["frozen_scores"]["coverage"]["dancetrack0000"]["missing_tracker_observations"] == 1
    assert not report["frozen_scores"]["all_tracker_candidates_scored"]
