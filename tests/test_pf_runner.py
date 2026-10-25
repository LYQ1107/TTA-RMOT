import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_script(relative_path):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# The patch is tested standalone before being applied to the existing repo.
# When the original project's flex_tta.dance_io is not available locally,
# provide a minimal compatible fixture shim, without placing it in the patch.
try:
    import flex_tta.dance_io  # noqa: F401
except ModuleNotFoundError:
    import types
    shim = types.ModuleType("flex_tta.dance_io")
    def dance_videos(tracker_root):
        return sorted(x.stem for x in Path(tracker_root).glob("*.txt"))
    def load_expressions(dataset_root, video):
        return [(f.stem, json.loads(f.read_text())["sentence"])
                for f in sorted((Path(dataset_root)/"expression"/video).glob("*.json"))]
    def load_tracker_lines(tracker_root, video):
        d = {}
        for line in (Path(tracker_root)/(video+".txt")).read_text().splitlines():
            v = line.split(",")
            d.setdefault(int(v[0]), {})[int(v[1])] = line
        return d
    def score_frames(source, video, sentence):
        d = {}
        for track, frames in source[video].items():
            for frame, expressions in frames.items():
                a, b = expressions[sentence]
                d.setdefault(int(frame), {})[int(track)] = b - a
        return dict(sorted(d.items()))
    shim.dance_videos = dance_videos
    shim.load_expressions = load_expressions
    shim.load_tracker_lines = load_tracker_lines
    shim.score_frames = score_frames
    sys.modules["flex_tta.dance_io"] = shim

runner = load_script("scripts/run_flexhook_positivefree.py")
preflight = load_script("scripts/pf_preflight.py")


def fixture_files(tmp_path, *, videos=1):
    data = tmp_path / "dance"
    track = tmp_path / "tracker"
    track.mkdir()
    source = {}
    for index in range(videos):
        name = f"dancetrack{index:04d}"
        expdir = data / "expression" / name
        images = data / "DanceTrack" / "training" / "image_02" / name
        expdir.mkdir(parents=True)
        images.mkdir(parents=True)
        (images / "000001.png").write_bytes(b"fake image only for inventory")
        sentences = [f"describe person {i}" for i in range(17)]
        for i, s in enumerate(sentences):
            (expdir / f"expression{i:02d}.json").write_text(json.dumps({"sentence": s}))
        tr = []
        source[name] = {}
        for tid in range(1, 6):
            scores = {str(f): {sentences[0]: [0.0, {1:-4.0,2:-4.1,3:-11.0,4:-11.2,5:-11.3}[tid]]} for f in range(1,5)}
            source[name][str(tid)] = scores
            for f in range(1,5):
                tr.append(f"{f},{tid},1,2,30,40,1,-1,-1,-1")
        (track / (name + ".txt")).write_text("\n".join(tr) + "\n")
    src = tmp_path / "source.json"
    src.write_text(json.dumps(source))
    return data, track, src


def test_default_plan_does_not_access_assets(tmp_path):
    args = runner.parse_args(["--output-root", str(tmp_path / "new")])
    result = runner.run(args)
    assert result["scope"] == "plan"
    assert not (tmp_path / "new").exists()


def test_full_study_requires_user_authorization(monkeypatch, tmp_path):
    monkeypatch.delenv("FLEXHOOK_FULL_RUN_AUTHORIZED", raising=False)
    with pytest.raises(SystemExit):
        runner.parse_args(["--scope", "full", "--max-expressions", "0", "--max-frames", "0",
                           "--output-root", str(tmp_path / "full")])


def test_smoke_replay_preserves_tracker_and_no_gt(tmp_path):
    data, track, src = fixture_files(tmp_path)
    args = runner.parse_args([
        "--scope", "smoke", "--method", "RELATIVE_ONLY", "--max-frames", "3",
        "--videos", "dancetrack0000",
        "--dataset-root", str(data), "--tracker-root", str(track),
        "--source-result", str(src), "--output-root", str(tmp_path / "out"),
    ])
    receipt = runner.run(args)
    assert receipt["stats"]["episodes"] == 1
    assert receipt["sequences"] == 1
    assert receipt["episodes"][0]["video"] == "dancetrack0000"
    assert receipt["episodes"][0]["stats"]["frames"] == 3
    assert receipt["episodes"][0]["stats"]["selected_rows"] == 2
    assert receipt["stats"]["optimizer_steps"] == 0
    predicted = list((tmp_path / "out" / "RELATIVE_ONLY" / "results").rglob("predict.txt"))
    assert len(predicted) == 1
    lines = predicted[0].read_text().strip().splitlines()
    assert all(line in (track / "dancetrack0000.txt").read_text().splitlines() for line in lines)
    assert not list((tmp_path / "out").rglob("gt.txt"))
    assert receipt["stats"]["selected"] == 2  # two stable candidate tails at frame 3


def test_no_target_gt_read_in_preflight(tmp_path):
    data, track, src = fixture_files(tmp_path, videos=1)
    (data / "gt_template").mkdir()
    (data / "gt_template" / "secret.txt").write_text("NEVER OPEN THIS")
    report = preflight.inventory(data, track, tmp_path / "refer-kitti-v2-best.pth")
    assert report["target_gt_opened"] is False
    assert report["target_video_count"] == 1
    assert report["target_expression_count"] == 17
    assert report["target_inventory_pass"] is False


def test_four_gpu_script_stays_dry_by_default(tmp_path):
    data, track, _ = fixture_files(tmp_path, videos=25)
    env = os.environ.copy()
    env["PF_TRACKER_ROOT"] = str(track)
    proc = subprocess.run(["bash", str(ROOT / "scripts/run_pf_4gpu.sh")],
                          capture_output=True, text=True, env=env, check=False)
    assert proc.returncode == 0, proc.stderr
    assert "DRY_RUN_ONLY" in proc.stdout
    assert "LAUNCH GPU" not in proc.stdout
    assert "GPU" in proc.stdout


def test_complete_25_425_inventory(tmp_path):
    data,track,src=fixture_files(tmp_path,videos=25)
    report=preflight.inventory(data,track,tmp_path/"refer-kitti-v2-best.pth")
    assert report["target_video_count"]==25
    assert report["target_expression_count"]==425
    assert report["target_inventory_pass"]


def test_four_gpu_full_launch_cannot_start_without_consent(tmp_path):
    _,track,_=fixture_files(tmp_path,videos=25)
    env=os.environ.copy()
    env["PF_TRACKER_ROOT"]=str(track)
    env.pop("PF_LAUNCH_FULL",None)
    env.pop("FLEXHOOK_FULL_RUN_AUTHORIZED",None)
    proc=subprocess.run(["bash",str(ROOT/"scripts/run_pf_4gpu.sh"),"--launch-full"],
                        capture_output=True,text=True,env=env,check=False)
    assert proc.returncode==3
    assert "BLOCKED" in proc.stderr


def test_relative_output_survives_vendor_cwd_change(tmp_path, monkeypatch):
    data, track, src = fixture_files(tmp_path)
    monkeypatch.chdir(tmp_path)
    args = runner.parse_args(["--scope", "smoke", "--method", "RELATIVE_ONLY",
        "--videos", "dancetrack0000", "--dataset-root", str(data), "--tracker-root", str(track),
        "--source-result", str(src), "--output-root", "relative-output"])
    replay = runner._episodic_replay
    def change_cwd(*values):
        monkeypatch.chdir(data)
        return replay(*values)
    monkeypatch.setattr(runner, "_episodic_replay", change_cwd)
    receipt = runner.run(args)
    assert Path(receipt["output_root"]) == tmp_path / "relative-output"
    assert (tmp_path / "relative-output/RELATIVE_ONLY/run_metadata.json").is_file()


def test_pf_metadata_merges_with_existing_shard_tool(tmp_path):
    # Synthetic empty predictions only; no real data, GPU, GT or evaluation.
    for shard in range(4):
        method = tmp_path / "shards" / f"shard{shard}" / "PF_RANK"
        method.mkdir(parents=True)
        episodes = []
        for index in range(shard, 25, 4):
            video = f"dancetrack{index:04d}"
            for exp in range(17):
                expr_id = f"e{exp}"
                output = method / "results" / video / expr_id / "predict.txt"
                output.parent.mkdir(parents=True)
                output.touch()
                episodes.append({"video": video, "expression_id": expr_id,
                    "stats": {"frames": 12, "selected_rows": 0}})
        (method / "run_metadata.json").write_text(json.dumps({"episodes": episodes}))
    proc = subprocess.run([sys.executable, str(ROOT / "scripts/merge_flexhook_pcd_shards.py"),
        "--shards-root", str(tmp_path / "shards"), "--output-root", str(tmp_path / "merged"),
        "--method", "PF_RANK"], text=True, capture_output=True)
    assert proc.returncode == 0, proc.stderr
    result = json.loads((tmp_path / "merged/PF_RANK/run_metadata.json").read_text())
    assert result["sequences"] == 425 and len(result["videos"]) == 25
    assert result["aggregate_stats"]["frames"] == 425 * 12
