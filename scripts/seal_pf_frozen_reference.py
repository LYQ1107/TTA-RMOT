#!/usr/bin/env python3
"""Seal a complete, actual FP32 Frozen score bank, independently replaying MOT rows."""
import argparse
import gzip
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flex_tta.dance_io import load_tracker_lines
from flex_tta.runtime.pf_receipts import atomic_json, file_sha
from scripts.run_flexhook_positivefree import SOURCE_SHA256


def seal(root, tracker_root, diagnosis):
    root = Path(root).resolve()
    folder = root / "ZERO_UPDATE"
    audit = json.loads((folder / "full_audit.json").read_text())
    metadata = json.loads((folder / "run_metadata.json").read_text())
    probe = json.loads(Path(diagnosis).read_text())
    if audit["status"] != "PASS" or audit["valid_episodes"] != 425 or metadata["sequences"] != 425:
        raise ValueError("all ZERO_UPDATE episodes must be complete and audited")
    if metadata["aggregate_stats"]["optimizer_steps"] != 0:
        raise ValueError("a trainable run cannot define Frozen")
    if len(probe["mismatches"]) != audit["source_sign_mismatches"]:
        raise ValueError("every historical sign difference needs original-vs-cached diagnosis")
    if probe["target_gt_opened"] or any(r["original_cached_sign_mismatches"] or
                                         r["original_cached_max_error"] > 1e-3 for r in probe["mismatches"]):
        raise ValueError("a cached/original decision discrepancy must be fixed, not relabeled")
    index, cached_tracks, selected = {}, {}, 0
    for record in metadata["episodes"]:
        video, expr = record["video"], record["expression_id"]
        episode = folder / "results" / video / expr
        path = episode / "candidate_margins.jsonl.gz"
        if file_sha(path) != record["margins_sha256"]:
            raise ValueError("unsealed source score stream")
        if video not in cached_tracks:
            cached_tracks[video] = load_tracker_lines(tracker_root, video)
        tracks = cached_tracks[video]
        rows = []
        frames = []
        for line in gzip.open(path, "rt"):
            row = json.loads(line)
            t = row["frame"]
            frames.append(t)
            if set(row["ids"]) != set(tracks[t]):
                raise ValueError("Frozen source candidate coverage differs from fixed MOT")
            rows.extend(tracks[t][i] for i, value in sorted(zip(row["ids"], row["adapted"])) if value > 0)
        if frames != sorted(tracks) or (episode / "predict.txt").read_text().splitlines() != rows:
            raise ValueError("independent Frozen score replay does not reproduce ZERO_UPDATE")
        selected += len(rows)
        index[video+"/"+expr] = dict(margins_sha256=record["margins_sha256"],
                                    prediction_sha256=record["prediction_sha256"], selected_rows=len(rows))
    result = dict(status="PASS", protocol="user-approved rebuilt matched Frozen; actual ZERO_UPDATE FP32 margins",
                  checkpoint_sha256=SOURCE_SHA256, episodes=len(index), unique_videos=len(cached_tracks),
                  selected_rows=selected, frozen_replay_prediction_mismatches=0,
                  historical_sign_differences=audit["source_sign_mismatches"],
                  historical_max_abs_margin_error=audit["source_margin_max_abs_error"],
                  original_vs_cached_diagnosis=str(Path(diagnosis).resolve()),
                  diagnosis_sha256=file_sha(diagnosis), episodes_index=index,
                  target_gt_opened=False, optimizer_steps=0, threshold=0.,
                  independence="independent CPU score-to-MOT replay, not a second independent GPU model run",
                  old_frozen_and_supplement_preserved=True)
    atomic_json(folder / "frozen_reference_receipt.json", result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--diagnosis", type=Path, required=True)
    p.add_argument("--tracker-root", type=Path, default=Path("/data2/user/TTA-RMOT-assets/flexhook/runs/dance_detr_neuralsort_official/tracker_outputs"))
    args = p.parse_args()
    result = seal(args.output_root, args.tracker_root, args.diagnosis)
    print(json.dumps({k:v for k,v in result.items() if k != "episodes_index"}, indent=2))


if __name__ == "__main__": main()
