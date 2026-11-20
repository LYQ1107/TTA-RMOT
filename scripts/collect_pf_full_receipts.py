#!/usr/bin/env python3
"""Compact recovery index for the six already-audited full runs; no raw GT."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json
from scripts.summarize_pf_full import METHOD_ORDER


def collect(publish):
    publish = Path(publish)
    methods = {}
    for method in METHOD_ORDER:
        folder = publish/method.lower()
        metrics = json.loads((folder/"metrics.json").read_text())
        audit = metrics["full_audit"]
        if (audit["status"] != "PASS" or audit["valid_episodes"] != 425 or
                audit["unique_videos"] != 25 or audit["candidates"] != 3588445 or
                audit["frames"] != 433636 or audit["missing_episodes"] or audit["duplicated_episodes"]):
            raise ValueError(f"incomplete audited run: {method}")
        if any(audit[k] for k in ("target_gt_opened", "tracker_geometry_changed", "tracker_ids_changed")):
            raise ValueError(f"protocol violation in audit: {method}")
        evaluation = json.loads((folder/"evaluation_metadata.json").read_text())
        if len(evaluation) != 1 or evaluation[0]["trackeval_exit_status"] != 0 or evaluation[0]["attached_gt_symlinks"] != 425:
            raise ValueError(f"incomplete official TrackEval: {method}")
        raw = Path(metrics["raw_output_root"])/method
        metadata = json.loads((raw/"run_metadata.json").read_text())
        if metadata["status"] != "complete" or metadata["sequences"] != 425:
            raise ValueError(f"unsealed method metadata: {method}")
        records = []
        for episode in metadata["episodes"]:
            video, expression = episode["video"], episode["expression_id"]
            path = raw/"results"/video/expression/"completion.json"
            receipt = json.loads(path.read_text())
            if receipt["status"] != "complete" or not receipt["full_frame_coverage"]:
                raise ValueError(f"incomplete episode receipt: {path}")
            records.append(dict(video=video, expression=expression,
                                completion=str(path.relative_to(raw)), prediction_sha256=receipt["prediction_sha256"],
                                margins_sha256=receipt["margins_sha256"],
                                wall_seconds=receipt["wall_seconds"], statistics=receipt["stats"]))
        if len(records) != 425 or len({(e["video"],e["expression"]) for e in records}) != 425:
            raise ValueError(f"duplicate or missing recovery receipts: {method}")
        runtime = []
        for shard in metadata.get("shards", []):
            stats = shard["stats"]
            item = dict(experiment_commit=shard["experiment_commit"], gpu_uuid=shard["gpu_uuid"],
                        sequences=shard["sequences"], hyperparameters=shard["hyperparameters"],
                        checkpoint_strict_load=stats.get("checkpoint_strict_load"),
                        trainable_parameters=stats.get("trainable_parameters"),
                        pcd_channels=stats.get("pcd_channels"), feature_channels=stats.get("feature_channels"),
                        frozen_parameters_have_grad=stats.get("frozen_parameters_have_grad"),
                        rollback_restored=stats.get("rollback_restored"),
                        pre_episode_reset_checks=stats.get("episode_resets",[]),
                        post_episode_reset_smoke_check=stats.get("post_episode_reset_pass"))
            if method in {"ZERO_UPDATE","PF_RANK","PF_IDENTITY","PF_HYBRID"}:
                resets = item["pre_episode_reset_checks"]
                if (not item["checkpoint_strict_load"] or item["frozen_parameters_have_grad"] is not False or
                        len(resets) != shard["sequences"] or not all(r["passed"] for r in resets)):
                    raise ValueError(f"failed actual model/reset checks: {method}")
                if any(stats.get(k,0) for k in ("nonfinite_gradient_steps","zero_gradient_steps")):
                    raise ValueError(f"failed gradient checks: {method}")
                if method != "ZERO_UPDATE" and stats.get("sign_mismatches_before_updates",0):
                    raise ValueError(f"initial source mismatch: {method}")
            runtime.append(item)
        methods[method] = dict(status="COMPLETE_25_VIDEOS_425_EPISODES", raw_method_root=str(raw),
                               official_evaluation=evaluation, full_audit=audit,
                               reused_from=metadata.get("derived_from"), model_shards=runtime, episodes=records)
    result = dict(status="COMPLETE_6_METHODS", methods=methods,
                  scope="index of existing full-audit validated completion receipts; no new GPU inference or raw GT access",
                  hash_validation="prediction/margin hashes were checked by each full_audit; this index preserves those receipts",
                  missing_post_episode_reset_flag="not emitted by full runs; null is not a passed smoke check",
                  protocol="fixed V2 / official Dance MOT / rank4 / lr5e-5 / interval4 / seed0 / FP32",
                  formal_runs_stopped_after_six=True)
    destination = publish/"manifests/full_recovery_index.json"
    atomic_json(destination,result)
    print(json.dumps(dict(status=result["status"], methods=len(methods),
                          unique_episode_receipts=sum(len(m["episodes"]) for m in methods.values()), output=str(destination))))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--publish-root",type=Path,default=ROOT/"results/positive_free/full")
    args=p.parse_args()
    collect(args.publish_root)


if __name__ == "__main__": main()
