# QCS-TTA result receipts

This directory contains compact, reproducible receipts and selected sealed
runs. Raw AerialMind images, labels, checkpoints, and the complete 30-video
manifests are intentionally kept on `/data2` and are not committed to GitHub.

Local full-data paths:

- dataset: `/data2/user/LocateMOT/data/AerialMind`
- runtime manifest: `/data2/user/LocateMOT/results/manifests/aerialmind_test_runtime_manifest.jsonl`
- evaluation manifest: `/data2/user/LocateMOT/results/manifests/aerialmind_test_evaluation_manifest.jsonl`

The two complete AerialMind episode receipts committed here use 1,226 frame
queries and are under `runs/aerialmind_two_episode_*`. Their runtime metadata
records `gt_used_by_runner=false`, `future_frames_used=false`, and
`evaluation_manifest_opened=false`; TrackEval files are generated only under
the sealed evaluation subdirectories.
