"""A matched Frozen reference from immutable, full ZERO_UPDATE score streams.

No pseudo labels or new logits are manufactured. The recorded real FP32
margins are reused at their own chronological frame by every subsequent method.
Historical logits remain immutable and separately auditable.
"""
from functools import lru_cache
import gzip
import json
from pathlib import Path

from flex_tta.runtime.pf_receipts import file_sha


@lru_cache(maxsize=4)
def reference_receipt(root):
    path = Path(root) / "ZERO_UPDATE/frozen_reference_receipt.json"
    receipt = json.loads(path.read_text())
    if receipt["status"] != "PASS" or receipt["episodes"] != 425 or receipt["unique_videos"] != 25:
        raise ValueError("matched Frozen reference must cover all 25 videos / 425 episodes")
    return receipt


def source_frames(root, video, expression):
    receipt = reference_receipt(str(Path(root).resolve()))
    record = receipt["episodes_index"][video+"/"+expression]
    path = Path(root) / "ZERO_UPDATE/results" / video / expression / "candidate_margins.jsonl.gz"
    if file_sha(path) != record["margins_sha256"]:
        raise ValueError("matched Frozen margin stream changed after sealing")
    frames = {}
    with gzip.open(path, "rt") as stream:
        for line in stream:
            row = json.loads(line)
            frames[row["frame"]] = dict(zip(row["ids"], row["adapted"]))
    return frames
