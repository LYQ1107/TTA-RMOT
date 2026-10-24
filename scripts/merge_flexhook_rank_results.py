#!/usr/bin/env python3
"""Merge FlexHook validation shards without starting a DDP process group.

The released evaluator uses a DistributedSampler.  When the number of test
items is not divisible by the world size, PyTorch pads the sampler by
repeating the first item.  Therefore a small number of prediction leaves can
appear in both shards; the first shard is kept for those leaves.
"""

import argparse
import json
import os
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def merge_dicts(destination, source, stats):
    for key, value in source.items():
        if key not in destination:
            destination[key] = value
        elif isinstance(destination[key], dict) and isinstance(value, dict):
            merge_dicts(destination[key], value, stats)
        else:
            # Duplicate leaves are expected only from DistributedSampler
            # padding.  Keep the earlier shard so recovery batch-size changes
            # cannot replace the original rank-0 value.
            stats["duplicate_leaves"] += 1


def main():
    args = parse_args()
    if len(args.shards) < 2:
        raise ValueError("at least two shard files are required")

    shards = []
    for shard_path in args.shards:
        with open(shard_path) as handle:
            shard = json.load(handle)
        if not isinstance(shard, dict):
            raise TypeError(f"expected a dict at {shard_path}")
        shards.append(shard)

    merged = shards[0]
    stats = {"duplicate_leaves": 0}
    for shard in shards[1:]:
        merge_dicts(merged, shard, stats)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    with temporary.open("w") as handle:
        json.dump(merged, handle, indent=1)
    os.replace(temporary, output)

    # Re-open the final artifact so a successful command also proves that the
    # on-disk JSON is complete and parseable.
    with output.open() as handle:
        merged_check = json.load(handle)
    print(
        f"merged={output} videos={len(merged_check)} "
        f"duplicate_leaves_kept_from_first_shard={stats['duplicate_leaves']} "
        f"bytes={output.stat().st_size}"
    )


if __name__ == "__main__":
    main()
