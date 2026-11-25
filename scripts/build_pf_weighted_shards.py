#!/usr/bin/env python3
"""Build complete, GT-free, LPT-balanced logical video shards."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flex_tta.runtime.pf_shards import weighted_manifest, save_manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tracker-root", type=Path, default=Path("/data2/user/TTA-RMOT-assets/flexhook/runs/dance_detr_neuralsort_official/tracker_outputs"))
    p.add_argument("--dataset-root", type=Path, default=Path("/data2/user/TTA-RMOT-assets/flexhook/refer-dance-official/Refer-Dance"))
    p.add_argument("--num-shards", type=int, default=4)
    p.add_argument("--output", type=Path, default=Path("results/pf_four_gpu/manifest.json"))
    args = p.parse_args()
    manifest = weighted_manifest(args.tracker_root, args.dataset_root, num_shards=args.num_shards)
    save_manifest(args.output, manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__": main()
