#!/usr/bin/env python3
"""Run the causal LocateMOT QCS-TTA pilot."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rmot_tta.runtime.runner import OnlineEpisodeRunner


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--max-expressions", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--episodes", nargs="*", default=None,
                        help="episode keys such as 0012/cars-in-front-of-the-camera")
    args = parser.parse_args()
    result = OnlineEpisodeRunner(
        args.config,
        args.runtime_manifest,
        args.output_dir,
        methods=args.methods,
        max_expressions=args.max_expressions,
        max_frames=args.max_frames,
        episodes=args.episodes,
    ).run()
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
