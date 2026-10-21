#!/usr/bin/env python3
"""Run a configured target suite through the same strict causal runner."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rmot_tta.runtime.runner import OnlineEpisodeRunner  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--runtime-manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--episodes", nargs="*", default=None)
    parser.add_argument("--max-expressions", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()
    config = __import__("yaml").safe_load(args.config.read_text(encoding="utf-8"))
    manifest = args.runtime_manifest or Path(config["data"]["runtime_manifest"])
    if not manifest.is_absolute():
        manifest = ROOT / manifest
    result = OnlineEpisodeRunner(
        args.config, manifest, args.output_dir, methods=args.methods,
        episodes=args.episodes, max_expressions=args.max_expressions, max_frames=args.max_frames,
        shard_index=args.shard_index, num_shards=args.num_shards,
    ).run()
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
