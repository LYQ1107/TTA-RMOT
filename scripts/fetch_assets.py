#!/usr/bin/env python3
"""Fetch optional official assets without inheriting proxy variables.

The default command only reports missing assets.  Pass ``--download-v2`` to
fetch the official V2 release asset with a resumable, checksum-verified write.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V2_URL = "https://github.com/zyn213/TempRMOT/releases/download/v1.0/checkpoint_rk2.pth"
V2_EXPECTED_BYTES = 1096316824


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download(url: str, destination: Path) -> dict:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    existing = partial.stat().st_size if partial.exists() else 0
    request = urllib.request.Request(url, headers={"User-Agent": "LocateMOT/QCS-TTA"})
    if existing:
        request.add_header("Range", f"bytes={existing}-")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=60) as response:
        mode = "ab" if existing and response.status == 206 else "wb"
        if mode == "wb":
            existing = 0
        with partial.open(mode) as handle:
            shutil.copyfileobj(response, handle, length=1 << 20)
    partial.replace(destination)
    size = destination.stat().st_size
    if url == V2_URL and size != V2_EXPECTED_BYTES:
        raise RuntimeError(f"unexpected V2 checkpoint size: {size} != {V2_EXPECTED_BYTES}")
    return {"url": url, "path": str(destination.resolve()), "bytes": size, "sha256": digest(destination)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download-v2", action="store_true")
    parser.add_argument("--destination", type=Path, default=ROOT / "data/checkpoints/checkpoint_rk2.pth")
    args = parser.parse_args()
    report = {"format": "locatemot-qcs-fetch-assets-v1", "proxy_disabled_for_download": True, "assets": []}
    if args.download_v2:
        destination = args.destination if args.destination.is_absolute() else ROOT / args.destination
        report["assets"].append(download(V2_URL, destination))
    else:
        report["next_action"] = "run with --download-v2 only after verifying the official release asset is desired"
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
