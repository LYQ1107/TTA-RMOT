#!/usr/bin/env python3
"""Materialize the official FlexHook Dance DETR+NeuralSORT tracker bank.

The public FlexHook repository releases the binary tracker bank through its
official Baidu share.  A fresh share listing contains signed ``picdocpreview``
URLs for the text files; this script consumes that listing and verifies the
25 DanceTrack validation files without reading Refer-Dance labels or
expressions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import requests


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--listing-json", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    listing = json.loads(args.listing_json.read_text(encoding="utf-8"))
    files = [
        item
        for item in listing.get("list", [])
        if item.get("isdir") == 0
        and item.get("server_filename", "").startswith("dancetrack")
        and item.get("server_filename", "").endswith(".txt")
    ]
    if len(files) != 25:
        raise RuntimeError(f"expected 25 DanceTrack tracker files, got {len(files)}")

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})
    receipt_files = []
    for item in sorted(files, key=lambda value: value["server_filename"]):
        name = item["server_filename"]
        response = session.get(item["picdocpreview"], timeout=90)
        response.raise_for_status()
        data = response.content
        if len(data) != int(item["size"]):
            raise RuntimeError(f"{name}: byte count does not match the official listing")
        lines = data.decode("utf-8").splitlines()
        if not data.endswith(b"\n") or not lines or any(len(line.split(",")) != 10 for line in lines):
            raise RuntimeError(f"{name}: invalid 10-column MOT tracker format")
        target = output_root / name
        temporary = target.with_suffix(target.suffix + ".part")
        temporary.write_bytes(data)
        temporary.replace(target)
        receipt_files.append(
            {
                "file": name,
                "source_path": item.get("path", ""),
                "fs_id": str(item.get("fs_id", "")),
                "bytes": len(data),
                "rows": len(lines),
                "sha256": hashlib.sha256(data).hexdigest(),
                "source_md5": item.get("md5", ""),
            }
        )

    receipt = {
        "source": "official FlexHook Baidu share",
        "share_path": "/FlexHook/tracker_outputs/DETR-NeuralSORT-Dance",
        "tracker": "DETR detector + NeuralSORT tracker (official released outputs)",
        "download_method": "share/list picdocpreview signed text endpoints",
        "ground_truth_read": False,
        "expressions_read": False,
        "file_count": len(receipt_files),
        "total_bytes": sum(item["bytes"] for item in receipt_files),
        "files": receipt_files,
    }
    receipt_path = output_root.parent / "official_detr_neuralsort_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_root": str(output_root), "receipt": str(receipt_path), "file_count": len(receipt_files)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
