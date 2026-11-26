#!/usr/bin/env python3
"""Audit the configured source checkpoint and runtime/evaluation assets."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/tta/qcs_tta.yaml")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    model = cfg["model"]; data = cfg["data"]
    paths = {name: resolve(model[name]) for name in ("source_root", "checkpoint", "text_root")}
    paths.update({name: resolve(data[name]) for name in ("runtime_manifest", "evaluation_manifest")})
    report = {
        "format": "locatemot-qcs-asset-audit-v1", "cwd": str(Path.cwd().resolve()),
        "user": os.environ.get("USER", "unknown"), "paths": {}, "data": {}, "status": "complete",
    }
    for name, path in paths.items():
        item = {"path": str(path), "exists": path.exists(), "is_file": path.is_file(), "is_dir": path.is_dir()}
        if path.is_file():
            item.update({"bytes": path.stat().st_size, "sha256": sha256(path)})
        if name == "source_root" and path.is_dir():
            try:
                item["git_commit"] = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
            except Exception as exc:
                item["git_commit_error"] = str(exc)
        report["paths"][name] = item
        if not path.exists():
            report["status"] = "incomplete"
    runtime = paths["runtime_manifest"]
    if runtime.is_file():
        rows = [json.loads(line) for line in runtime.read_text(encoding="utf-8").splitlines() if line.strip()]
        forbidden = {"label_path", "expression_path", "gt_box", "gt_id", "label", "target"}
        bad = sorted(forbidden.intersection(*(set(row) for row in rows))) if rows else []
        report["data"]["runtime_rows"] = len(rows)
        report["data"]["runtime_forbidden_fields_found"] = bad
        report["data"]["runtime_image_missing"] = sum(not Path(row["image_path"]).is_file() for row in rows)
        if bad or report["data"]["runtime_image_missing"]:
            report["status"] = "incomplete"
    if args.out is not None:
        out = args.out if args.out.is_absolute() else ROOT / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
