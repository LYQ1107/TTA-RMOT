#!/usr/bin/env python3
"""Build separate label-free runtime and evaluator manifests."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _label_format(data: dict, video: str) -> str:
    configured = data.get("label_format_by_video_prefix", data.get("label_format", "cxcywh"))
    if isinstance(configured, dict):
        for prefix, value in sorted(configured.items(), key=lambda item: len(str(item[0])), reverse=True):
            if video.startswith(str(prefix)):
                return str(value)
        return "cxcywh"
    return str(configured)


def _query_text(expression_path: Path, payload: dict, source: str) -> str:
    if source in {"filename", "stem", "expression_id"}:
        sentence = expression_path.stem.strip()
    else:
        sentence = str(payload.get(source, "")).strip()
    if not sentence:
        raise ValueError(f"empty query text ({source}): {expression_path}")
    return sentence


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/tta/qcs_tta.yaml")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--videos", nargs="*", default=None)
    parser.add_argument("--expressions", nargs="*", default=None)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results/manifests")
    args = parser.parse_args()
    cfg = load_config(args.config.resolve())
    data = cfg["data"]
    dataset = str(args.dataset or data["dataset"])
    data_root = (ROOT / data["data_root"]).resolve() if not Path(data["data_root"]).is_absolute() else Path(data["data_root"]).resolve()

    def configured_path(value: str) -> Path:
        path = Path(value)
        return path.resolve() if path.is_absolute() else (ROOT / path).resolve()

    image_root = configured_path(data["image_root"])
    expression_root = configured_path(data["expression_root"])
    label_root = configured_path(data["label_root"])
    videos = [str(value) for value in (args.videos or data.get("videos") or [])]
    if not videos:
        videos = sorted(path.name for path in expression_root.iterdir() if path.is_dir())
    query_text_source = str(data.get("query_text_source", "sentence"))
    trackeval_frame_offset = int(data.get("trackeval_frame_offset", 1))
    expression_filter = {str(value) for value in args.expressions} if args.expressions else None
    runtime_rows: list[dict] = []
    evaluation_rows: list[dict] = []
    for video in videos:
        image_dir = image_root / video
        expr_dir = expression_root / video
        if not image_dir.is_dir():
            raise FileNotFoundError(f"runtime image directory missing: {image_dir}")
        if not expr_dir.is_dir():
            raise FileNotFoundError(f"expression directory missing: {expr_dir}")
        images = sorted(
            (path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES),
            key=lambda path: (int(path.stem), path.name),
        )
        if not images:
            raise FileNotFoundError(f"no images for video {video}: {image_dir}")
        expressions = sorted(
            path for path in expr_dir.glob("*.json")
            if expression_filter is None or path.stem in expression_filter
        )
        if not expressions:
            raise FileNotFoundError(f"no expressions for video {video}: {expr_dir}")
        for expression_path in expressions:
            payload = json.loads(expression_path.read_text(encoding="utf-8"))
            sentence = _query_text(expression_path, payload, query_text_source)
            expression_id = expression_path.stem
            label_dir = label_root / video
            for image_path in images:
                frame_index = int(image_path.stem)
                runtime_rows.append({
                    "dataset": dataset, "video_id": video, "expression_id": expression_id,
                    "query_text": sentence, "frame_index": frame_index,
                    "trackeval_frame_offset": trackeval_frame_offset,
                    "image_path": str(image_path.resolve()),
                })
                evaluation_rows.append({
                    "dataset": dataset, "video_id": video, "expression_id": expression_id,
                    "frame_index": frame_index, "image_path": str(image_path.resolve()),
                    "expression_path": str(expression_path.resolve()),
                    "label_path": str((label_dir / f"{image_path.stem}.txt").resolve()),
                    "label_format": _label_format(data, video),
                    "trackeval_frame_offset": trackeval_frame_offset,
                })
    runtime_rows.sort(key=lambda row: (row["video_id"], row["expression_id"], row["frame_index"]))
    evaluation_rows.sort(key=lambda row: (row["video_id"], row["expression_id"], row["frame_index"]))
    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    runtime_path = out_dir / f"{dataset}_runtime_manifest.jsonl"
    evaluation_path = out_dir / f"{dataset}_evaluation_manifest.jsonl"
    for path in (runtime_path, evaluation_path):
        if path.exists():
            path.unlink()
    runtime_path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in runtime_rows), encoding="utf-8")
    evaluation_path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in evaluation_rows), encoding="utf-8")
    metadata = {
        "format": "locatemot-qcs-manifests-v1", "dataset": dataset, "videos": videos,
        "runtime_rows": len(runtime_rows), "evaluation_rows": len(evaluation_rows),
        "runtime_manifest": str(runtime_path.resolve()), "evaluation_manifest": str(evaluation_path.resolve()),
        "runtime_contains_gt": False, "runtime_forbidden_field_names": ["label_path", "expression_path", "gt_box", "gt_id", "label"],
        "runtime_forbidden_fields_found": [],
        "evaluation_contains_gt_paths": True,
        "query_text_source": query_text_source,
        "trackeval_frame_offset": trackeval_frame_offset,
        "label_formats": {video: _label_format(data, video) for video in videos},
        "expression_filter": sorted(expression_filter) if expression_filter is not None else None,
        "source_files": {"config": str(args.config.resolve()), "config_sha256": sha256_file(args.config.resolve())},
    }
    (out_dir / f"{dataset}_manifest_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
