#!/usr/bin/env python3
"""Create a source-side, label-free calibration report.

This report freezes parser/query budgets and reports coverage only.  It does
not open the evaluation manifest or use target GT to select parameters.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rmot_tta.query.variants import generate_variants  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--runtime-manifest", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    manifest = args.runtime_manifest or Path(config["data"]["runtime_manifest"])
    if not manifest.is_absolute():
        manifest = ROOT / manifest
    expressions: dict[tuple[str, str], str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            expressions[(str(row["video_id"]), str(row["expression_id"]))] = str(row["query_text"])
    reasons = Counter()
    types = Counter()
    variants = Counter()
    records = []
    for (video, expression), text in sorted(expressions.items()):
        spec, query_variants = generate_variants(text)
        reasons[spec.unsupported_reason or "supported"] += 1
        for kind in spec.constraint_types or ("unknown",):
            types[kind] += 1
        for variant in query_variants:
            variants[variant.query_type] += 1
        records.append({
            "video_id": video, "expression_id": expression, "query_text": text,
            "constraint_types": list(spec.constraint_types), "unsupported_reason": spec.unsupported_reason,
            "variants": [variant.__dict__ | {"relation": variant.relation.value} for variant in query_variants],
        })
    payload = {
        "format": "locatemot-qcs-source-calibration-v1", "status": "CALIBRATION_DESCRIPTIVE",
        "config": str(args.config.resolve()), "runtime_manifest": str(manifest.resolve()),
        "evaluation_manifest_opened": False, "gt_used": False,
        "expression_count": len(records), "unsupported_or_supported": dict(reasons),
        "constraint_type_counts": dict(types), "variant_counts": dict(variants),
        "fixed_budget": config.get("probe", {}), "records": records,
        "parameter_selection_from_target_metrics": False,
    }
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "records"} |
                     {"output": str(output.resolve())}, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
