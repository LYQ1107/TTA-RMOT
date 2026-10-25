#!/usr/bin/env python3
"""Map actual local experiment commits to identical published GitHub trees."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json


def git(*args, cwd=ROOT):
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def commits(metadata):
    values = {s["experiment_commit"] for s in metadata.get("shards", []) if s.get("experiment_commit")}
    values.update(metadata[k] for k in ("experiment_commit", "derivation_experiment_commit") if metadata.get(k))
    if metadata.get("derived_from"):
        values.update(commits(json.loads((Path(metadata["derived_from"])/"PF_RANK/run_metadata.json").read_text())))
    return values


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--public-ref", default="FETCH_HEAD")
    p.add_argument("--publish-root", type=Path, default=ROOT/"results/positive_free/full")
    p.add_argument("--require-six", action="store_true")
    args = p.parse_args()
    public_tip = git("rev-parse", args.public_ref)
    public_trees = {}
    for line in git("log", public_tip, "--format=%H %T").splitlines():
        commit, tree = line.split()
        public_trees.setdefault(tree, commit)
    method_sources = {}
    for path in sorted(args.publish_root.glob("*/metrics.json")):
        metric = json.loads(path.read_text())
        method = metric["method"]
        meta = json.loads((Path(metric["raw_output_root"])/method/"run_metadata.json").read_text())
        method_sources[method] = sorted(commits(meta))
    if args.require_six and len(method_sources) != 6:
        raise ValueError("final provenance requires six completed methods")
    mappings = []
    for local in sorted({c for values in method_sources.values() for c in values}):
        tree = git("rev-parse", local+"^{tree}")
        if tree not in public_trees:
            raise ValueError(f"experiment tree is not published under fetched ref: {local}")
        public = public_trees[tree]
        mappings.append(dict(local_experiment_commit=local, github_commit=public, tree_sha=tree,
                             identical_tree=True, url="https://github.com/LYQ1107/TTA-RMOT/commit/"+public))
    vendor = ROOT/"vendor/FlexHook"
    patch = subprocess.check_output(["git", "diff", "--binary"], cwd=vendor)
    patch_path = args.publish_root/"manifests/vendor_tracked_source.patch"
    patch_path.parent.mkdir(parents=True, exist_ok=True)
    patch_path.write_bytes(patch)
    result = dict(status="COMPLETE_6_METHODS" if len(method_sources) == 6 else "INTERIM",
                  public_ref=args.public_ref, public_tip=public_tip, method_sources=method_sources,
                  publication_mappings=mappings, git_https_push_available=False,
                  publication_route="GitHub Git-object API; tree equality verified before leased non-force ref update",
                  vendor_commit=git("rev-parse", "HEAD", cwd=vendor),
                  vendor_tracked_patch=str(patch_path.relative_to(ROOT)),
                  vendor_patch_sha256=hashlib.sha256(patch).hexdigest(),
                  untracked_vendor_weights_images_caches_not_committed=True)
    atomic_json(args.publish_root/"manifests/experiment_publication_provenance.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
