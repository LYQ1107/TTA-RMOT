"""Check the lightweight branch's snapshot and reachable-object budget."""
import argparse
import json
import re
import subprocess


def git(*args, data=None):
    return subprocess.check_output(["git", *args], input=data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", default="HEAD")
    parser.add_argument("--limit-bytes", type=int, default=100_000_000)
    args = parser.parse_args()
    blocked = re.compile(
        r"\.(?:ckpt|pth|pt|bin|safetensors|onnx|npy|npz|sqlite3?|db|"
        r"gz|zip|tar|tgz|7z|parquet|pkl|pickle)$|"
        r"(?:^|/)[^/]+\.parts(?:/|$)|\.part[-_]\d+(?:\.|$)", re.I
    )
    files = []
    for entry in git("ls-tree", "-rlz", args.ref).split(b"\0"):
        if entry:
            metadata, path = entry.split(b"\t", 1)
            fields = metadata.split()
            files.append((path.decode(), int(fields[3])))
    objects = git("rev-list", "--objects", args.ref)
    object_ids = b"\n".join(line.split(b" ", 1)[0] for line in objects.splitlines())
    sizes = git("cat-file", "--batch-check=%(objectsize)", data=object_ids + b"\n")
    reachable_bytes = sum(int(size) for size in sizes.splitlines())
    forbidden = [path for path, _ in files if blocked.search(path)]
    for line in objects.splitlines():
        fields = line.split(b" ", 1)
        if len(fields) == 2 and blocked.search(fields[1].decode()):
            forbidden.append(fields[1].decode())
    snapshot_bytes = sum(size for _, size in files)
    passed = (snapshot_bytes < args.limit_bytes
              and reachable_bytes < args.limit_bytes and not forbidden)
    print(json.dumps(dict(
        status="PASS" if passed else "FAIL", ref=args.ref, files=len(files),
        snapshot_bytes=snapshot_bytes, reachable_object_bytes=reachable_bytes,
        limit_bytes=args.limit_bytes, forbidden_paths=sorted(set(forbidden))
    ), ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
