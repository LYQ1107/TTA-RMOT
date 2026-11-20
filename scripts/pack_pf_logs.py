#!/usr/bin/env python3
"""Archive complete experiment text logs, excluding images/weights/margin streams."""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from scripts.summarize_pf_full import METHOD_ORDER


def pack(publish):
    publish=Path(publish).resolve()
    paths=set((publish/"manifests").glob("*.log"))
    paths.add(publish/"manifests/query_presence_prototype_gpu.txt")
    for method in METHOD_ORDER:
        metric=json.loads((publish/method.lower()/"metrics.json").read_text())
        if metric["full_audit"]["status"]!="PASS" or metric["full_audit"]["valid_episodes"]!=425:
            raise ValueError("complete logs are exported only after all six full audits")
        merged=Path(metric["raw_output_root"])
        raw=Path(str(merged).removesuffix("_merged"))
        paths.update(raw.glob("shard*/stdout.log"))
        paths.update(raw.glob("shard*/"+method+"/log_rank*.txt"))
        paths.add(merged/method/"trackeval.log")
        paths.update((merged/method).glob("evaluation_history/**/trackeval.log"))
    paths.update((ROOT/"results/runs/pf_guard_equivalence_completebank_20261025").glob("direct/PF_RANK_GUARD/log_rank*.txt"))
    paths.update((ROOT/"results/runs/pf_query_presence_prototype_20261025").glob("model_log/log_rank*.txt"))
    paths=sorted(paths)
    if not all(p.is_file() for p in paths):
        raise ValueError("an expected final log is missing")
    output=publish/"logs"
    output.mkdir(parents=True,exist_ok=True)
    archive=output/"runtime_logs.tar.gz"
    records=[]
    with archive.open("wb") as destination,gzip.GzipFile(filename="",mode="wb",fileobj=destination,mtime=0) as zipped:
        with tarfile.open(fileobj=zipped,mode="w") as tar:
            for path in paths:
                name=path.relative_to(ROOT).as_posix()
                data=path.read_bytes()
                info=tarfile.TarInfo(name)
                info.size=len(data)
                info.mtime=0
                info.mode=0o644
                tar.addfile(info,io.BytesIO(data))
                records.append(dict(path=name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
    with tarfile.open(archive,"r:gz") as tar:
        members=tar.getmembers()
        if len(members)!=len(records):
            raise ValueError("log archive member count changed")
        for member,record in zip(members,records):
            if member.name!=record["path"] or hashlib.sha256(tar.extractfile(member).read()).hexdigest()!=record["sha256"]:
                raise ValueError("log archive content differs from original")
    result=dict(status="VERIFIED_COMPLETE_TEXT_LOG_ARCHIVE",files=len(records),
                uncompressed_bytes=sum(r["bytes"] for r in records),archive_bytes=archive.stat().st_size,
                archive_sha256=file_sha(archive),archive=str(archive.relative_to(ROOT)),records=records,
                scope="full suite and preparation stdout, actual worker/model logs, six official TrackEval logs, Guard verification and bounded frozen prototype",
                raw_images_checkpoints_predictions_margin_streams_excluded=True)
    atomic_json(output/"runtime_log_index.json",result)
    print(json.dumps({k:v for k,v in result.items() if k!="records"},indent=2))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--publish-root",type=Path,default=ROOT/"results/positive_free/full")
    args=p.parse_args()
    pack(args.publish_root)


if __name__=="__main__":main()
