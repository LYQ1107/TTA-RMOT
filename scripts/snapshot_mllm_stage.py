#!/usr/bin/env python3
"""Publish small real receipts and complete text logs, never weights or GT."""
from datetime import datetime,timezone
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tarfile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json,file_sha

RUN=ROOT/'results/runs/mllm_val_distill_20261028_v1'
PUBLIC=ROOT/'results/mllm_distillation'
PSEUDO=Path('/data2/user/TTA-RMOT-assets/mllm-distill/val_pseudo_v1')


def snapshot():
    assets=PUBLIC/'assets';assets.mkdir(parents=True,exist_ok=True);records={}
    for stage in ('interface_v6','smoke_v2'):
        folder=RUN/stage
        for name in ('training_receipt.json','provenance.json','initial_frozen_equivalence.json','source_only_lambda_validation.json'):
            path=folder/name
            if path.exists():shutil.copy2(path,assets/(stage+'_'+name))
        records[stage]=json.loads((folder/'training_receipt.json').read_text())
    if records['interface_v6']['optimizer_updates'] or records['smoke_v2']['steps_per_cycle']!=8973:
        raise ValueError('invalid interface/source-complete smoke')
    sealed_copy=assets/'pseudo_seal.json'
    if not sealed_copy.exists():shutil.copy2(PSEUDO/'seal.json',sealed_copy)
    elif file_sha(sealed_copy)!=file_sha(PSEUDO/'seal.json'):raise ValueError('preserve the previously published immutable pseudo seal')
    full=RUN/'MLLM_DISTILL';training=PUBLIC/'training'
    if (full/'training_receipt.json').exists():
        training.mkdir(exist_ok=True)
        for name in ('training_receipt.json','provenance.json','config.json','source_validation.jsonl','source_only_lambda_validation.json','initial_frozen_equivalence.json','teacher_fit_summary.json','teacher_fit_after_training.jsonl.gz'):
            if (full/name).exists():shutil.copy2(full/name,training/name)
        with (full/'metrics_train.jsonl').open('rb') as src,(training/'metrics_train.jsonl.gz').open('wb') as dst:
            with gzip.GzipFile(filename='',fileobj=dst,mode='wb',mtime=0) as compressed:shutil.copyfileobj(src,compressed)
    archives=[]
    for name in ('manifest.jsonl','soft_labels.jsonl','runtime.jsonl','windows.jsonl'):
        target=assets/(name+'.gz')
        with (PSEUDO/name).open('rb') as src,target.open('wb') as dst:
            with gzip.GzipFile(filename='',fileobj=dst,mode='wb',mtime=0) as compressed:shutil.copyfileobj(src,compressed)
        with gzip.open(target,'rb') as stream:
            if hashlib.sha256(stream.read()).hexdigest()!=file_sha(PSEUDO/name):raise ValueError('compressed cache copy changed')
        archives.append(dict(path=str(target.relative_to(ROOT)),compressed_sha256=file_sha(target),uncompressed_sha256=file_sha(PSEUDO/name)))
    files=sorted(p for p in RUN.rglob('*') if p.is_file() and p.name not in ('gt.txt','predict.txt','full_evaluation_launcher_stdout.txt') and (p.suffix=='.txt' or p.name in
        ('training_receipt.json','provenance.json','config.json','source_only_lambda_validation.json','initial_frozen_equivalence.json','progress.json','source_validation.jsonl','metrics_train.jsonl','numerical_retries.jsonl') or
        'attempt_runtimes' in p.parts or 'admission' in p.name and p.suffix=='.json'))
    logs=PUBLIC/'logs';logs.mkdir(exist_ok=True);archive=logs/'complete_training_and_smoke_text_logs.tar.gz';members=[]
    with tarfile.open(archive,'w:gz') as tar:
        for path in files:
            relative=str(path.relative_to(RUN));tar.add(path,arcname=relative,recursive=False)
            members.append(dict(member=relative,sha256=file_sha(path),bytes=path.stat().st_size))
    with tarfile.open(archive,'r:gz') as tar:
        if len(tar.getnames())!=len(members):raise ValueError('missing text archive member')
        for r in members:
            if hashlib.sha256(tar.extractfile(r['member']).read()).hexdigest()!=r['sha256']:raise ValueError('log archive mismatch')
    atomic_json(logs/'archive_index.json',dict(path=str(archive.relative_to(ROOT)),sha256=file_sha(archive),members=members,
        includes_failed_and_revoked_attempts=True,excludes=['images','weights','GT','raw SQLite','source label plans']))
    result=dict(status='COMPLETE_TRAINING_AWAITING_FULL_EVALUATION' if (full/'training_receipt.json').exists() else 'INTERFACE_AND_VALID_STUDENT_SMOKE_COMPLETE',updated_at_utc=datetime.now(timezone.utc).isoformat(),
        formal_training_complete=(RUN/'MLLM_DISTILL/training_receipt.json').exists(),
        target_protocol='25Val same-set transductive development diagnosis, NOT Train-to-Val',
        stage_receipts={k:dict(status=v['status'],wall_seconds=v['wall_seconds'],target_GT_open_attempts=v['target_GT_open_attempts']) for k,v in records.items()},
        compressed_cache_copies=archives,log_archive_members=len(members),run_root=str(RUN),large_checkpoints_retained_on_server=True)
    atomic_json(PUBLIC/'stage_status.json',result);print(json.dumps(result,indent=2));return result


if __name__=='__main__':snapshot()
