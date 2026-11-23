#!/usr/bin/env python3
"""CPU-only exact merge after all original/recovery workers have completed."""
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.audit_teacher_leakage import METHODS,RUN,validate_receipt
from scripts.launch_teacher_probe_sharded import merge_method
from flex_tta.runtime.pf_receipts import atomic_json,file_sha


def main():
    phase=RUN/'full/T0';manifest=ROOT/'results/teacher_probe/manifest.jsonl'
    if (phase/'phase_completion.json').exists():
        old=json.loads((phase/'phase_completion.json').read_text())
        if old['status']!='COMPLETE_FIXED_PHASE':raise ValueError('unexpected existing final phase status')
        for method in METHODS:
            receipt=json.loads((phase/method/'receipt.json').read_text())
            validate_receipt(receipt,method,file_sha(manifest),8381)
            if file_sha(phase/method/'teacher_outputs.jsonl')!=receipt['predictions_sha256']:raise ValueError('sealed prediction changed')
        print(json.dumps(dict(status='PRESERVED_COMPLETE_PHASE',actual_wall_seconds=old['actual_wall_seconds'])));return
    shards=[phase/'shards'/f'shard{i:02d}'/'manifest.jsonl' for i in range(4)]
    final=[]
    for method in METHODS:
        merged=merge_method(method,manifest,phase,shards)
        final.extend(dict(method=method,shard=i,status='SEALED_FIXED_PROBE',**item)
            for i,item in enumerate(merged['source_worker_receipts']))
    attempts=sorted((phase/'launcher_attempts').glob('*/launcher_status.json'))
    starts=[r['start_epoch_seconds'] for path in attempts for r in json.loads(path.read_text())['workers'] if 'start_epoch_seconds' in r]
    if not starts:raise ValueError('actual worker timestamps are required; no invented wall time')
    status=dict(status='COMPLETE_FIXED_PHASE',phase='full',condition='T0',protocol='DEVELOPMENT_TEACHER_PROBE',
        actual_wall_seconds=time.time()-min(starts),manifest_sha256=file_sha(manifest),workers=final,
        recovery_preserves_all_completed_shards=True,attempt_status_files=[str(p) for p in attempts],
        no_other_process_stopped=True,target_GT_evaluation_not_started=True)
    atomic_json(phase/'phase_completion.json',status)
    print(json.dumps(dict(status=status['status'],merged_methods=list(METHODS),actual_wall_seconds=status['actual_wall_seconds']),indent=2))


if __name__=='__main__':main()
