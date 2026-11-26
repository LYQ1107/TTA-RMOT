"""Actual-failure recovery: preserve traces beyond the last valid checkpoint."""
import json
from pathlib import Path
import time
from flex_tta.runtime.pf_receipts import atomic_json


def reconcile_resume(output,state):
    output=Path(output);step=state['step'];tracking=state.get('tracking') or {}
    archive=output/'resume_discarded_traces'/str(time.time_ns());changed={};tail_seconds=0.
    for name in ('metrics_train.jsonl','source_validation.jsonl'):
        path=output/name
        if not path.exists():continue
        lines=path.read_text().splitlines(keepends=True)
        keep=[s for s in lines if json.loads(s)['step']<=step]
        discard=[s for s in lines if json.loads(s)['step']>step]
        if discard:
            archive.mkdir(parents=True,exist_ok=True);(archive/name).write_text(''.join(discard))
            path.write_text(''.join(keep));changed[name]=len(discard)
            if name=='metrics_train.jsonl':
                last=json.loads(discard[-1])['wall_seconds']
                saved=tracking.get('attempt_wall_seconds',tracking.get('wall_seconds',0.))
                tail_seconds=max(0.,last-saved)
    # Historical teacher evidence has one line per actual target candidate,
    # and checkpoint counters count accepted exposures, so its exact prefix
    # can be restored even when old records lack an optimizer-step field.
    path=output/'pseudo_evidence.jsonl'
    if path.exists():
        count=tracking.get('counters',{}).get('target_candidate_rows',0)
        with path.open('rb') as stream:
            for _ in range(count):
                if not stream.readline():raise ValueError('checkpoint evidence prefix missing')
            boundary=stream.tell();discard=stream.read()
        if discard:
            archive.mkdir(parents=True,exist_ok=True);(archive/path.name).write_bytes(discard)
            with path.open('r+b') as stream:stream.truncate(boundary)
            changed[path.name]=discard.count(b'\n')
    record=dict(checkpoint_step=step,archived_uncheckpointed_records=changed,
        archive=str(archive) if changed else None,failed_tail_elapsed_lower_bound_seconds=tail_seconds,
        action='replay unfinished suffix from latest valid checkpoint; do not skip samples')
    atomic_json(output/'resume_trace_recovery.json',record)
    return tail_seconds
