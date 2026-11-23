"""Collect the predeclared paired three-view bank; frozen teacher, no GT."""
import time
PROCESS_BEGIN = time.perf_counter()
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json, append_json, file_sha
from flex_uda.a100.mechanism import OUT, metadata, generation_guard


def main():
    import torch
    from flex_uda.a100.runtime import teacher
    from scripts.active_run import video_spec
    attempts=generation_guard(); folder=OUT/'evidence';folder.mkdir(parents=True,exist_ok=True)
    rows=json.loads((OUT/'evidence_manifest.json').read_text())['samples']
    manifest_sha=file_sha(OUT/'evidence_manifest.json')
    completed=folder/'completion.json'
    if completed.is_file():
        r=json.loads(completed.read_text());assert r['manifest_sha256']==manifest_sha
        print(json.dumps(dict(event='EVIDENCE_ALREADY_COMPLETE',receipt=r)),flush=True);return
    specs=[]
    for target in sorted({s['target'] for s in rows}):
        m=metadata(target)
        for video in sorted({s['video_id'] for s in rows if s['target']==target}):
            specs.append(video_spec(target,video,m['videos'][video]))
    requests=[];identities=[]
    for sample in rows:
        for view in ('CROP','CLIP','FRAME_BOX'):
            request={k:sample[k] for k in ('target','video_id','query_id','query_text','track_id','frame_id')}
            request['view']=view;requests.append(request);identities.append((sample['sample_id'],view))
    assert len(requests)<=900
    phases={};t=time.perf_counter()
    print(json.dumps(dict(event='EVIDENCE_TEACHER_LOAD_BEGIN',physical_GPU=__import__('os').environ.get('CUDA_VISIBLE_DEVICES'))),flush=True)
    client=teacher(specs,folder/'teacher.sqlite')
    phases['teacher_load']=time.perf_counter()-t
    atomic_json(folder/'current_session_loaded.json',dict(teacher_load_seconds=phases['teacher_load'],
        teacher_weight_sha256=client.backend.weight_sha))
    saved={}
    path=folder/'responses.jsonl'
    if path.is_file():
        saved={(r['sample_id'],r['requested_view']):r for r in map(json.loads,path.read_text().splitlines())}
    t=time.perf_counter()
    for start in range(0,len(requests),8):
        block=[i for i in range(start,min(start+8,len(requests))) if identities[i] not in saved]
        if not block:continue
        returned=client.ask_many([requests[i] for i in block])
        for i,response in zip(block,returned):
            sample_id,view=identities[i]
            row=dict(response,sample_id=sample_id,requested_view=view)
            append_json(path,row);saved[sample_id,view]=row
        print(json.dumps(dict(event='EVIDENCE_PROGRESS',completed=len(saved),total=len(requests),new_calls=client.stats['new_calls'])),flush=True)
    phases['teacher_service']=time.perf_counter()-t
    t=time.perf_counter();receipt=client.close();phases['final_frozen_seal']=time.perf_counter()-t
    assert len(saved)==len(requests) and not attempts
    assert receipt['new_calls']<=900
    assert {r['frozen_weight_sha256'] for r in saved.values()}=={receipt['teacher_weight_sha256']}
    # The first600 returned before a missing-image-path exception. They remain
    # valid: per-forward frozen version checks passed. Preserve the failed
    # process cost and its lack of a terminal hash seal, rather than silently
    # charging only the resumed300 requests.
    sessions=[]
    previous=folder/'attempt00_failed_supervisor_receipt.json'
    if previous.is_file():sessions.append(json.loads(previous.read_text()))
    aggregate=dict(receipt,logical_calls=len(saved),new_calls=sum(not r['cache_hit'] for r in saved.values()),
        cache_hits=sum(bool(r['cache_hit']) for r in saved.values()),
        forward_seconds=sum(r['batch_forward_seconds']/r['batch_size'] for r in saved.values() if not r['cache_hit']),
        equivalent_forward_seconds=sum(r['batch_forward_seconds']/r['batch_size'] for r in saved.values()),
        batches=None,aggregation='returned responses across attempts; original pre-forward file-open failure did not launch an unreturned batch')
    assert aggregate['new_calls']<=900
    result=dict(status='PASS_FROZEN_PAIRED_EVIDENCE',teacher=aggregate,final_session_teacher=receipt,phase_wall_seconds=phases,
        process_wall_seconds=time.perf_counter()-PROCESS_BEGIN,physical_GPU=__import__('os').environ.get('CUDA_VISIBLE_DEVICES'),
        responses=len(saved),locations=len(rows),manifest_sha256=manifest_sha,
        responses_sha256=file_sha(path),GT_open_attempts=len(attempts),
        clip_fallback_responses=sum(bool(r['clip_fallback']) for r in saved.values()),failed_sessions=sessions,
        failed_first_session_terminal_weight_hash_available=False if sessions else None,
        all_returned_responses_share_final_frozen_parameter_hash=True,
        phase_wall_seconds_scope='final resumed session only; failed attempt cost retained separately',
        all_attempts_wall_seconds=sum(r['wall_seconds'] for r in sessions)+time.perf_counter()-PROCESS_BEGIN,
        preparation_cost_not_deployment_benchmark=True)
    atomic_json(completed,result)
    print(json.dumps(dict(event='EVIDENCE_COMPLETE',receipt=result)),flush=True)


if __name__=='__main__':main()
