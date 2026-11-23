"""Small physical-Qwen pilot for the new active-observation request protocol.

One common frozen teacher answers the union of four policy requests on four
text-selected queries in three first videos. This validates new real inputs,
not policy quality. Evidence preparation costs are not deployment runtimes.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from flex_uda.a100.mechanism import PAIRS,metadata,generation_guard
from flex_uda.active.acquisition import expression_types
from flex_uda.active.relational_transport import horizontal_direction
from scripts.a100_six_direction import PUBLIC
from scripts.a100_observation_design_prepare import OUT as PREP,POLICIES,STATE as PREP_STATE

OUT=PREP.parent/'teacher_pilot';STATE=OUT/'supervisor/state.json'
SUCCESS='PASS_ACTIVE_OBSERVATION_PHYSICAL_TEACHER_PILOT_NO_GT'
FIELDS=('target','video_id','query_id','query_text','track_id','frame_id','view')


def key(request):return json.dumps([request[k] for k in FIELDS],ensure_ascii=False,separators=(',',':'))


def inputs():
    from scripts.active_run import video_spec
    cases=[];union={};policy_requests={};specs=[]
    for pair in ('C1','C2','C4'):
        target=PAIRS[pair][1];m=metadata(target);video,bank=next(iter(sorted(m['videos'].items())))
        queries=[q for q in m['episodes'] if q['video_id']==video]
        chosen=[queries[0]] if pair!='C2' else [
            next(q for q in queries if horizontal_direction(q['query_text'])),
            next(q for q in queries if 'MOTION' in expression_types(q['query_text']))]
        qids={q['expression_id'] for q in chosen};specs.append(video_spec(target,video,bank))
        cases.append(dict(pair=pair,target=target,video=video,query_ids=sorted(qids)))
        for policy in POLICIES:
            rows=json.loads((PREP/pair/video/(policy+'_requests.json')).read_text())['requests']
            selected=[r for r in rows if r['query_id'] in qids];policy_requests[pair,policy]=selected
            for r in selected:union[key(r)]={f:r[f] for f in FIELDS}
    return cases,specs,union,policy_requests


def run():
    import torch
    from flex_uda.a100.runtime import teacher
    guard=generation_guard();begin=time.perf_counter();torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'completion.json').is_file():return
    phases=[]
    def phase(name,start):
        torch.cuda.synchronize();row=dict(phase=name,seconds=time.perf_counter()-start);phases.append(row)
        atomic_json(OUT/'phase_ledger.json',dict(phases=phases,evidence_preparation_not_deployment=True))
    # Occupancy reservation prevents the existing free-GPU schedulers from
    # allocating a new quality head while teacher weights are still loading.
    t=time.perf_counter();torch.cuda.set_device(0);reservation=torch.empty(256<<20,dtype=torch.float32,device='cuda:0');phase('GPU_context_and_temporary_reservation',t)
    t=time.perf_counter();cases,specs,union,policy_requests=inputs();phase('image_query_manifest_read',t)
    atomic_json(OUT/'questions_before_answers.json',dict(cases=cases,unique_requests=list(union.values()),GT_open_attempts=len(guard),not_to_Git=True))
    checkpoint=OUT/'answers_checkpoint.json';old=json.loads(checkpoint.read_text()) if checkpoint.is_file() else dict(answers={})
    answers=old['answers'];assert set(answers)<=set(union);resumed=len(answers)
    t=time.perf_counter();client=teacher(specs,OUT/'raw.sqlite');phase('teacher_load_and_integrity',t)
    del reservation;torch.cuda.empty_cache();pending=[(k,r) for k,r in union.items() if k not in answers]
    t=time.perf_counter()
    for offset in range(0,len(pending),8):
        batch=pending[offset:offset+8];responses=client.ask_many([r for k,r in batch])
        for (k,request),answer in zip(batch,responses):
            assert key(answer)==k and 0<=answer['yes_probability']<=1
            answers[k]=answer
        atomic_json(checkpoint,dict(answers=answers,current_session_stats=client.stats,not_to_Git=True))
        print(json.dumps(dict(event='OBSERVATION_PILOT_REAL_QWEN',completed=len(answers),total=len(union))),flush=True)
    phase('real_teacher_service_including_image_IO',t)
    t=time.perf_counter();teacher_receipt=client.close();phase('teacher_final_weight_integrity',t)
    assert set(answers)==set(union) and len({r['request_id'] for r in answers.values()})==len(union)
    assert {r['frozen_weight_sha256'] for r in answers.values()}=={teacher_receipt['teacher_weight_sha256']}
    if resumed==0:assert teacher_receipt['new_calls']==len(union) and teacher_receipt['cache_hits']==0
    policy_receipts={}
    for (pair,policy),requests in policy_requests.items():
        result=[dict(answers[key(r)],local_query=r['local_query'],local_track=r['local_track'],allocation_kind=r['allocation_kind']) for r in requests]
        atomic_json(OUT/pair/(policy+'_answers.json'),dict(answers=result,GT_open_attempts=len(guard),only_this_policy_requested_inputs=True,not_to_Git=True))
        policy_receipts[pair+'/'+policy]=dict(logical_requested=len(requests),distinct_query_track_cells=len({(r['query_id'],r['track_id']) for r in requests}),
            requested_views=dict(Counter(r['view'] for r in requests)),effective_views=dict(Counter(r['effective_view'] for r in result)),
            CLIP_fallbacks=sum(r['clip_fallback'] for r in result))
    assert not guard
    report=dict(status=SUCCESS,cases=cases,policies=policy_receipts,unique_physical_questions=len(union),resumed_answer_rows=resumed,
        teacher=teacher_receipt,phases=phases,worker_wall_seconds=time.perf_counter()-begin,
        actual_physical_GPU=os.environ.get('CUDA_VISIBLE_DEVICES'),manual_GT_open_attempts=0,student_or_neural_adaptation_fits=0,
        teacher_model='Qwen3-VL-2B-Instruct; frozen adapter-free base',no_old_teacher_probabilities_read=True,
        scope='Union of four policy requests, one teacher load, four fixed queries on three first videos. New physical evidence pilot only; not an end-to-end single-policy deployment or a HOTA comparison.',
        other_known_quality_GPU_tasks_may_run_on_other_cards=True,source_features_prepaid=True,OS_NAS_cache_not_cleared=True,
        future_use='Only each policy actually requested responses may enter its causal-prefix fit; no other policy or future-video answer privilege.',
        new_quality_predictions=0,goal_status='ACTIVE')
    atomic_json(OUT/'completion.json',report)
    (PUBLIC/'observation_teacher_physical_pilot.json').write_text(json.dumps(report,separators=(',',':'))+'\n')


def main():
    plan=json.loads((PUBLIC/'observation_teacher_pilot_registration.json').read_text())
    for name,sha in plan['source_sha256'].items():assert file_sha(ROOT/name)==sha,name
    begin=time.perf_counter();STATE.parent.mkdir(parents=True,exist_ok=True)
    def state(status,**extra):atomic_json(STATE,dict(status=status,pid=os.getpid(),completed=0,goal_status='ACTIVE',**extra))
    state('AWAITING_FULL_REQUEST_MANIFEST_PREPARATION_FOR_GPU_PILOT')
    while not PREP_STATE.is_file() or json.loads(PREP_STATE.read_text())['status']!='SIX_SETTINGS_ACTIVE_OBSERVATION_MANIFESTS_PREPARED':time.sleep(5)
    state('AWAITING_FREE_PHYSICAL_GPU_0_TO_3_FOR_TEACHER_PILOT')
    while True:
        text=subprocess.check_output(['nvidia-smi','-i','0,1,2,3','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
        memory={int(line.split(',')[0]):int(line.split(',')[1]) for line in text.strip().splitlines()}
        candidates=[g for g in (0,1,2,3) if memory[g]<512]
        if candidates:break
        time.sleep(15)
    gpu=candidates[0];log=OUT/'supervisor/worker.log'
    with log.open('a',buffering=1) as handle:
        proc=subprocess.Popen([sys.executable,'-u',str(Path(__file__)),'--worker'],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4'))
        state('RUNNING_NEW_OBSERVATION_PHYSICAL_TEACHER_PILOT',worker_pid=proc.pid,physical_GPU=gpu,allocation_memory_MiB=memory)
        code=proc.wait()
    if code or not (OUT/'completion.json').is_file():raise RuntimeError(dict(exit_code=code,log=log.read_text()[-8000:]))
    atomic_json(STATE,dict(status=SUCCESS,pid=os.getpid(),completed=1,physical_GPU=gpu,controller_wall_including_wait_seconds=time.perf_counter()-begin,goal_status='ACTIVE'))
    print(SUCCESS,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--worker',action='store_true');args=parser.parse_args()
    if args.worker:run()
    else:
        try:main()
        except Exception as error:
            previous=json.loads(STATE.read_text()) if STATE.is_file() else {};previous.update(status='FAILED_NEW_OBSERVATION_TEACHER_PILOT',error=repr(error),goal_status='ACTIVE')
            atomic_json(STATE,previous);raise
