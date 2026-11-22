"""Paired same-raster temporal-input pilot; never a teacher-accuracy claim.

CPU preparation fixes three first-video KITTI queries by text and four genuine
CLIP requests each. The frozen GPU teacher later answers five prompt variants
on identical images. All raw responses stay local; target GT is never used.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from flex_uda.a100.mechanism import metadata,generation_guard
from flex_uda.active.acquisition import expression_types
from flex_uda.active.state_teacher_inputs import StateAwareTeacher,VARIANTS
from scripts.a100_query_observability_audit import explicit_location_cue
from scripts.a100_observation_design_prepare import OUT as ACTIONS,POLICIES
from scripts.a100_six_direction import PUBLIC

OUT=ACTIONS.parent/'state_input_pilot';STATE=OUT/'supervisor/state.json'
SUCCESS='PASS_STATE_PRESERVING_PAIRED_TEACHER_INPUT_PILOT_NO_GT'
FIELDS=('target','video_id','query_id','query_text','track_id','frame_id','view')


def question_key(request):return json.dumps([request[k] for k in FIELDS],separators=(',',':'))


def video():
    from scripts.active_run import video_spec
    m=metadata('KITTI-V2');name,bank=next(iter(sorted(m['videos'].items())))
    return m,name,bank,video_spec('KITTI-V2',name,bank)


def prepare():
    import yaml
    from flex_uda.a100.runtime import relocate
    from flex_uda.active.teacher_client import LocalTeacher
    from flex_uda.teacher.decompose_inputs import PairedViews
    from flex_uda.teacher.yestrack_backend import upstream_functions
    guard=generation_guard();m,name,bank,spec=video()
    queries=[q for q in m['episodes'] if q['video_id']==name]
    composites=[q for q in queries if 'MOTION' in expression_types(q['query_text']) and explicit_location_cue(q['query_text'])]
    chosen=composites[:2]+[next(q for q in queries if 'MOTION' in expression_types(q['query_text']) and not explicit_location_cue(q['query_text']))]
    assert len(chosen)==3 and len({q['expression_id'] for q in chosen})==3
    old=json.loads((PUBLIC/'observation_teacher_physical_pilot.json').read_text());expected_weight=old['teacher']['teacher_weight_sha256']
    config=yaml.safe_load((ROOT/'configs/teacher/base_qwen2b.yaml').read_text());upstream=upstream_functions(relocate(config['yestrack_source']))
    # A metadata-only renderer. No model is instantiated by upstream_functions.
    base=LocalTeacher.__new__(LocalTeacher);base.root=ROOT;base.prompts=yaml.safe_load((ROOT/'configs/teacher/prompts_v1.yaml').read_text())
    base.backend=SimpleNamespace(weight_sha=expected_weight);base.views=PairedViews([spec],upstream);base.image_shas={}
    client=StateAwareTeacher.wrap(base);requests=[];proof=[];filtered=Counter()
    for q in chosen:
        pool={}
        for policy in POLICIES:
            rows=json.loads((ACTIONS/'C2'/name/(policy+'_requests.json')).read_text())['requests']
            for row in rows:
                if row['query_id']==q['expression_id']:pool.setdefault(question_key(row),{k:row[k] for k in FIELDS})
        selected=[]
        for r in pool.values():
            _,_,_,original=client.prepare(dict(r,input_variant='ORIGINAL'))
            if original['effective_view']!='CLIP':filtered['view_fallback']+=1;continue
            if original['clip']['unique_frames']<2:filtered['fewer_than_two_unique_frames']+=1;continue
            variants={}
            for variant in VARIANTS:
                cache_key,images,prompt,info=client.prepare(dict(r,input_variant=variant))
                assert info['input']['visual']==original['input']['visual'] and info['effective_view']=='CLIP'
                assert info['clip']['frames'][-1]==r['frame_id']
                variants[variant]=dict(request_key=cache_key,visual_sha=info['input']['visual'],prompt=prompt)
            assert len({v['request_key'] for v in variants.values()})==5
            assert len({v['visual_sha'] for v in variants.values()})==1
            selected.append(r);proof.append(dict(question=r,variants=variants))
            if len(selected)==4:break
        assert len(selected)==4,(q['expression_id'],len(selected))
        requests.extend(selected)
    assert len(requests)==12 and not guard
    atomic_json(OUT/'questions_before_answers.json',dict(requests=requests,expected_teacher_weight_sha256=expected_weight,
        renderer_proof=proof,filtered_pilot_inputs=dict(filtered),GT_open_attempts=0,not_to_Git=True))
    report=dict(status='STATE_INPUT_REAL_RASTER_AND_METADATA_PREPARATION_PASS',video=name,
        selected_query_ids=[q['expression_id'] for q in chosen],questions=12,variants=list(VARIANTS),variant_inputs=60,
        source_sha256={n:file_sha(ROOT/n) for n in ('scripts/a100_state_teacher_input_pilot.py','flex_uda/active/state_teacher_inputs.py','tests/test_state_teacher_inputs.py')},
        every_variant_identical_visual_sha=True,distinct_prompt_keys_each_question=True,last_frame_is_current=True,
        only_genuine_CLIP_pilot_input_condition=True,filtered_pilot_inputs=dict(filtered),
        manual_GT_open_attempts=0,neural_models_instantiated=0,new_teacher_GPU_calls=0,
        scope='Input construction only. The pilot eligibility is not a filter for full RMOT evaluation. No semantic accuracy/quality claim.',goal_status='ACTIVE')
    atomic_json(OUT/'preparation.json',report)
    (PUBLIC/'state_teacher_input_preparation.json').write_text(json.dumps(report,separators=(',',':'))+'\n')
    print(json.dumps(report),flush=True)


def run():
    import numpy as np
    import torch
    from flex_uda.a100.runtime import teacher
    guard=generation_guard();torch.set_num_threads(4);begin=time.perf_counter();torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if (OUT/'completion.json').is_file():return
    phases=[]
    def phase(name,start):
        torch.cuda.synchronize();phases.append(dict(phase=name,seconds=time.perf_counter()-start))
        atomic_json(OUT/'phase_ledger.json',dict(phases=phases,not_single_method_deployment=True))
    t=time.perf_counter();reservation=torch.empty(256<<20,dtype=torch.float32,device='cuda:0');phase('temporary_GPU_reservation',t)
    prepared=json.loads((OUT/'questions_before_answers.json').read_text());requests=prepared['requests'];_,_,_,spec=video()
    cache_preexisted=(OUT/'raw.sqlite').is_file()
    t=time.perf_counter();client=StateAwareTeacher.wrap(teacher([spec],OUT/'raw.sqlite'));phase('teacher_load_and_integrity',t)
    del reservation;torch.cuda.empty_cache();rows={};receipts={};t_all=time.perf_counter()
    checkpoint=OUT/'answer_checkpoint.json'
    if checkpoint.is_file():rows=json.loads(checkpoint.read_text())['variants']
    resumed=sum(len(v) for v in rows.values())
    for variant in VARIANTS:
        if variant in rows:continue
        t=time.perf_counter();forward=client.stats['forward_seconds']
        result=client.ask_many([dict(r,input_variant=variant) for r in requests]);rows[variant]=result
        receipts[variant]=dict(service_wall_seconds=time.perf_counter()-t,backend_forward_seconds=client.stats['forward_seconds']-forward)
        atomic_json(checkpoint,dict(variants=rows,current_session_stats=client.stats,not_to_Git=True))
        print(json.dumps(dict(event='STATE_INPUT_PHYSICAL_VARIANT',variant=variant,questions=len(result))),flush=True)
    phase('paired_real_teacher_service_including_IO',t_all)
    t=time.perf_counter();receipt=client.close();phase('teacher_final_weight_integrity',t)
    assert receipt['teacher_weight_sha256']==prepared['expected_teacher_weight_sha256']
    if not cache_preexisted:assert receipt['new_calls']==60 and receipt['cache_hits']==0
    originals=rows['ORIGINAL'];summary={}
    for variant in VARIANTS:
        values=rows[variant]
        for old,new,request in zip(originals,values,requests):
            assert question_key(old)==question_key(new)==question_key(request)
            assert old['input']['visual']==new['input']['visual'] and old['visual_tokens']==new['visual_tokens']
            assert new['frozen_weight_sha256']==receipt['teacher_weight_sha256']
        difference=np.array([new['yes_probability']-old['yes_probability'] for old,new in zip(originals,values)])
        summary[variant]=dict(questions=len(values),changed_binary_answers_vs_original=sum((r['yes_probability']>.5)!=(b['yes_probability']>.5) for r,b in zip(values,originals)),
            mean_absolute_probability_change_vs_original=float(np.abs(difference).mean()),mean_signed_probability_change_vs_original=float(difference.mean()),
            mean_language_tokens=float(np.mean([r['language_tokens'] for r in values])),mean_visual_tokens=float(np.mean([r['visual_tokens'] for r in values])),
            top1_yes_no=sum(r['top1_is_yes_no'] for r in values),**receipts.get(variant,dict(reused_completed_variant=True)))
    assert not guard
    report=dict(status=SUCCESS,questions=12,variants=summary,physical_new_calls_this_attempt=receipt['new_calls'],resumed_answer_rows=resumed,
        teacher=receipt,cache_preexisted_this_attempt=cache_preexisted,phases=phases,worker_wall_seconds=time.perf_counter()-begin,physical_GPU=os.environ.get('CUDA_VISIBLE_DEVICES'),
        all_variants_same_raster_and_visual_token_count=True,manual_GT_open_attempts=0,new_adaptation_fits_or_quality_predictions=0,
        primary_fixed_before_teacher_answers='STATE',no_posthoc_confidence_based_variant_selection=True,
        interpretation='Same frozen teacher/images/query; response sensitivity and format only. No manual truth, no claim STATE is more accurate or superior to ordinary distillation. All five controls retained.',
        cost_scope='One shared teacher load and warm crop cache, fixed variant order; preparation costs, not per-variant deployment/Pareto claims.',
        future_opponent='Any adapted main method using STATE must give adequately trained ordinary KD the same STATE evidence.',goal_status='ACTIVE')
    atomic_json(OUT/'completion.json',report)
    (PUBLIC/'state_teacher_input_physical_pilot.json').write_text(json.dumps(report,separators=(',',':'))+'\n')


def main():
    plan=json.loads((PUBLIC/'state_teacher_input_pilot_registration.json').read_text())
    for name,sha in plan['source_sha256'].items():assert file_sha(ROOT/name)==sha,name
    assert (OUT/'preparation.json').is_file();STATE.parent.mkdir(parents=True,exist_ok=True)
    atomic_json(STATE,dict(status='AWAITING_IDLE_GPU_0_TO_3_FOR_PAIRED_STATE_INPUT_PILOT',pid=os.getpid(),completed=0,goal_status='ACTIVE'))
    while True:
        output=subprocess.check_output(['nvidia-smi','-i','0,1,2,3','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
        memory={int(line.split(',')[0]):int(line.split(',')[1]) for line in output.strip().splitlines()};available=[g for g in (0,1,2,3) if memory[g]<512]
        if available:break
        time.sleep(15)
    gpu=available[0];log=STATE.parent/'worker.log'
    with log.open('a',buffering=1) as handle:
        proc=subprocess.Popen([sys.executable,'-u',str(Path(__file__)),'--worker'],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4'))
        atomic_json(STATE,dict(status='RUNNING_PAIRED_STATE_INPUT_REAL_TEACHER',pid=os.getpid(),worker_pid=proc.pid,completed=0,physical_GPU=gpu,goal_status='ACTIVE'))
        code=proc.wait()
    if code or not (OUT/'completion.json').is_file():raise RuntimeError(dict(exit_code=code,log=log.read_text()[-8000:]))
    atomic_json(STATE,dict(status=SUCCESS,pid=os.getpid(),completed=5,physical_GPU=gpu,goal_status='ACTIVE'))
    print(SUCCESS,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--worker',action='store_true');args=parser.parse_args()
    if args.prepare_only:prepare()
    elif args.worker:run()
    else:
        try:main()
        except Exception as error:
            previous=json.loads(STATE.read_text()) if STATE.is_file() else {};previous.update(status='FAILED_STATE_INPUT_PAIRED_TEACHER_PILOT',error=repr(error),goal_status='ACTIVE')
            atomic_json(STATE,previous);raise
