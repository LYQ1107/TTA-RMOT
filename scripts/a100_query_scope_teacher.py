"""Frozen-teacher text-only structure cache for future scoped learning.

It waits for the current I1 neural stage to release GPUs. No target images,
image QA, prediction or GT enters this extraction. All raw outputs stay local.
Every downstream learner must load only its current/past query structures;
the same structures must be offered to ordinary/factorized KD controls.
"""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from flex_uda.a100.mechanism import metadata,generation_guard
from flex_uda.active.query_scope import decode,consensus
from flex_uda.active.query_scope_prompts import PROMPTS
from scripts.a100_six_direction import OUT as SIX,PUBLIC
from scripts.a100_observation_design_quality import STATE as I1_STATE
from scripts.a100_observation_design_collect import TEACHER_SHA

OUT=SIX/'query_scope_teacher_20261123'
STATE=OUT/'supervisor/state.json'
REGISTRATION=PUBLIC/'query_scope_teacher_registration_v1.json'
SUCCESS='PASS_QUERY_TEXT_SCOPE_TEACHER_PREPARATION_NO_GT'


def text_key(text):return hashlib.sha256(text.encode()).hexdigest()


def prepare():
    guard=generation_guard();texts=defaultdict(set);counts={}
    for target in ('Dance','KITTI-V2','SportsMOT'):
        queries=metadata(target)['episodes'];counts[target]=dict(episodes=len(queries),unique_texts=len({q['query_text'] for q in queries}))
        for q in queries:
            assert q['query_text'].strip();texts[q['query_text']].add(target)
    # The first batch checks real formats from each task family without
    # consulting image responses, predictions or target labels.
    priority=[]
    for target in ('Dance','SportsMOT','KITTI-V2'):
        pool=sorted(t for t,owners in texts.items() if target in owners)
        priority.append(pool[0])
    kitti=sorted(t for t,owners in texts.items() if 'KITTI-V2' in owners)
    for word in ('left','direction','than','white','parked'):
        options=[t for t in kitti if word in t.lower() and t not in priority]
        if options:priority.append(options[0])
    priority=list(dict.fromkeys(priority))
    ordered=priority+sorted(set(texts)-set(priority),key=text_key)
    assert len(ordered)==len(texts)
    rows=[dict(key=text_key(t),text=t,targets=sorted(texts[t])) for t in ordered]
    assert len({r['key'] for r in rows})==len(rows) and not guard
    atomic_json(OUT/'input_manifest.json',dict(queries=rows,only_task_texts=True,
        no_image_answers_or_predictions_or_manual_GT_read=True,not_to_Git=True))
    report=dict(status='SEALED_QUERY_TEXT_SCOPE_INPUTS',target_counts=counts,
        unique_queries=len(rows),fixed_prompts=list(PROMPTS),total_generation_prompts=2*len(rows),
        same_query_only_preprocessing_for_every_downstream_method=True,
        future_image_answers_or_features_not_inputs=True,
        downstream_prefix_rule='Load only current/past query structures; never aggregate future-video structures into learned normalization or parameters.',
        manual_GT_open_attempts=0,new_GPU_generation_calls=0,goal_status='ACTIVE')
    atomic_json(OUT/'preparation.json',report)
    (PUBLIC/'query_scope_teacher_input_audit.json').write_text(json.dumps(report,separators=(',',':'))+'\n')


def worker():
    import torch
    import yaml
    from flex_uda.a100.runtime import relocate
    from flex_uda.active.teacher_client import REVISION
    from flex_uda.teacher.yestrack_backend import FrozenYesTrack
    if (OUT/'completion.json').is_file():return
    guard=generation_guard();torch.set_num_threads(4);torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    start=time.perf_counter();torch.cuda.set_device(0)
    reservation=torch.empty(256<<20,dtype=torch.float32,device='cuda:0')
    plan=json.loads(REGISTRATION.read_text());rows=json.loads((OUT/'input_manifest.json').read_text())['queries']
    expected=len(rows)*len(PROMPTS);phases=[];checkpoint=OUT/'raw_outputs.json'
    previous=json.loads(checkpoint.read_text()) if checkpoint.is_file() else None
    prompt_shas={v:hashlib.sha256(p.encode()).hexdigest() for v,p in PROMPTS.items()}
    if previous is not None:
        assert previous['teacher_weight_SHA']==TEACHER_SHA and previous['prompt_sha256']==prompt_shas
        assert previous['registered_source_sha256']==plan['source_sha256']
    answers=previous['variants'] if previous is not None else {}
    resumed=sum(len(v) for v in answers.values());progress_path=OUT/'worker_progress.json'
    def progress(phase):atomic_json(progress_path,dict(phase=phase,completed_prompts=sum(len(v) for v in answers.values()),
        expected_prompts=expected,physical_GPU=os.environ.get('CUDA_VISIBLE_DEVICES')))
    progress('frozen_teacher_load_and_integrity')
    config=yaml.safe_load((ROOT/'configs/teacher/base_qwen2b.yaml').read_text())
    config.update(model_path=str(ROOT/'assets/teacher_Qwen3-VL-2B-Instruct'),yestrack_source=str(relocate(config['yestrack_source'])))
    assert config['revision']==REVISION and config['lora_path'] is None
    t=time.perf_counter();backend=FrozenYesTrack(config,device='cuda:0');torch.cuda.synchronize()
    phases.append(dict(phase='teacher_load_and_integrity',seconds=time.perf_counter()-t))
    assert backend.weight_sha==TEACHER_SHA
    # Only instrumentation on this separate model instance is changed.
    # Its original hooks record binary Yes/No, which is not this task.
    backend.pre_hook.remove();backend.post_hook.remove();actual_forward_calls=0
    def count_forward(module,args,kwargs):
        nonlocal actual_forward_calls
        actual_forward_calls+=1
    handle=backend.model.register_forward_pre_hook(count_forward,with_kwargs=True)
    backend.processor.tokenizer.padding_side='left';del reservation;torch.cuda.empty_cache()
    input_tokens=generated_tokens=padded_generation_token_slots=0;t_all=time.perf_counter()
    for variant,prompt in PROMPTS.items():
        answers.setdefault(variant,{})
        pending=[r for r in rows if r['key'] not in answers[variant]]
        for offset in range(0,len(pending),plan['batch_size']):
            batch=pending[offset:offset+plan['batch_size']]
            messages=[[dict(role='system',content=[dict(type='text',text=prompt)]),
                dict(role='user',content=[dict(type='text',text=json.dumps(dict(description=r['text']),ensure_ascii=False))])]
                for r in batch]
            inputs=backend.processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,
                return_dict=True,return_tensors='pt',padding=True).to('cuda:0')
            assert not any(k.startswith('pixel_values') for k in inputs)
            assert not bool((inputs['input_ids']==backend.model.config.image_token_id).any())
            width=inputs['input_ids'].shape[1];torch.cuda.synchronize();t=time.perf_counter()
            with torch.inference_mode():
                output=backend.model.generate(**inputs,max_new_tokens=plan['max_new_tokens'],
                    do_sample=False,num_beams=1,use_cache=True)
            torch.cuda.synchronize();elapsed=time.perf_counter()-t
            generated=output[:,width:];strings=backend.processor.batch_decode(generated,skip_special_tokens=True)
            input_tokens+=int(inputs['attention_mask'].sum());pad=backend.processor.tokenizer.pad_token_id
            generated_tokens+=int((generated!=pad).sum());padded_generation_token_slots+=generated.numel()
            for r,response in zip(batch,strings):
                parsed=decode(r['text'],response)
                answers[variant][r['key']]=dict(text=r['text'],raw_response=response,parsed=parsed,
                    batch_generation_wall_seconds=elapsed,batch_size=len(batch),teacher_weight_SHA=TEACHER_SHA)
            backend.check_versions()
            atomic_json(checkpoint,dict(variants=answers,teacher_weight_SHA=TEACHER_SHA,
                prompt_sha256=prompt_shas,registered_source_sha256=plan['source_sha256'],not_to_Git=True))
            progress('text_only_scope_generation')
            print(json.dumps(dict(event='TEXT_SCOPE_GENERATION',variant=variant,
                completed=len(answers[variant]),total=len(rows))),flush=True)
            # This fixed format gate precedes any image labels or quality.
            if offset==0 and len(batch)>=8:
                valid=sum(answers[variant][r['key']]['parsed']['valid_format'] for r in batch)
                assert valid>=plan['first_batch_minimum_valid'],('Firstbatchschemafailure',variant,valid)
    phases.append(dict(phase='text_generation_and_checkpoint_IO',seconds=time.perf_counter()-t_all))
    handle.remove();t=time.perf_counter();frozen=backend.seal_frozen()
    phases.append(dict(phase='teacher_final_weight_integrity',seconds=time.perf_counter()-t))
    counts=Counter();structures={}
    for row in rows:
        one=answers['PARSE'][row['key']]['parsed'];two=answers['SCOPE_REVIEW'][row['key']]['parsed']
        combined=consensus(row['text'],one,two);structures[row['key']]=combined
        counts.update(queries=1,valid_parse=one['valid_format'],valid_scope_review=two['valid_format'],
            full_structure_consensus=combined['full_structure_consensus'],
            with_known_subject_appearance_terms=bool(combined['known_subject_appearance_terms']),
            with_ambiguity_notes=bool(combined['all_ambiguity_notes']))
    assert len(structures)==len(rows) and not guard
    atomic_json(OUT/'structures.json',dict(structures=structures,no_target_truth=True,not_to_Git=True))
    report=dict(status=SUCCESS,counts=dict(counts),logical_text_generation_prompts=expected,
        resumed_text_outputs=resumed,actual_new_generation_prompts=expected-resumed,
        actual_model_forward_calls=actual_forward_calls,language_input_tokens=input_tokens,
        nonpadding_generated_tokens=generated_tokens,padded_generation_token_slots=padded_generation_token_slots,
        teacher=frozen,legacy_yes_no_counter_does_not_count_text_generation=True,
        physical_GPU=os.environ.get('CUDA_VISIBLE_DEVICES'),phases=phases,
        worker_wall_seconds=time.perf_counter()-start,manual_target_GT_open_attempts=0,
        images_or_image_QA_or_quality_predictions_read=0,new_image_teacher_questions=0,
        new_student_or_adaptation_fits=0,raw_outputs_and_full_structures_not_to_Git=True,
        no_posthoc_prompt_promoted=True,teacher_agreement_not_semantic_accuracy=True,
        cost_scope='One shared frozen teacher load for language-only preprocessing, not an RMOT deployment cost or accuracy gain.',
        downstream_fairness='All ordinary/factorized KD receive identical structures; prefix adaptation reads no future-video information.',
        registered_source_sha256=plan['source_sha256'],
        goal_status='ACTIVE',research_goal_complete=False)
    atomic_json(OUT/'completion.json',report)
    (PUBLIC/'query_scope_teacher_receipt.json').write_text(json.dumps(report,separators=(',',':'))+'\n')
    progress('sealed')


def main(prepare_only=False):
    plan=json.loads(REGISTRATION.read_text())
    for name,sha in plan['source_sha256'].items():assert file_sha(ROOT/name)==sha,name
    STATE.parent.mkdir(parents=True,exist_ok=True)
    if (OUT/'completion.json').is_file():
        completed=json.loads((OUT/'completion.json').read_text())
        assert completed['registered_source_sha256']==plan['source_sha256']
        atomic_json(STATE,dict(status=SUCCESS,pid=os.getpid(),completed=completed['logical_text_generation_prompts'],
            reused_sealed_without_new_model_load=True,goal_status='ACTIVE'))
        return
    prepare()
    if prepare_only:return
    atomic_json(STATE,dict(status='AWAITING_I1_GPU_STAGE_RELEASE_FOR_TEXT_SCOPE',pid=os.getpid(),completed=0,goal_status='ACTIVE'))
    while True:
        previous=json.loads(I1_STATE.read_text())
        if previous.get('failure') or previous['status'].startswith('FAILED_'):raise RuntimeError('Resolve I1 failure first')
        if previous.get('completed')==18 and not previous.get('active') and not previous.get('pending'):break
        time.sleep(10)
    while True:
        text=subprocess.check_output(['nvidia-smi','-i','0,1,2,3','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
        memory={int(r.split(',')[0]):int(r.split(',')[1]) for r in text.strip().splitlines()};free=[g for g in (0,1,2,3) if memory[g]<512]
        if free:break
        time.sleep(15)
    gpu=free[0];log=STATE.parent/'worker.log'
    with log.open('a',buffering=1) as output:
        proc=subprocess.Popen([sys.executable,'-u',str(Path(__file__)),'--worker'],cwd=ROOT,
            stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4'))
        while proc.poll() is None:
            path=OUT/'worker_progress.json';progress=json.loads(path.read_text()) if path.is_file() else {}
            atomic_json(STATE,dict(status='RUNNING_FROZEN_QUERY_TEXT_SCOPE_EXTRACTION',pid=os.getpid(),
                worker_pid=proc.pid,physical_GPU=gpu,completed=progress.get('completed_prompts',0),progress=progress,goal_status='ACTIVE'))
            time.sleep(15)
    if proc.returncode or not (OUT/'completion.json').is_file():raise RuntimeError(dict(exit_code=proc.returncode,log=log.read_text()[-7000:]))
    result=json.loads((OUT/'completion.json').read_text())
    atomic_json(STATE,dict(status=SUCCESS,pid=os.getpid(),completed=result['logical_text_generation_prompts'],physical_GPU=gpu,goal_status='ACTIVE'))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--worker',action='store_true');args=parser.parse_args()
    if args.worker:worker()
    else:
        try:main(args.prepare_only)
        except Exception as error:
            previous=json.loads(STATE.read_text()) if STATE.is_file() else {}
            previous.update(status='FAILED_QUERY_TEXT_SCOPE_EXTRACTION',error=repr(error),goal_status='ACTIVE');atomic_json(STATE,previous);raise
