"""Prospective joint scene-and-fovea evidence, before training a new teacher.

Old local clips removed camera/background context. This tests a different
observation, without turning geometric residuals into semantic pseudo-labels.
All72 outputs are sealed before a separate canonical-membership diagnosis.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json, file_sha
from scripts.a100_six_direction import OUT as SIX

OUT = SIX/'context_pair_teacher_pilot_20261125'
PLAN = OUT/'registration.json'
PUBLIC = ROOT/'docs/a100_six_direction_20261122'
DATA = SIX/'appearance_learning_common_20261125'
SOURCES = ('scripts/a100_context_pair_teacher_pilot.py', 'flex_uda/a100/mechanism.py',
           'flex_uda/a100/runtime.py', 'scripts/active_run.py',
           'flex_uda/teacher/yestrack_backend.py', 'flex_uda/teacher/decompose_inputs.py',
           'flex_uda/teacher/visual_input_builder.py', 'configs/teacher/base_qwen2b.yaml',
           'configs/teacher/prompts_v1.yaml', 'flex_uda/active/deterministic_observation_runtime.py')


def rank(key):
    return hashlib.sha256(json.dumps(key).encode()).hexdigest()


def prepare():
    import numpy as np
    import yaml
    from flex_uda.a100.mechanism import metadata, generation_guard
    from flex_uda.a100.runtime import relocate
    from flex_uda.teacher.yestrack_backend import upstream_functions
    from flex_uda.teacher.decompose_inputs import PairedViews, images_sha
    from scripts.active_run import video_spec
    guard = generation_guard(); assert not PLAN.exists()
    config = yaml.safe_load((ROOT/'configs/teacher/base_qwen2b.yaml').read_text())
    templates = yaml.safe_load((ROOT/'configs/teacher/prompts_v1.yaml').read_text())
    old = json.loads((DATA/'data_preparation.json').read_text())
    selection, specs, provenance = [], [], []
    for case in old['cases']:
        pair, target, video = case['pair'], case['target'], case['evaluated_video']
        folder = DATA/pair; assert file_sha(folder/'teacher_inputs.npz') == case['data_sha256']
        keys = json.loads((folder/'row_keys.json').read_text())['keys']
        with np.load(folder/'teacher_inputs.npz') as saved: probability = saved['probability'][:, 0]
        eligible = [i for i, k in enumerate(keys) if k[0] == video and np.isfinite(probability[i])]
        selected = []
        for positive in (True, False):
            choices = [i for i in eligible if (probability[i] > .5) == positive]
            selected.extend(sorted(choices, key=lambda i: rank(keys[i]))[:4])
        # Preserve rare original-teacher classes, then challenge near its
        # boundary. No canonical membership or prior method output is read.
        remaining = [i for i in eligible if i not in selected]
        selected.extend(sorted(remaining, key=lambda i: (abs(probability[i]-.5), rank(keys[i])))[:12-len(selected)])
        assert len(selected) == 12 and len(set(selected)) == 12
        md = metadata(target); bank = md['videos'][video]
        queries = {q['expression_id']: q['query_text'] for q in md['episodes'] if q['video_id'] == video}
        specs.append(video_spec(target, video, bank))
        for i in selected:
            _, qid, tid, frame = keys[i]
            selection.append(dict(pair=pair, target=target, video=video, key=keys[i],
                query_text=queries[qid], track=int(tid), frame=int(frame),
                selection_original_CROP_probability=float(probability[i])))
        provenance.append(dict(pair=pair, data_sha256=case['data_sha256'],
            row_keys_sha256=file_sha(folder/'row_keys.json')))
    views = PairedViews(specs, upstream_functions(relocate(config['yestrack_source'])))
    cases = []
    for index, row in enumerate(selection):
        key = row['target'], row['video']; frame, tid = row['frame'], row['track']
        ob = views.observations[key, frame, tid]; full = views.frame(key, frame)[0]
        history = [o for o in views.tracks[key, tid] if o.frame_id < frame]
        distant = [o for o in history if o.frame_id <= frame-32]
        previous = distant[-1] if distant else (history[0] if history else None)
        images = []; roles = []
        if previous is not None:
            images.extend([views.frame_box(key, previous.frame_id, tid), views.patch(key, previous)])
            roles.extend(['earlier_scene_with_red_box', 'earlier_target_crop'])
        images.extend([views.frame_box(key, frame, tid), views.patch(key, ob)])
        roles.extend(['current_scene_with_red_box', 'current_target_crop'])
        paths = []
        folder = OUT/'images'/str(index); folder.mkdir(parents=True, exist_ok=True)
        for i, image in enumerate(images):
            path = folder/f'{i}.png'; image.save(path); paths.append(str(path.relative_to(OUT)))
        coordinate = f'{(ob.bbox_xyxy[0]+ob.bbox_xyxy[2])/2/full.width:.3f} {(ob.bbox_xyxy[1]+ob.bbox_xyxy[3])/2/full.height:.3f}'
        crop_prompt = templates['official_single'].format(coord=coordinate, sentence=row['query_text'])
        if previous is None:
            context = 'The images show the current scene with the candidate inside a red box, followed by a close view of that candidate. No earlier observation is supplied.'
        else:
            context = 'The images are ordered: earlier scene with a red box, earlier close view, current scene with a red box, current close view. The red boxes mark a provided candidate track. Use the earlier scene as temporal and camera/background context; answer about the object in the CURRENT scene.'
        context += f' Does the current marked object match the complete description: "{row["query_text"]}"? Answer Yes or No.'
        cases.append(dict(index=index, **row, previous_frame=None if previous is None else previous.frame_id,
            frame_gap=None if previous is None else frame-previous.frame_id,
            temporal_context_missing=previous is None, image_paths=paths, image_roles=roles,
            context_images_sha256=images_sha(images), crop_image_sha256=images_sha([images[-1]]),
            crop_prompt=crop_prompt, context_prompt=context))
        for image in images: image.close()
    assert len(cases) == 36 and not guard
    plan = dict(status='REGISTERED_PAIRED_LOCAL_VS_SCENE_CONTEXT_TEACHER_DIAGNOSIS', cases=cases,
        source_sha256={n: file_sha(ROOT/n) for n in SOURCES}, data_provenance=provenance,
        original_teacher_sha256='9a13380b4279bc98f4680bd083ff4995fdcfec120cfded459513be90d041a956',
        hypothesis='Joint background/target views can resolve some state evidence missing from local crops. A numerical F/H residual is never assigned semantic truth.',
        human_visual_inspiration='Inspect the object, then compare it with the surrounding scene at an earlier moment; preserve the question and the current target.',
        classical_thought='Observability and nuisance reference frames determine which observation is informative. This is an input hypothesis, not a new calibration algorithm or claimed theorem.',
        selection='First videos C1/C2/C4; up to4oldCROPYes+4No by SHA, then nearest-boundary unused events to12/case. Uses only frozen teacher replies, never GT or student scores.',
        prior_DEV_exposure=True, no_semantic_GT_labels_for_generation=True,
        temporal_rule='Closest same provided-track observation at least32raw frames earlier; otherwise earliest available past observation; if no past, current views only with explicit missing flag. No fabricated duplicate motion.',
        controls='Both originalCROP and newSCENE_CONTEXT recomputed freshly, one sample/batch, same frozen teacher; no old cache/precision mixture.',
        true_cost_difference='Context uses2or4images versus1crop. Report actual token/time difference; not iso-compute, HOTA or a main-method comparison.',
        actual_inputs=72, cases_per_domain=12, seed=0, physical_GPU=2, memory_fraction=.25,
        original_base_and_all_prior_services_preserved=True,
        no_adapter_training_or_RMOT_student_predictions=True,
        postseal_evaluation='Only after all72 answers and original-teacher seal, separate process compares paired binary responses with existing nominal canonical membership. Tiny teacher-stratified DEV sample, not pure semantics, statistical proof or module acceptance.',
        future_fairness='Every future ordinary KD receives the same new responses/images/physical information, including an adapted-teacher direct-KD control if teacher training is later introduced.',
        all_data_DEV=True, goal_status='ACTIVE')
    atomic_json(PLAN, plan)
    atomic_json(PUBLIC/'context_pair_teacher_pilot_registration_20261125.json',
                {k: v for k, v in plan.items() if k != 'cases'})
    print(json.dumps(dict(status=plan['status'], cases=36, real_teacher_inputs=72,
        missing_temporal_context=sum(c['temporal_context_missing'] for c in cases))), flush=True)


def registration():
    plan = json.loads(PLAN.read_text())
    for n, sha in plan['source_sha256'].items(): assert file_sha(ROOT/n) == sha, n
    return plan


def worker():
    from flex_uda.active.deterministic_observation_runtime import configure_before_cuda
    execution = configure_before_cuda()
    import torch
    import yaml
    from PIL import Image
    from flex_uda.a100.mechanism import generation_guard
    from flex_uda.a100.runtime import relocate
    from flex_uda.teacher.yestrack_backend import FrozenYesTrack
    from flex_uda.teacher.decompose_inputs import images_sha
    guard = generation_guard(); plan = registration()
    torch.set_num_threads(2); torch.cuda.set_device(0); torch.cuda.set_per_process_memory_fraction(.25)
    torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False); torch.backends.cuda.enable_math_sdp(True)
    config = yaml.safe_load((ROOT/'configs/teacher/base_qwen2b.yaml').read_text())
    config.update(model_path=str(ROOT/'assets/teacher_Qwen3-VL-2B-Instruct'), yestrack_source=str(relocate(config['yestrack_source'])))
    started = time.perf_counter(); teacher = FrozenYesTrack(config)
    assert teacher.weight_sha == plan['original_teacher_sha256']
    print(json.dumps(dict(status='SCENE_CONTEXT_ORIGINAL_QWEN_LOADED',load_seconds=teacher.load_seconds)), flush=True)
    path = OUT/'answers.jsonl'; assert not path.exists()
    with path.open('w', buffering=1) as stream:
        for case in plan['cases']:
            images = []
            for relative in case['image_paths']:
                with Image.open(OUT/relative) as im: images.append(im.convert('RGB'))
            assert images_sha(images) == case['context_images_sha256'] and images_sha([images[-1]]) == case['crop_image_sha256']
            for condition, group, prompt in [('CROP',[images[-1]],case['crop_prompt']),('SCENE_CONTEXT',images,case['context_prompt'])]:
                response = teacher.forward([group], [prompt])[0]
                stream.write(json.dumps(dict(case=case['index'], pair=case['pair'], condition=condition, **response))+'\n')
            for im in images: im.close()
            print(json.dumps(dict(status='PAIRED_CONTEXT_TEACHER_CASE_COMPLETE',case=case['index']+1,total=36)),flush=True)
    frozen = teacher.seal_frozen(); assert not guard
    assert len(path.read_text().splitlines()) == 72
    report = dict(status='ALL72_CONTEXT_CONTROL_ANSWERS_SEALED_BEFORE_GT',answers_sha256=file_sha(path),
        registration_sha256=file_sha(PLAN),teacher=frozen,GT_open_attempts=0,
        worker_wall_seconds=time.perf_counter()-started,execution=execution,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),no_teacher_or_student_updates=True,goal_status='ACTIVE')
    atomic_json(OUT/'all72_sealed_before_GT.json',report)
    print(json.dumps(report),flush=True)


def evaluate():
    import numpy as np
    from flex_uda.a100.mechanism import metadata, OUT as BANK
    from scripts.a100_observation_evidence_transfer_diagnose import CANONICAL
    plan = registration(); sealed = json.loads((OUT/'all72_sealed_before_GT.json').read_text())
    assert sealed['answers_sha256'] == file_sha(OUT/'answers.jsonl')
    rows = [json.loads(s) for s in (OUT/'answers.jsonl').read_text().splitlines()]
    assert len(rows) == 72 and len({(r['case'],r['condition']) for r in rows}) == 72
    grouped = {(r['case'],r['condition']):r for r in rows}; reports=[]
    for pair in ('C1','C2','C4'):
        cases=[c for c in plan['cases'] if c['pair']==pair];first=cases[0]
        md=metadata(first['target']);video=first['video'];obs=md['videos'][video]['observations']
        pos={tuple(r):i for i,r in enumerate(obs)};qs=[q for q in md['episodes'] if q['video_id']==video];qi={q['expression_id']:i for i,q in enumerate(qs)}
        truth=np.load(BANK/'offline_GT'/CANONICAL[pair]/video/'labels.npy')>=0
        labels=[bool(truth[qi[c['key'][1]],pos[c['frame'],c['track']]]) for c in cases]
        methods={}
        for name in ('CROP','SCENE_CONTEXT'):
            rr=[grouped[c['index'],name] for c in cases];pred=[r['yes_probability_fp32']>=.5 for r in rr]
            methods[name]=dict(canonical_agreements=sum(a==b for a,b in zip(labels,pred)),
                TP=sum(a and b for a,b in zip(labels,pred)),FP=sum(not a and b for a,b in zip(labels,pred)),
                FN=sum(a and not b for a,b in zip(labels,pred)),TN=sum(not a and not b for a,b in zip(labels,pred)),
                actual_forward_seconds=sum(r['batch_forward_seconds'] for r in rr),visual_tokens=sum(r['visual_tokens'] for r in rr))
        reports.append(dict(pair=pair,target=first['target'],cases=12,canonical_positive_cases=sum(labels),
                            always_No_agreements=len(labels)-sum(labels),methods=methods))
    report=dict(status='COMPLETE_POSTSEAL_PAIRED_CONTEXT_TEACHER_DIAGNOSIS',cases=reports,
        teacher_measurements=72,teacher_unchanged=True,isolated_GT_only_after_full72seal=True,
        teacher_stratified_tiny_DEV_sample_not_pure_semantic_accuracy=True,
        more_visual_information_and_cost_than_CROP_not_same_compute=True,
        no_RMOT_HOTA_or_main_module_or_RSI_success_claim=True,goal_status='ACTIVE')
    atomic_json(OUT/'analysis.json',report);atomic_json(PUBLIC/'context_pair_teacher_pilot_analysis_20261125.json',report)
    print(json.dumps(report),flush=True)


def main():
    registration(); assert not (OUT/'answers.jsonl').exists()
    used=int(subprocess.check_output(['nvidia-smi','-i','2','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
    total=int(subprocess.check_output(['nvidia-smi','-i','2','--query-gpu=memory.total','--format=csv,noheader,nounits'],text=True).strip())
    owners=subprocess.check_output(['nvidia-smi','-i','2','--query-compute-apps=pid,process_name','--format=csv,noheader,nounits'],text=True).strip()
    assert total-used>=24576,(used,total)
    atomic_json(OUT/'GPU_preflight.json',dict(GPU=2,used_MiB=used,resident_processes=owners,all_preserved=True))
    proc=subprocess.Popen([sys.executable,'-u',str(Path(__file__)),'--worker'],cwd=ROOT,stdin=subprocess.DEVNULL,
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='2',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'))
    atomic_json(OUT/'actual_launch.json',dict(pid=proc.pid,GPU=2));print(json.dumps(dict(status='STARTED_PAIRED_SCENE_CONTEXT_TEACHER',GPU=2,pid=proc.pid)),flush=True)
    code=proc.wait()
    if code:raise RuntimeError(('Preserve partial teacher measurements and diagnose before resuming',code))
    subprocess.run([sys.executable,'-u',str(Path(__file__)),'--evaluate'],cwd=ROOT,check=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--worker',action='store_true');p.add_argument('--evaluate',action='store_true');args=p.parse_args()
    if args.prepare:prepare()
    elif args.worker:worker()
    elif args.evaluate:evaluate()
    else:main()
