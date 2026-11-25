"""A bounded, label-free test of the teacher's specified-box grounding.

The known answers are rectangle geometry from the fixed candidate inputs,
not human semantic labels. Passing this test is not RMOT correctness.
No teacher adapter is trained or selected by this audit.
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

OUT = SIX/'teacher_grounding_audit_20261125'
PLAN = OUT/'registration.json'
PUBLIC = ROOT/'docs/a100_six_direction_20261122'
SOURCES = ('scripts/a100_teacher_grounding_audit.py',
           'flex_uda/teacher/yestrack_backend.py', 'flex_uda/teacher/decompose_inputs.py',
           'flex_uda/teacher/visual_input_builder.py', 'flex_uda/a100/mechanism.py',
           'flex_uda/a100/runtime.py', 'scripts/active_run.py',
           'configs/teacher/base_qwen2b.yaml', 'configs/teacher/prompts_v1.yaml',
           'flex_uda/active/deterministic_observation_runtime.py')


def rank(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def prepare():
    from PIL import Image
    from flex_uda.a100.mechanism import metadata, generation_guard
    from scripts.active_run import video_spec
    guard = generation_guard()
    assert not PLAN.exists(), 'Never replace an executed registration'
    cases = []
    for target in ('Dance', 'KITTI-V2', 'SportsMOT'):
        md = metadata(target); video = sorted(md['videos'])[0]
        spec = video_spec(target, video, md['videos'][video])
        first = spec['observations'][0]['frame_id']
        path = Path(spec['image_root'])/f"{first+spec['image_offset']:06d}{spec['extension']}"
        with Image.open(path) as image: width, height = image.size
        queries = sorted((q for q in md['episodes'] if q['video_id'] == video),
                         key=lambda q: rank(q['expression_id']))
        choices = {False: [], True: []}
        for ob in spec['observations']:
            x, y, w, h = ob['bbox_xywh']
            box = [max(0., x), max(0., y), min(float(width), x+w), min(float(height), y+h)]
            if box[2]-box[0] < 8 or box[3]-box[1] < 8: continue
            cx = (box[0]+box[2])/(2*width)
            if .35 < cx < .65: continue
            choices[cx < .5].append(dict(frame=int(ob['frame_id']), track=int(ob['track_id']),
                box=box, center_x=cx, expected_left=cx < .5))
        selected = []
        for side in (False, True):
            candidates = sorted(choices[side], key=lambda r: rank([target, video, r['frame'], r['track']]))
            assert len(candidates) >= 4
            used, taken = set(), []
            # Prefer different physical tracks; remaining slots use other frames
            # only if this side has fewer than four provided track IDs.
            for row in candidates:
                if row['track'] not in used:
                    taken.append(row); used.add(row['track'])
                if len(taken) == 4: break
            if len(taken) < 4:
                for row in candidates:
                    if row not in taken: taken.append(row)
                    if len(taken) == 4: break
            selected.extend(taken)
        for i, row in enumerate(selected):
            query = queries[i % len(queries)]
            cases.append(dict(target=target, video=video, query_id=query['expression_id'],
                query_text=query['query_text'], width=width, height=height, **row))
    plan = dict(status='REGISTERED_NO_GT_TEACHER_CANDIDATE_GROUNDING_AUDIT', cases=cases,
        source_sha256={n: file_sha(ROOT/n) for n in SOURCES}, neural_seed=0,
        hypothesis='A teacher can appear confident while failing to bind its answer to the specified candidate. First test known rectangle geometry before designing a learned visual-attention objective.',
        selection='First physical video/domain; SHA-order candidates, four left/four right outside center [.35,.65], prefer distinct tracks. Queries SHA-order, no teacher probabilities/semantic predictions/GT used.',
        known_geometry='Clipped candidate rectangle center. Left/right complements and horizontal mirror have exactly known geometric answers. No claim that these are human semantic target labels.',
        queries_per_case=['rectangle-left', 'rectangle-right', 'mirror-rectangle-left',
                          'mirror-rectangle-right', 'original-task-CROP', 'original-task-FRAME_BOX'],
        expected_real_teacher_inputs=144, expected_geometry_inputs=96, expected_original_task_inputs=48,
        semantic_scope='Original RMOT query is never mirrored, parsed or relabeled. Original task answers are preserved without treating them as truth.',
        failure_constraints=['No blur/no-image/occlusion is assigned a semantic No.',
            'Do not repeat the old five-variant sensitivity-based rejection method.',
            'Geometry grounding is a proxy, not causal attention attribution or RMOT accuracy.',
            'No training, checkpoint selection, target GT or HOTA is part of this audit.',
            'Any future KD control must receive all the same new teacher answers and geometry information.'],
        teacher='Original frozen Qwen3-VL-2B; no feasibility-pilot adapter loaded.',
        physical_GPU=2, process_memory_fraction=.20, real_batch_size=1,
        concurrent_DINO_and_existing_vision_service_preserved=True,
        not_single_method_fresh_cost=True, all_data_DEV=True, goal_status='ACTIVE')
    assert len(cases) == 24 and not guard
    atomic_json(PLAN, plan)
    atomic_json(PUBLIC/'teacher_grounding_audit_registration_20261125.json', plan)
    print(json.dumps(dict(status=plan['status'], cases=24, real_teacher_inputs=144)), flush=True)


def registration():
    plan = json.loads(PLAN.read_text())
    for name, sha in plan['source_sha256'].items(): assert file_sha(ROOT/name) == sha, name
    return plan


def worker():
    from flex_uda.active.deterministic_observation_runtime import configure_before_cuda
    execution = configure_before_cuda()
    import numpy as np
    import torch
    import yaml
    from PIL import Image, ImageDraw, ImageOps
    from flex_uda.a100.mechanism import metadata, generation_guard
    from flex_uda.a100.runtime import relocate
    from flex_uda.teacher.yestrack_backend import FrozenYesTrack
    from flex_uda.teacher.decompose_inputs import PairedViews, FRAME_PROMPT, images_sha
    from scripts.active_run import video_spec
    plan = registration(); guard = generation_guard()
    torch.set_num_threads(2); torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(plan['process_memory_fraction'])
    torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    started = time.perf_counter()
    config = yaml.safe_load((ROOT/'configs/teacher/base_qwen2b.yaml').read_text())
    config['model_path'] = str(ROOT/'assets/teacher_Qwen3-VL-2B-Instruct')
    config['yestrack_source'] = str(relocate(config['yestrack_source']))
    assert config['lora_path'] is None
    teacher = FrozenYesTrack(config)
    print(json.dumps(dict(status='ORIGINAL_QWEN_GROUNDING_AUDIT_LOADED',
        original_base_sha256=teacher.weight_sha, load_seconds=teacher.load_seconds)), flush=True)
    specs = [video_spec(t, v, metadata(t)['videos'][v]) for t, v in
             sorted({(c['target'], c['video']) for c in plan['cases']})]
    views = PairedViews(specs, teacher.upstream)
    prompts = yaml.safe_load((ROOT/'configs/teacher/prompts_v1.yaml').read_text())
    records = []; forward_wall = 0.
    output = OUT/'answers.jsonl'
    assert not output.exists(), 'Do not append duplicate completed teacher measurements'
    with output.open('w', buffering=1) as stream:
        for ci, case in enumerate(plan['cases']):
            key = case['target'], case['video']; frame = case['frame']; track = case['track']
            full = views.frame(key, frame)[0]
            assert full.size == (case['width'], case['height']), 'Variable image dimensions need a new fixed protocol'
            ratio = 1024/max(full.size)
            image = full.resize((round(full.width*ratio), round(full.height*ratio)), Image.Resampling.BICUBIC)
            ImageDraw.Draw(image).rectangle([v*ratio for v in case['box']], outline=(255, 0, 0), width=4)
            mirror = ImageOps.mirror(image)
            tests = []
            for mirrored, im in ((False, image), (True, mirror)):
                for half in ('left', 'right'):
                    prompt = f'Consider the red rectangular outline in this image. Is the center of this rectangle in the {half} half of the image? Answer Yes or No.'
                    expected = bool(case['expected_left']) ^ mirrored ^ (half == 'right')
                    tests.append((f'GEOMETRY_{"MIRROR" if mirrored else "ORIGINAL"}_{half.upper()}', im, prompt, expected))
            ob = views.observations[key, frame, track]
            crop = views.patch(key, ob)
            coordinate = f'{(ob.bbox_xyxy[0]+ob.bbox_xyxy[2])/2/full.width:.3f} {(ob.bbox_xyxy[1]+ob.bbox_xyxy[3])/2/full.height:.3f}'
            tests.extend([('TASK_CROP', crop, prompts['official_single'].format(coord=coordinate, sentence=case['query_text']), None),
                          ('TASK_FRAME_BOX', image, FRAME_PROMPT.format(expression=case['query_text']), None)])
            for condition, im, prompt, expected in tests:
                response = teacher.forward([[im]], [prompt])[0]
                forward_wall += response['batch_forward_seconds']
                record = dict(case=ci, target=case['target'], video=case['video'], frame=frame, track=track,
                    condition=condition, prompt=prompt, automatic_rectangle_truth=expected,
                    visual_sha256=images_sha([im]), **response)
                records.append(record); stream.write(json.dumps(record)+'\n')
            crop.close(); mirror.close(); image.close()
            print(json.dumps(dict(status='REAL_TEACHER_GROUNDING_CASE_COMPLETE', completed=ci+1, cases=24,
                real_teacher_inputs=len(records))), flush=True)
    assert len(records) == plan['expected_real_teacher_inputs'] and not guard
    frozen = teacher.seal_frozen()
    geometry = [r for r in records if r['automatic_rectangle_truth'] is not None]
    groups = {}
    for target in ('Dance', 'KITTI-V2', 'SportsMOT'):
        values = [r for r in geometry if r['target'] == target]
        groups[target] = dict(inputs=len(values), errors=sum((r['yes_probability_fp32'] >= .5) != r['automatic_rectangle_truth'] for r in values),
            mean_proper_binary_CE=float(np.mean([-np.log(np.clip(r['yes_probability_fp32'] if r['automatic_rectangle_truth'] else 1-r['yes_probability_fp32'], 1e-8, 1)) for r in values])))
    report = dict(status='COMPLETE_LABEL_FREE_SPECIFIED_RECTANGLE_GROUNDING_AUDIT', cases=24,
        real_inputs=len(records), cache_hits=0, geometry_by_domain=groups,
        original_task_inputs=48, original_task_semantic_quality_unknown=True,
        exact_answers_sha256=file_sha(output), registration_sha256=file_sha(PLAN),
        teacher=frozen, synchronized_forward_seconds=forward_wall,
        worker_wall_seconds=time.perf_counter()-started, execution=execution,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(), GT_open_attempts=0,
        no_adapter_training_or_new_RMOT_predictions=True, not_single_method_fresh_cost=True,
        geometric_proxy_not_semantic_accuracy_or_causal_attention=True,
        no_automatic_main_promotion_or_accuracy_claim=True, goal_status='ACTIVE')
    atomic_json(OUT/'completion.json', report)
    atomic_json(PUBLIC/'teacher_grounding_audit_completion_20261125.json', report)
    print(json.dumps(report), flush=True)


def main():
    registration(); OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT/'answers.jsonl').exists(), 'Preserve previous execution; diagnose before a resume design'
    used = int(subprocess.check_output(['nvidia-smi', '-i', '2', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True).strip())
    total = int(subprocess.check_output(['nvidia-smi', '-i', '2', '--query-gpu=memory.total', '--format=csv,noheader,nounits'], text=True).strip())
    owners = subprocess.check_output(['nvidia-smi', '-i', '2', '--query-compute-apps=pid,process_name', '--format=csv,noheader,nounits'], text=True).strip()
    assert total-used >= 16384, (used, total)
    atomic_json(OUT/'GPU_preflight.json', dict(physical_GPU=2, used_MiB=used, resident_processes=owners,
        all_existing_processes_preserved=True, process_memory_fraction=.20))
    process = subprocess.Popen([sys.executable, '-u', str(Path(__file__)), '--worker'], cwd=ROOT,
        stdin=subprocess.DEVNULL, env=dict(os.environ, CUDA_VISIBLE_DEVICES='2', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2'))
    atomic_json(OUT/'actual_launch.json', dict(pid=process.pid, physical_GPU=2))
    print(json.dumps(dict(status='STARTED_REAL_TEACHER_GROUNDING_AUDIT', GPU=2, pid=process.pid)), flush=True)
    code = process.wait()
    if code: raise RuntimeError(('Teacher grounding audit failed; preserve all inputs and partial answers', code))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--prepare', action='store_true'); parser.add_argument('--worker', action='store_true'); args = parser.parse_args()
    if args.prepare: prepare()
    elif args.worker: worker()
    else: main()
