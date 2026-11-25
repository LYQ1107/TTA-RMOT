"""One real-image adapter update: feasibility, not an RMOT quality study.

Targets come from the frozen model on a resampled version of the same ROI.
Choosing a nonzero resampling difference is only a gradient-wiring probe;
the resulting adapter is isolated and must not seed a research main model.
"""
import argparse
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

OUT = SIX/'trainable_Qwen_feasibility_20261125'
PUBLIC = ROOT/'docs/a100_six_direction_20261122'
PLAN = OUT/'registration.json'
SOURCES = ('flex_uda/active/trainable_qwen.py', 'scripts/a100_trainable_qwen_pilot.py',
           'flex_uda/teacher/yestrack_backend.py', 'flex_uda/teacher/decompose_inputs.py',
           'flex_uda/teacher/visual_input_builder.py', 'flex_uda/a100/runtime.py',
           'flex_uda/active/deterministic_observation_runtime.py', 'configs/teacher/base_qwen2b.yaml',
           'configs/teacher/prompts_v1.yaml')


def prepare():
    from flex_uda.a100.mechanism import generation_guard
    guard = generation_guard()
    source = SIX/'joint_teacher_noise_primary_20261125/C1'
    keys = json.loads((source/'row_keys.json').read_text())['keys']
    key = next(k for k in keys if k[0] == 'dancetrack0004')
    expected = json.loads((source/'completion.json').read_text())['teacher_weight_sha256']
    plan = dict(status='REGISTERED_ONE_REAL_IMAGE_QWEN_ADAPTER_FEASIBILITY', pair='C1', key=key,
        expected_original_base_sha256=expected, source_sha256={name: file_sha(ROOT/name) for name in SOURCES},
        adapter='Only language-decoder q_proj/v_proj LoRA r8/alpha16/dropout0; base and vision frozen.',
        target='Frozen original-model Yes probability on the downsample224or160->restore320 ROI with larger nonzero response difference. Engineering signal only, not a semantic correctness criterion.',
        fitting='Exactly one AdamW lr1e-5 step, original-view soft BCE plus.1 proper binary KL to frozen original. Seed0, no GT or HOTA.',
        outputs='Isolated pilot adapter only; do not reuse for a future research main or claim RSI/HOTA improvement.',
        physical_GPU=0, existing_processes_preserved=True, new_RMOT_predictions=0, goal_status='ACTIVE')
    assert not guard
    atomic_json(PLAN, plan); atomic_json(PUBLIC/'trainable_Qwen_feasibility_registration_20261125.json', plan)
    print(json.dumps(dict(status=plan['status'], key=key)), flush=True)


def run():
    from flex_uda.active.deterministic_observation_runtime import configure_before_cuda
    execution = configure_before_cuda()
    import torch
    import yaml
    from PIL import Image
    from flex_uda.a100.mechanism import metadata, generation_guard
    from flex_uda.a100.runtime import relocate
    from flex_uda.teacher.yestrack_backend import upstream_functions
    from flex_uda.teacher.decompose_inputs import PairedViews, images_sha
    from flex_uda.active.trainable_qwen import TrainableQwen
    from scripts.active_run import video_spec
    plan = json.loads(PLAN.read_text())
    for name, sha in plan['source_sha256'].items(): assert file_sha(ROOT/name) == sha, name
    guard = generation_guard(); torch.set_num_threads(2)
    torch.cuda.set_device(0); torch.cuda.set_per_process_memory_fraction(.65)
    torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False); torch.backends.cuda.enable_math_sdp(True)
    started = time.perf_counter()
    model = TrainableQwen(ROOT)
    assert model.initial_base_sha256 == plan['expected_original_base_sha256']
    print(json.dumps(dict(status='TRAINABLE_QWEN_ORIGINAL_BASE_LOADED', base_sha256=model.initial_base_sha256,
                          trainable_parameters=sum(p.numel() for _, p in model.trainable))), flush=True)
    video, qid, track, frame = plan['key']; md = metadata('Dance'); bank = md['videos'][video]
    query = next(q for q in md['episodes'] if q['video_id'] == video and q['expression_id'] == qid)
    config = yaml.safe_load((ROOT/'configs/teacher/base_qwen2b.yaml').read_text())
    views = PairedViews([video_spec('Dance', video, bank)], upstream_functions(relocate(config['yestrack_source'])))
    observation = views.observations[('Dance', video), int(frame), int(track)]
    full = views.frame(('Dance', video), int(frame))[0]
    x1, y1, x2, y2 = observation.bbox_xyxy
    coordinate = f'{(x1+x2)/2/full.width:.3f} {(y1+y2)/2/full.height:.3f}'
    prompt = yaml.safe_load((ROOT/'configs/teacher/prompts_v1.yaml').read_text())['official_single'].format(coord=coordinate, sentence=query['query_text'])
    original = views.patch(('Dance', video), observation)
    original_inputs = model.inputs([[original]], [prompt])
    with torch.no_grad(): reference_logits = model.logits(original_inputs, reference=True)
    initial = float(model.probability(original_inputs))
    reference = float(reference_logits.softmax(-1)[:, 0])
    assert initial == reference, 'Zero-initialized adapter must exactly preserve this input prediction'
    alternatives = []
    for side in (224, 160):
        down = original.resize((side, side), Image.Resampling.BICUBIC)
        resampled = down.resize(original.size, Image.Resampling.BICUBIC)
        value = float(model.probability(model.inputs([[resampled]], [prompt]), reference=True))
        alternatives.append(dict(downsample_side=side, frozen_probability=value, visual_sha256=images_sha([resampled])))
        down.close(); resampled.close()
    selected = max(alternatives, key=lambda row: abs(row['frozen_probability']-reference))
    assert abs(selected['frozen_probability']-reference) > 1e-8, 'No view-consistency gradient signal; do not fabricate a correctness label'
    initial_adapter = {name: p.detach().cpu().clone() for name, p in model.trainable}
    step_started = time.perf_counter()
    result = model.soft_evidence_step(original_inputs, [selected['frozen_probability']], reference_logits)
    step_wall = time.perf_counter()-step_started
    changed = max(float((p.detach().cpu()-initial_adapter[name]).abs().max()) for name, p in model.trainable)
    assert changed > 0
    final = float(model.probability(original_inputs))
    seal = model.seal(); assert not guard
    # These are genuine Qwen adapter parameters, isolated from all main cohorts.
    model.model.save_pretrained(OUT/'isolated_adapter_not_for_main')
    report = dict(status='PASS_ONE_REAL_IMAGE_QWEN_ADAPTER_FORWARD_AND_BACKWARD', pair='C1',
        key=plan['key'], initial_probability=initial, original_reference_probability=reference,
        resampling_probes=alternatives, selected_external_target=selected, final_probability=final,
        adapter_maximum_parameter_change=changed, step=result, adapter_step_wall_seconds=step_wall,
        model_seal=seal, execution=execution, worker_wall_seconds=time.perf_counter()-started,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(), GT_open_attempts=0,
        image_source='One actual Dance candidate and original task query; resampling only, no synthetic semantic answer.',
        adapter_checkpoint_not_to_Git_or_future_main=True, new_RMOT_predictions_or_HOTA=0,
        not_evidence_of_teacher_accuracy_student_RMOT_or_RSI_improvement=True,
        source_PCD_student_training_not_included=True, goal_status='ACTIVE')
    atomic_json(OUT/'completion.json', report); atomic_json(PUBLIC/'trainable_Qwen_feasibility_20261125.json', report)
    print(json.dumps(report), flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    residents = subprocess.check_output(['nvidia-smi', '-i', '0', '--query-compute-apps=pid,process_name', '--format=csv,noheader,nounits'], text=True).strip()
    used = int(subprocess.check_output(['nvidia-smi', '-i', '0', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True).strip())
    assert not residents and used < 512, (residents, used)
    atomic_json(OUT/'GPU_preflight.json', dict(physical_GPU=0, residents=residents, used_MiB=used, existing_processes_preserved=True))
    process = subprocess.Popen([sys.executable, '-u', str(Path(__file__)), '--worker'], cwd=ROOT,
        stdin=subprocess.DEVNULL, env=dict(os.environ, CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2'))
    atomic_json(OUT/'actual_launch.json', dict(pid=process.pid, physical_GPU=0))
    print(json.dumps(dict(status='STARTED_REAL_QWEN_ADAPTER_FEASIBILITY', pid=process.pid, GPU=0)), flush=True)
    code = process.wait()
    if code: raise RuntimeError(('Qwen adapter pilot failed; preserve all evidence', code))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--prepare', action='store_true'); parser.add_argument('--worker', action='store_true'); args = parser.parse_args()
    if args.prepare: prepare()
    elif args.worker: run()
    else: main()
