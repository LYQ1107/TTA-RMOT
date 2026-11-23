"""Isolate CUDA reduction repeatability using the existing teacher-only risk.

This is a diagnostic on the first C2 prefix, not a replacement RMOT method.
Two default and two deterministic FULL fits share the seed, GPU, inputs and
optimizer helper. No GT, teacher call, official prediction or HOTA is used.
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
import numpy as np
from flex_tta.runtime.pf_receipts import atomic_json, file_sha
from flex_uda.a100.mechanism import PAIRS, metadata, generation_guard
from scripts.a100_conditional_standard_kd import STATE as PREVIOUS_STATE, use_verified_cache
from scripts.a100_six_direction import OUT as SIX, PUBLIC

OUT = SIX/'conditional_repeatability_20261123'
STATE = OUT/'supervisor/state.json'
REGISTRATION = PUBLIC/'conditional_repeatability_registration_v1.json'
SUCCESS = 'PASS_CONDITIONAL_REPEATABILITY_DIAGNOSIS_RESEARCH_GOAL_ACTIVE'
CASES = ('DEFAULT_0', 'DEFAULT_1', 'DETERMINISTIC_0', 'DETERMINISTIC_1')


def check_registration():
    plan = json.loads(REGISTRATION.read_text())
    for name, sha in plan['source_sha256'].items():
        assert file_sha(ROOT/name) == sha, name


def kernel_repeats(data, deterministic):
    import torch
    import torch.nn.functional as F
    from flex_uda.active.conditional_observation import weighted_bernoulli_loss
    from flex_uda.active.query_profile import supported_query_profile_kl
    device = 'cuda:0'
    yy = torch.as_tensor(data['target'], device=device)
    raw = torch.as_tensor(data['raw_target'], device=device)
    ww = torch.as_tensor(data['balanced_weight'], device=device)
    groups = torch.as_tensor(data['group'], device=device)
    rng = torch.Generator(device=device).manual_seed(2718)
    base = torch.randn(len(yy), device=device, generator=rng)
    result = {}
    for name, profile in (('POINT_ONLY', False), ('POINT_PLUS_PROFILE', True)):
        reference = None
        max_error = 0.
        changed = 0
        losses = []
        for _ in range(32):
            z = base.clone().requires_grad_()
            loss = weighted_bernoulli_loss(z, yy, ww)
            if profile:
                loss = loss+supported_query_profile_kl(F.logsigmoid(z), yy, raw, groups, ww)
            loss.backward()
            gradient = z.grad.detach().cpu().numpy()
            losses.append(float(loss.detach()))
            if reference is None:
                reference = gradient.copy()
            else:
                changed += int(np.count_nonzero(gradient != reference))
                max_error = max(max_error, float(np.max(np.abs(gradient-reference))))
        result[name] = dict(repeats=32, deterministic_algorithms=deterministic,
            changed_gradient_elements_across31_comparisons=changed, maximum_absolute_gradient_difference=max_error,
            loss_range=[min(losses), max(losses)])
    return result


def run(case):
    import torch
    from flex_uda.active.observation_learning_inputs import ObservationVideo, prefix_data, DIMENSION
    from flex_uda.active.observation_learning_fit import fit, forward
    check_registration()
    guard = generation_guard()
    cache = use_verified_cache()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_device(0)
    deterministic = case.startswith('DETERMINISTIC')
    torch.use_deterministic_algorithms(deterministic, warn_only=False)
    started = time.perf_counter()
    pair = 'C2'
    target = PAIRS[pair][1]
    video = sorted(metadata(target)['videos'])[0]
    assert video == '0005'
    root = OUT/case
    root.mkdir(parents=True, exist_ok=True)
    inputs = ObservationVideo(pair, video)
    flat = inputs.raw_bags[:, inputs.present].reshape(-1, DIMENSION).astype(float)
    mean = flat.sum(0)/len(flat)
    std = np.maximum(np.sqrt(np.maximum(np.square(flat).sum(0)/len(flat)-mean**2, 0)), 1e-3)
    data = prefix_data([inputs.supervision('DESIGN')], mean, std)
    kernel = kernel_repeats(data, deterministic)
    model, early, checkpoint, receipt = fit(data, 'FULL', seed=0, mean=mean, std=std)
    assert receipt['target_fit_gate_pass'], receipt['metrics']
    nq, no = len(inputs.queries), len(inputs.obs)
    sample = np.unique(np.linspace(0, nq*no-1, min(8192, nq*no), dtype=int))
    normalized = ((inputs.raw_bags-mean)/std).astype('float32')
    normalized[:, ~inputs.present] = 0
    predictions = []
    with torch.no_grad():
        for a in range(0, len(sample), 2048):
            ids = sample[a:a+2048]
            qi, rows = ids//no, ids % no
            ti = inputs.columns[rows]
            b = torch.as_tensor(normalized[qi, ti], device='cuda:0')
            m = torch.as_tensor(inputs.present[ti], device='cuda:0')
            c = torch.as_tensor(((inputs.dense_rows(qi, rows)-mean)/std).astype('float32'), device='cuda:0')
            predictions.append(forward(model, 'conditional', b, m, c).cpu().numpy())
    values = np.concatenate(predictions)
    np.savez_compressed(root/'diagnostic_unlabelled_sample_logits.npz', flat_indices=sample, logits=values)
    assert not guard
    atomic_json(root/'completion.json', dict(status='COMPLETE_TEACHER_ONLY_REPEATABILITY_FIT', case=case,
        pair=pair, video=video, seed=0, deterministic_algorithms=deterministic,
        CUBLAS_WORKSPACE_CONFIG=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        kernel_gradient_repeats=kernel, fit=receipt, sample_size=len(sample),
        no_target_GT_or_current_study_quality_read=True, GT_open_attempts=0,
        no_new_teacher_or_backbone_calls=True, no_official_predictions_or_HOTA=True,
        cache_aggregate_sha256=cache['aggregate_sha256'], wall_seconds=time.perf_counter()-started,
        not_a_method_quality_or_cold_cost_result=True, goal_status='ACTIVE'))


def analyze():
    result = {}
    for mode in ('DEFAULT', 'DETERMINISTIC'):
        left, right = OUT/(mode+'_0'), OUT/(mode+'_1')
        with np.load(left/'diagnostic_unlabelled_sample_logits.npz') as a, np.load(right/'diagnostic_unlabelled_sample_logits.npz') as b:
            assert np.array_equal(a['flat_indices'], b['flat_indices'])
            x, y = a['logits'], b['logits']
            result[mode] = dict(sample_size=len(x), exact_same_logits=bool(np.array_equal(x, y)),
                maximum_absolute_logit_difference=float(np.max(np.abs(x-y))),
                changed_raw_decisions=int(((x>0)!=(y>0)).sum()))
    receipts = {case:json.loads((OUT/case/'completion.json').read_text()) for case in CASES}
    assert all(r['GT_open_attempts'] == 0 and r['fit']['target_fit_gate_pass'] for r in receipts.values())
    report = dict(status='CONTROLLED_CONDITIONAL_REPEATABILITY_DIAGNOSIS_COMPLETE', comparisons=result,
        cases=receipts, same_physical_GPU_for_all_four_cases=True,
        deterministic_repeatability_observed=result['DETERMINISTIC']['exact_same_logits'],
        no_HOTA_or_teacher_accuracy_inference=True, no_method_or_primary_promotion=True,
        scope='One real first-video prefix; cannot certify every training path or explain all prior drift',
        goal_status='ACTIVE')
    atomic_json(OUT/'analysis.json', report)
    atomic_json(PUBLIC/'conditional_repeatability_analysis.json', report)


def main():
    check_registration()
    STATE.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(STATE, dict(status='AWAITING_STANDARD_KD_GPU_STAGE_RELEASE', pid=os.getpid(), completed=0, goal_status='ACTIVE'))
    while True:
        previous = json.loads(PREVIOUS_STATE.read_text()) if PREVIOUS_STATE.is_file() else {}
        if previous.get('status', '').startswith('FAILED_'):
            raise RuntimeError('Resolve registered standard KD study first')
        if previous.get('completed') == 18 and not previous.get('active'):
            break
        time.sleep(15)
    gpu = None
    while gpu is None:
        output = subprocess.check_output(['nvidia-smi', '-i', '0,1,2,3',
            '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'], text=True)
        memory = {int(r.split(',')[0]):int(r.split(',')[1]) for r in output.strip().splitlines()}
        gpu = next((g for g in (0, 1, 2, 3) if memory[g] < 512), None)
        if gpu is None:
            time.sleep(15)
    for i, case in enumerate(CASES):
        if (OUT/case/'completion.json').is_file():
            continue
        while True:
            used = int(subprocess.check_output(['nvidia-smi', '-i', str(gpu),
                '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True).strip())
            if used < 512:
                break
            time.sleep(15)
        log = STATE.parent/(case+'.log')
        with log.open('a', buffering=1) as handle:
            proc = subprocess.Popen([sys.executable, '-u', str(Path(__file__)), '--case', case], cwd=ROOT,
                stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), CUBLAS_WORKSPACE_CONFIG=':4096:8',
                    OPENBLAS_NUM_THREADS='4', OMP_NUM_THREADS='4'))
            atomic_json(STATE, dict(status='RUNNING_CONTROLLED_REPEATABILITY_NO_GT', pid=os.getpid(), completed=i,
                case=case, physical_GPU=gpu, worker_pid=proc.pid, log=str(log), goal_status='ACTIVE'))
            code = proc.wait()
        if code or not (OUT/case/'completion.json').is_file():
            raise RuntimeError(json.dumps(dict(case=case, exit_code=code, last_log=log.read_text()[-8000:])))
    analyze()
    atomic_json(STATE, dict(status=SUCCESS, pid=os.getpid(), completed=4, physical_GPU=gpu, goal_status='ACTIVE'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', choices=CASES)
    args = parser.parse_args()
    if args.case:
        run(args.case)
    else:
        try:
            main()
        except Exception as error:
            old = json.loads(STATE.read_text()) if STATE.is_file() else {}
            old.update(status='FAILED_REPEATABILITY_DIAGNOSIS_REQUIRES_ROOT_CAUSE', error=repr(error), goal_status='ACTIVE')
            atomic_json(STATE, old)
            raise
