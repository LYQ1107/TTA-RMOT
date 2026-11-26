"""A preregistered teacher-only ordinary-head capacity sufficiency probe.

No RMOT quality output, target annotation or post-hoc method choice enters.
Only a previously unsuccessful causal prefix is optimized; current studies
and their predictions remain immutable.
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
from scripts.a100_six_direction import PUBLIC
from scripts.a100_object_reference_fit_complete import OUT as FIT_ROOT

OUT = FIT_ROOT/'supervisor/capacity_probe'
STATE = OUT/'supervisor/state.json'
SUCCESS = 'TEACHER_ONLY_ORDINARY_CAPACITY_PROBE_COMPLETE'
REGISTRATION = PUBLIC/'frame_kd_capacity_probe_registration.json'


def run():
    import numpy as np
    import torch
    from flex_uda.a100.mechanism import generation_guard
    from flex_uda.active.fit_capacity import continue_capacity
    plan = json.loads(REGISTRATION.read_text()); guard = generation_guard()
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    case = plan['case']; root = FIT_ROOT/case['pair']/f"seed{case['seed']}"
    stream = torch.load(root/'stream.pt', map_location='cpu', weights_only=False)
    prefix = case['prefix_videos']; video = stream['receipts'][prefix-1]['video']
    assert video == case['video'] and prefix <= stream['next_video']
    old = stream['receipts'][prefix-1]['fits']['FRAME_COMPLETE']
    assert not old['teacher_fit_within_0_005']
    checkpoint = torch.load(root/'videos'/video/'fit_complete_head.pt',
        map_location='cpu', weights_only=False)
    x = np.concatenate(stream['bx'][:prefix]); y = np.concatenate(stream['by'][:prefix])
    fitted, receipt = continue_capacity(x, y, checkpoint, seed=case['seed'],
        widths=tuple(plan['widths']), blocks=plan['blocks_per_width'])
    assert not guard
    torch.save(fitted, OUT/'ordinary_capacity_head.pt')
    report = dict(status=SUCCESS, case=case, **receipt,
        actual_physical_GPU=os.environ.get('CUDA_VISIBLE_DEVICES'),
        manual_target_GT_open_attempts=0, quality_predictions_read_or_generated=0,
        new_teacher_encoder_calls=0, existing_study_weights_or_outputs_changed=False,
        gate_changed=False, all_data_development=True,
        interpretation='Optimization/capacity diagnosis for a stronger ordinary comparator, not an RMOT innovation or quality acceptance.',
        next='A successful probe must be applied causally to every still-failing prefix in a separately sealed quality supplement; this single head is not a complete comparator.',
        goal_status='ACTIVE')
    atomic_json(OUT/'completion.json', report)
    (PUBLIC/'frame_kd_capacity_probe.json').write_text(
        json.dumps(report, ensure_ascii=False, separators=(',', ':'))+'\n')
    print(json.dumps(report, ensure_ascii=False), flush=True)


def main():
    plan = json.loads(REGISTRATION.read_text())
    for name, sha in plan['source_sha256'].items():
        assert file_sha(ROOT/name) == sha, name
    STATE.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(STATE, dict(status='AWAITING_FREE_GPU_0_TO_3_CAPACITY_PROBE',
        pid=os.getpid(), completed=0, goal_status='ACTIVE'))
    while True:
        text = subprocess.check_output(['nvidia-smi', '-i', '0,1,2,3',
            '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'], text=True)
        memory = {int(s.split(',')[0]):int(s.split(',')[1]) for s in text.strip().splitlines()}
        free = [g for g in (0, 1, 2, 3) if memory[g] < 512]
        if free:
            break
        time.sleep(15)
    gpu = free[0]
    with (OUT/'supervisor/worker.log').open('a', buffering=1) as handle:
        proc = subprocess.Popen([sys.executable, '-u', str(Path(__file__)), '--worker'],
            cwd=ROOT, stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4'))
        atomic_json(STATE, dict(status='RUNNING_TEACHER_ONLY_CAPACITY_PROBE',
            pid=os.getpid(), worker_pid=proc.pid, physical_GPU=gpu, completed=0,
            allocation_memory_MiB=memory, goal_status='ACTIVE'))
        code = proc.wait()
    if code or not (OUT/'completion.json').is_file():
        raise RuntimeError(dict(exit_code=code, log=(OUT/'supervisor/worker.log').read_text()[-6000:]))
    atomic_json(STATE, dict(status=SUCCESS, pid=os.getpid(), completed=1,
        physical_GPU=gpu, goal_status='ACTIVE'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    if args.worker:
        run()
    else:
        try:
            main()
        except Exception as error:
            atomic_json(STATE, dict(status='FAILED_TEACHER_ONLY_CAPACITY_PROBE',
                pid=os.getpid(), error=repr(error), goal_status='ACTIVE'))
            raise
