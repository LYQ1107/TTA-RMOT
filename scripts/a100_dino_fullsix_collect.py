"""Shared image-only inputs for all six settings, with exact old-row reuse.

This prepares all74 physical videos. It neither promotes the first-video
primary past its failed strict gate nor trains/evaluates a new RMOT method.
"""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json, file_sha
from scripts import a100_dino_appearance_collect as original

OUT = original.OUT.parent/'query_independent_DINO_fullsix_20261125'
PLAN = OUT/'input_plan.json'
OLD_OUT = original.OUT
OLD_PLAN = original.PLAN
PUBLIC = original.PUBLIC


def prepare():
    import hashlib
    import numpy as np
    from flex_uda.a100.mechanism import generation_guard, metadata
    guard = generation_guard()
    if PLAN.exists():
        registration(); print('REUSE_REGISTERED_FULLSIX_APPEARANCE_PLAN', flush=True); return
    prior = original.registration()
    previous = {(u['target'], u['video']): u for u in prior['units']}
    seeds, units = [], []
    for target in ('Dance', 'KITTI-V2', 'SportsMOT'):
        for video in sorted(metadata(target)['videos']):
            _, obs, boxes = original.input_arrays(target, video)
            folder = OUT/target/video; folder.mkdir(parents=True, exist_ok=True)
            n = len(obs); rows = np.arange(n, dtype='int64')
            np.save(folder/'selected_rows.npy', rows)
            reused = 0
            if (target, video) in previous:
                old = original.completed(previous[target, video]); assert old is not None
                source = OLD_OUT/target/video
                old_rows = np.load(source/'selected_rows.npy')
                assert np.array_equal(np.load(source/'observations.npy'), obs[old_rows])
                value = np.lib.format.open_memmap(folder/'features.npy', mode='w+', dtype='float32', shape=(n, 768))
                value[:] = 0.; value[old_rows] = np.load(source/'features.npy', mmap_mode='r'); value.flush()
                valid = np.zeros(n, dtype=bool); done = np.zeros(n, dtype=bool)
                valid[old_rows] = np.load(source/'valid.npy'); done[old_rows] = True
                np.save(folder/'valid.npy', valid); np.save(folder/'processed.npy', done)
                assert np.array_equal(value[old_rows], np.load(source/'features.npy', mmap_mode='r'))
                reused = len(old_rows)
                seeds.append(dict(target=target, video=video, reused_rows=reused,
                                  source_receipt_sha256=file_sha(source/'receipt.json'),
                                  source_output_sha256=old['output_sha256']))
            unit = dict(target=target, video=video, rows=n, reused_rows=reused,
                        selected_rows_sha256=file_sha(folder/'selected_rows.npy'),
                        crop_inputs_sha256=hashlib.sha256(obs.tobytes()+boxes.tobytes()).hexdigest())
            units.append(unit)
    loads = [0, 0, 0, 0]
    for unit in sorted(units, key=lambda u: (-(u['rows']-u['reused_rows']), u['target'], u['video'])):
        role = min(range(4), key=lambda i: (loads[i], i)); unit['role'] = str(role)
        loads[role] += unit['rows']-unit['reused_rows']
    plan = dict(status='REGISTERED_UNLABELED_APPEARANCE_INPUTS_FOR_ALL_SIX_SETTINGS', units=units,
        physical_videos=len(units), rows=sum(u['rows'] for u in units), reused_rows=sum(u['reused_rows'] for u in units),
        role_pending_rows=loads, exact_reuse=seeds, roles=['0', '1', '2', '3'], batch_size=prior['batch_size'],
        source_sha256={**prior['source_sha256'], 'scripts/a100_dino_fullsix_collect.py': file_sha(Path(__file__))},
        external_source_sha256=prior['external_source_sha256'], weight_path=prior['weight_path'], weight_sha256=prior['weight_sha256'],
        parent_input_plan_sha256=file_sha(OLD_PLAN),
        extractor='Unchanged historical extractor; OUT and PLAN rebound only within new worker processes. Historical descriptor files never mutated.',
        feature_version='Old subset descriptors reused exactly; remaining rows get the same frozen encoder/preprocessing/batch-size. Not advertised as identical to a hypothetical one-pass full-video encoding.',
        physical_domains=['Dance', 'KITTI-V2', 'SportsMOT'],
        cross_domain_settings=['V2->Dance', 'Dance->KITTI-V2', 'V2->SportsMOT', 'Dance->SportsMOT', 'LaMOT->KITTI-V2', 'LaMOT->Dance'],
        same_features_for_every_future_ordinary_and_candidate=True,
        only_unlabeled_images_and_fixed_candidates_used_for_selection=True,
        prior_first_video_strict_screen_still_failed=True,
        no_new_main_acceptance_or_module_ablation_claim=True,
        new_Qwen_calls_training_predictions_or_GT_reads=0,
        all_data_DEV=True, seed=0, GPU2_fraction=.10, other_GPUs_fraction=.35,
        arrays_and_weights_not_to_Git=True, not_single_method_fresh_cost=True, goal_status='ACTIVE')
    assert len(units) == 74 and not guard
    atomic_json(PLAN, plan)
    atomic_json(PUBLIC/'query_independent_DINO_fullsix_plan_20261125.json',
                {k: v for k, v in plan.items() if k not in ('units', 'exact_reuse', 'external_source_sha256')})
    print(json.dumps(dict(status=plan['status'], physical_videos=74, rows=plan['rows'], exact_reused_rows=plan['reused_rows'], pending_by_GPU=loads)), flush=True)


def registration():
    plan = json.loads(PLAN.read_text())
    assert file_sha(OLD_PLAN) == plan['parent_input_plan_sha256']
    for name, sha in plan['source_sha256'].items(): assert file_sha(ROOT/name) == sha, name
    for name, sha in plan['external_source_sha256'].items(): assert file_sha(name) == sha, name
    assert file_sha(plan['weight_path']) == plan['weight_sha256']
    return plan


def worker(role):
    from flex_uda.active.deterministic_observation_runtime import configure_before_cuda
    execution = configure_before_cuda()
    import torch
    from flex_uda.active.frozen_appearance import load_model
    from flex_uda.a100.mechanism import generation_guard
    plan = registration(); guard = generation_guard()
    torch.set_num_threads(2); torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(.10 if role == '2' else .35)
    torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False); torch.backends.cuda.enable_math_sdp(True)
    original.OUT, original.PLAN = OUT, PLAN
    started = time.perf_counter(); model = load_model('cuda:0'); torch.cuda.synchronize()
    load_wall = time.perf_counter()-started
    reports = [original.extract(u, model, 'cuda:0', plan) for u in plan['units'] if u['role'] == role]
    assert not guard
    atomic_json(OUT/'supervisor'/(role+'_completion.json'), dict(status='COMPLETE_FULLSIX_FROZEN_APPEARANCE_ROLE',
        role=role, units=len(reports), rows=sum(r['rows'] for r in reports),
        exact_reused_rows=sum(r['cached_processed_rows_at_start'] for r in reports),
        real_forward_batches=sum(r['new_forward_batches_in_execution'] for r in reports),
        model_load_wall_seconds=load_wall, worker_wall_seconds=time.perf_counter()-started,
        GT_reads=0, Qwen_calls=0, execution=execution))


def drain(process, log):
    with log.open('a', buffering=1) as stream:
        for line in process.stdout: stream.write(line); print(line, end='', flush=True)


def main():
    plan = registration(); supervisor = OUT/'supervisor'; supervisor.mkdir(parents=True, exist_ok=True)
    pending = [role for role in plan['roles'] if not (supervisor/(role+'_completion.json')).exists()]
    for role in pending:
        used = int(subprocess.check_output(['nvidia-smi', '-i', role, '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True).strip())
        total = int(subprocess.check_output(['nvidia-smi', '-i', role, '--query-gpu=memory.total', '--format=csv,noheader,nounits'], text=True).strip())
        owners = subprocess.check_output(['nvidia-smi', '-i', role, '--query-compute-apps=pid,process_name', '--format=csv,noheader,nounits'], text=True).strip()
        if role == '2': assert total-used >= 8192, (role, used, total)
        else: assert not owners and used < 512, (role, used, owners)
        atomic_json(supervisor/(role+'_preflight.json'), dict(GPU=int(role), owners=owners, used_MiB=used, all_existing_processes_preserved=True))
    started = time.perf_counter(); jobs = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as readers:
        for role in pending:
            proc = subprocess.Popen([sys.executable, '-u', str(Path(__file__)), '--worker', role], cwd=ROOT,
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES=role, OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2'))
            future = readers.submit(drain, proc, supervisor/(role+'.log')); jobs.append((role, proc, future))
            atomic_json(supervisor/(role+'_launch.json'), dict(pid=proc.pid, GPU=int(role)))
            print(json.dumps(dict(status='STARTED_FULLSIX_SHARED_APPEARANCE_ROLE', GPU=int(role), pid=proc.pid)), flush=True)
        failures = []
        for role, proc, future in jobs:
            code = proc.wait(); future.result()
            if code: failures.append(dict(role=role, exit_code=code))
    if failures: raise RuntimeError(failures)
    original.OUT, original.PLAN = OUT, PLAN
    receipts = [original.completed(u) for u in plan['units']]; assert all(receipts)
    report = dict(status='COMPLETE_SHARED_APPEARANCE_ALL74_PHYSICAL_VIDEOS', units=74,
        rows=sum(r['rows'] for r in receipts), valid_rows=sum(r['valid_rows'] for r in receipts),
        invalid_rows=sum(r['invalid_rows'] for r in receipts),
        cached_rows_at_start=sum(r['cached_processed_rows_at_start'] for r in receipts),
        real_new_forward_batches=sum(r['new_forward_batches_in_execution'] for r in receipts),
        parent_wall_seconds=time.perf_counter()-started, input_plan_sha256=file_sha(PLAN),
        new_training_Qwen_questions_RMOT_predictions_or_GT_reads=0,
        same_common_inputs_for_all_future_ordinary=True, not_single_method_fresh_cost=True, goal_status='ACTIVE')
    atomic_json(OUT/'completion.json', report); atomic_json(PUBLIC/'query_independent_DINO_fullsix_completion_20261125.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--prepare', action='store_true'); parser.add_argument('--worker'); args = parser.parse_args()
    if args.prepare: prepare()
    elif args.worker: worker(args.worker)
    else: main()
