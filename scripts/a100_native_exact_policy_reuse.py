"""Reuse an existing replay fit only when its numerical inputs are identical.

Execution aid for the already registered native726 study, not a new method.
The replay solver receives paid rows, their training schedule, a shared visual
prefix, normalization, a previous head and a mathematical recipe.  Policy names
only choose those inputs; the pinned fitting/model code never uses that name
numerically.  Equal inputs, equal warm weights and seed0 therefore define the
same fit, including the saved early head.  Both logical arms remain in the study.

This CPU producer waits for actual source heads, writes proven-equivalent target
checkpoints, and lets the original GPU worker produce predictions.  Reused fit
records are explicitly marked as shared computation, never independent fits.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
for option in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[option] = '2'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from flex_tta.runtime.pf_receipts import atomic_json, file_sha
from scripts import a100_native_binary_six_setting as native

PLAN_SHA = '9a19d1643c9c6a51ad92f723f558edcbbc496cafcb05e1e2a2f7161186ad28ec'
MAIN = ('FULL_CONDITIONAL_REFERENCE_LEARNED', 'FULL_CONDITIONAL_REFERENCE_FROZEN')
ORDER = [*MAIN, *[name for name in native.RECIPES if name not in MAIN]]


def same_array(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b)


def same_model(a, b):
    return a.keys() == b.keys() and all(
        v.dtype == b[k].dtype and v.shape == b[k].shape and torch.equal(v, b[k])
        for k, v in a.items())


def same_warm(a, b):
    return (a['width'] == b['width'] and same_array(a['mean'], b['mean'])
            and same_array(a['std'], b['std']) and same_model(a['model'], b['model']))


def math_recipe(recipe):
    return {k: v for k, v in recipe.items() if k != 'policy'}


def copied_head(source, target, binding):
    """Only policy metadata changes; all learned/early tensors stay exact."""
    result = copy.deepcopy({k: source[k] for k in ('checkpoint', 'early', 'report')})
    recipe = native.RECIPES[target]
    result['checkpoint']['recipe'] = copy.deepcopy(recipe)
    for i, fit in enumerate(result['report']['fits']):
        assert math_recipe(fit['recipe']) == math_recipe(recipe)
        fit['recipe'] = copy.deepcopy(recipe)
        fit['exact_policy_input_reuse'] = dict(source_arm=binding['source_arm'],
            source_fit_index=i, new_independent_fit=False,
            wall_seconds_belongs_to_source_fit=True)
    result['checkpoint']['receipt'] = result['report']['fits'][-1]
    result['report']['exact_policy_input_reuse'] = binding
    result['report']['fits_are_shared_computation_not_independent_runs'] = True
    result['exact_policy_input_binding'] = binding
    return result


def context(pair):
    root = native.OUT / pair / 'seed0'
    if (root / 'completion.json').is_file():
        return dict(status='PAIR_ORIGINAL_OUTPUT_COMPLETE', pair=pair)
    progress = json.loads((root / 'progress.json').read_text())
    video = progress['video']
    local = root / 'videos' / video
    closed = [local / 'policies' / p / 'closed.pt' for p in ('LEARNED', 'FROZEN')]
    if not all(p.is_file() for p in closed):
        return None
    stream = root / 'stream.pt'
    stream_sha = file_sha(stream)
    saved = torch.load(stream, map_location='cpu', weights_only=False)
    if progress['completed_videos'] != saved['next_video']:
        return None
    videos = [r['video'] for r in saved['receipts']] + [video]
    assert len(videos) == saved['next_video'] + 1
    hashes = {str(stream): stream_sha}
    for name in videos:
        folders = [root / 'videos' / name / 'policies' / p for p in ('LEARNED', 'FROZEN')]
        for filename, key in (('paid.json', 'paid'), ('sequence.json', 'sequence')):
            paths = [folder / filename for folder in folders]
            rows = [json.loads(p.read_text())[key] for p in paths]
            if rows[0] != rows[1]:
                return dict(status='NONIDENTICAL_POLICY_PREFIX_KEEP_ORIGINAL_FITS',
                    pair=pair, video=video, first_different_video=name, field=key)
            for p in paths:
                hashes[str(p)] = file_sha(p)
    heads = [torch.load(p, map_location='cpu', weights_only=False)['checkpoint'] for p in closed]
    assert same_array(heads[0]['mean'], heads[1]['mean'])
    assert same_array(heads[0]['std'], heads[1]['std'])
    for p in closed:
        hashes[str(p)] = file_sha(p)
    if file_sha(stream) != stream_sha:
        return None
    return dict(status='EXACT_POLICY_INPUT_PREFIX_CERTIFIED', pair=pair, video=video,
        root=root, local=local, saved=saved, stream=stream, stream_sha=stream_sha,
        mean=heads[1]['mean'], std=heads[1]['std'], hashes=hashes,
        prefix_videos=videos, finished=set(), refused=set())


def produce_ready(ctx, excluded):
    root, local = ctx['root'], ctx['local']
    installed = []
    for target in ORDER:
        if not target.endswith('_FROZEN') or target in MAIN or target in excluded:
            continue
        destination = local / 'heads' / (target + '.pt')
        if destination.is_file() or target in ctx['finished'] or target in ctx['refused']:
            continue
        source = target[:-len('FROZEN')] + 'LEARNED'
        source_path = local / 'heads' / (source + '.pt')
        if not source_path.is_file():
            continue
        progress = json.loads((root / 'progress.json').read_text())
        if progress['video'] != ctx['video'] or progress['completed_arms'] >= ORDER.index(target):
            ctx['finished'].add(target)
            continue
        previous = ctx['saved']['previous']
        assert math_recipe(native.RECIPES[source]) == math_recipe(native.RECIPES[target])
        assert previous[source]['recipe'] == native.RECIPES[source]
        assert previous[target]['recipe'] == native.RECIPES[target]
        if not same_warm(previous[source], previous[target]):
            ctx['refused'].add(target)
            print(json.dumps(dict(status='NONIDENTICAL_WARM_HEAD_KEEP_ORIGINAL_FIT',
                pair=ctx['pair'], video=ctx['video'], target=target)), flush=True)
            continue
        source_sha = file_sha(source_path)
        data = torch.load(source_path, map_location='cpu', weights_only=False)
        cp = data['checkpoint']
        assert cp['recipe'] == native.RECIPES[source]
        assert same_array(cp['mean'], ctx['mean']) and same_array(cp['std'], ctx['std'])
        assert len(data['report']['fits']) == (4 if native.scheduled(native.RECIPES[target]) else 1)
        if native.RECIPES[target]['family'] != 'structured':
            assert all(f['teacher_fit_gate_passed'] and f['excess_risk'] <= .005
                       for f in data['report']['fits'])
        binding = dict(producer_pid=os.getpid(), pair=ctx['pair'], video=ctx['video'],
            source_arm=source, target_arm=target, source_head=str(source_path),
            source_head_sha256=source_sha, original_registration_sha256=PLAN_SHA,
            entry_sha256=file_sha(Path(__file__)), exact_input_sha256=ctx['hashes'],
            prefix_videos=ctx['prefix_videos'], paid_dictionaries_and_sequence_identical=True,
            previous_model_dtype_values_and_normalization_identical=True,
            current_normalizer_identical=True, recipe_differs_only_in_policy_metadata=True,
            original_seed0_fit_function_and_schedule_identical=True,
            shared_visual_record_and_prefix_arrays_by_pinned_original_caller=True,
            new_fit_teacher_actor_question_prediction_GT_or_neural_seed=0,
            original_worker_generates_its_own_predictions=True,
            shared_fit_cost_not_independent_or_deployment_timing=True)
        copied = copied_head(data, target, binding)
        assert same_model(cp['model'], copied['checkpoint']['model'])
        assert same_model(data['early'], copied['early'])
        native.registration()
        if file_sha(ctx['stream']) != ctx['stream_sha']:
            return installed
        assert file_sha(source_path) == source_sha
        progress = json.loads((root / 'progress.json').read_text())
        if progress['video'] != ctx['video'] or progress['completed_arms'] >= ORDER.index(target):
            ctx['finished'].add(target)
            continue
        if (root / 'completion.json').exists() or (native.OUT / 'all_predictions_sealed_before_GT.json').exists():
            return installed
        folder = local / 'exact_policy_input_reuse' / target
        folder.mkdir(parents=True, exist_ok=True)
        prepared = folder / ('prepared_' + str(os.getpid()) + '.pt')
        torch.save(copied, prepared)
        try:
            os.link(prepared, destination)
        except FileExistsError:
            ctx['finished'].add(target)
            continue
        result = dict(status='EXACT_INPUT_REUSED_HEAD_INSTALLED_PARENT_REUSE_PENDING',
            pair=ctx['pair'], video=ctx['video'], source_arm=source, target_arm=target,
            producer_pid=os.getpid(), prepared_sha256=file_sha(prepared),
            destination=str(destination), binding=binding,
            actual_parent_reuse_verified=False, new_independent_fit=False)
        atomic_json(folder / 'receipt.json', result)
        installed.append(result)
        ctx['finished'].add(target)
        print(json.dumps({k: result[k] for k in ('status', 'pair', 'video', 'source_arm',
            'target_arm', 'producer_pid', 'actual_parent_reuse_verified')}), flush=True)
    return installed


def main(pairs, exclusions, once):
    assert file_sha(native.PLAN) == PLAN_SHA
    plan = native.registration()
    assert len(plan['source_sha256']) == 18 and not torch.cuda.is_initialized()
    torch.set_num_threads(2)
    contexts, done, outputs = {}, set(), []
    status_path = native.OUT / 'supervisor' / ('exact_policy_input_reuse_' + str(os.getpid()) + '.json')
    print(json.dumps(dict(status='RUNNING_CPU_EXACT_INPUT_CACHE_PRODUCER', pid=os.getpid(),
        pairs=pairs, exclusions=exclusions, new_quality_conditions=0, new_GPU_calls=0)), flush=True)
    atomic_json(status_path, dict(status='EXACT_INPUT_CACHE_PRODUCER_WAITING_SOURCE_HEADS',
        pid=os.getpid(), pairs=pairs, exclusions=exclusions, installed=[],
        new_quality_conditions=0, new_GPU_calls=0, GT_open_attempts=0, goal_status='ACTIVE'))
    while len(done) < len(pairs):
        if (native.OUT / 'all_predictions_sealed_before_GT.json').exists():
            break
        changed = False
        for pair in pairs:
            if pair in done:
                continue
            root = native.OUT / pair / 'seed0'
            if (root / 'completion.json').exists():
                done.add(pair)
                changed = True
                continue
            current = json.loads((root / 'progress.json').read_text())['video']
            if pair not in contexts or contexts[pair]['video'] != current:
                ctx = context(pair)
                if ctx is None:
                    continue
                if ctx['status'] != 'EXACT_POLICY_INPUT_PREFIX_CERTIFIED':
                    print(json.dumps(ctx), flush=True)
                    done.add(pair)
                    changed = True
                    continue
                contexts[pair] = ctx
                changed = True
            fresh = produce_ready(contexts[pair], exclusions.get(pair, []))
            outputs.extend(fresh)
            changed = changed or bool(fresh)
        if changed:
            atomic_json(status_path, dict(status='EXACT_INPUT_CACHE_PRODUCER_WORKING',
                pid=os.getpid(), pairs=pairs, done=sorted(done), installed=outputs,
                new_quality_conditions=0, new_GPU_calls=0, GT_open_attempts=0, goal_status='ACTIVE'))
        if once:
            break
        time.sleep(1)
    atomic_json(status_path, dict(status='PASS_EXACT_INPUT_CACHE_PRODUCER_FINISHED',
        pid=os.getpid(), pairs=pairs, done=sorted(done), installed=outputs,
        new_independent_fits=0, new_quality_conditions=0, new_GPU_calls=0,
        GT_open_attempts=0, goal_status='ACTIVE'))
    print(json.dumps(dict(status='PASS_EXACT_INPUT_CACHE_PRODUCER_FINISHED', pid=os.getpid(),
        installed_heads=len(outputs), inactive_pairs=sorted(done), receipt=str(status_path),
        new_independent_fits=0, new_quality_conditions=0, GT_open_attempts=0, goal_status='ACTIVE')), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pairs', nargs='+', choices=('C3', 'C4', 'C5'), required=True)
    parser.add_argument('--exclude', nargs='*', default=[], help='PAIR:ARM reserved by an actual fitting job')
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    exclusions = {}
    for item in args.exclude:
        pair, arm = item.split(':', 1)
        assert pair in args.pairs and arm in native.RECIPES
        exclusions.setdefault(pair, []).append(arm)
    main(args.pairs, exclusions, args.once)
