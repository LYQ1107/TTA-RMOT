"""Teacher-only representational floor of unsuccessful ordinary frame fits.

Identical float32 network inputs cannot produce distinct Bernoulli outputs.
The exact weighted optimum for such an input class is its weighted mean
target. This floor diagnoses observability; it is not a semantic-error rate
or a reason to relax the original fit gate. No quality predictions or GT.
"""
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from flex_tta.runtime.pf_receipts import atomic_json, file_sha
from flex_uda.a100.mechanism import generation_guard
from scripts.a100_object_reference_fit_complete import OUT as FIT_ROOT
from scripts.a100_six_direction import PUBLIC

OUT = FIT_ROOT / 'supervisor/representation_floor_audit'
REGISTRATION = PUBLIC / 'frame_fit_representation_audit_registration.json'


def exact_floor(x, target, weight):
    x = np.ascontiguousarray(x, dtype=np.float32)
    target, weight = np.asarray(target, float), np.asarray(weight, float)
    assert x.ndim == 2 and len(x) == len(target) == len(weight)
    assert np.isfinite(x).all() and (weight > 0).all()
    encoded = x.view(np.dtype((np.void, x.shape[1]*x.dtype.itemsize))).ravel()
    _, inverse, count = np.unique(encoded, return_inverse=True, return_counts=True)
    mass = np.bincount(inverse, weights=weight)
    optimum = np.bincount(inverse, weights=weight*target) / mass
    entropy = -target*np.log(target) - (1-target)*np.log1p(-target)
    p = optimum[inverse]
    ce = -target*np.log(p) - (1-target)*np.log1p(-p)
    floor = float(weight@(ce-entropy)/weight.sum())
    low = np.full(len(count), np.inf); high = np.full(len(count), -np.inf)
    np.minimum.at(low, inverse, target); np.maximum.at(high, inverse, target)
    contradictory = (count > 1) & (high-low > 1e-7)
    return dict(examples=len(x), exact_float32_input_classes=len(count),
        repeated_input_classes=int((count > 1).sum()),
        conflicting_target_classes=int(contradictory.sum()),
        conflicting_target_examples=int(count[contradictory].sum()),
        exact_input_class_optimal_teacher_KL=max(0., floor),
        interpretation='Only an equality-class lower bound. Distinct inputs do not imply this 64-unit head can fit them or that their targets are correct.')


def main():
    plan = json.loads(REGISTRATION.read_text())
    assert file_sha(Path(__file__)) == plan['source_sha256']
    guard = generation_guard(); begin = time.perf_counter(); rows = []
    torch.set_num_threads(1)
    for pair in ('C2', 'C5'):
        for seed in (0, 17, 42):
            root = FIT_ROOT/pair/f'seed{seed}'
            completion = json.loads((root/'completion.json').read_text())
            stream = torch.load(root/'stream.pt', map_location='cpu', weights_only=False)
            assert stream['next_video'] == len(completion['receipts']) == 4
            for vi, receipt in enumerate(completion['receipts']):
                fit = receipt['fits']['FRAME_COMPLETE']
                if fit['teacher_fit_within_0_005']:
                    continue
                video = receipt['video']
                checkpoint = torch.load(root/'videos'/video/'fit_complete_head.pt',
                    map_location='cpu', weights_only=False)
                x = np.concatenate(stream['bx'][:vi+1])
                raw_y = np.concatenate(stream['by'][:vi+1])
                x = ((x-checkpoint['mean'])/checkpoint['std']).astype('float32')
                result = exact_floor(x, np.clip(raw_y, .02, .98),
                    1.+3.*np.abs(2.*raw_y-1.))
                rows.append(dict(pair=pair, seed=seed, video=video, prefix_videos=vi+1,
                    actual_teacher_KL=fit['teacher_KL'], **result))
    assert not guard
    # A small algebraic witness checks the lower-bound calculation rather
    # than mirroring a neural implementation.
    witness = exact_floor(np.array([[1.], [1.]]), np.array([.1, .9]), np.ones(2))
    expected = np.log(2) + .1*np.log(.1)+.9*np.log(.9)
    assert abs(witness['exact_input_class_optimal_teacher_KL']-expected) < 1e-12
    report = dict(status='TEACHER_ONLY_FRAME_FIT_REPRESENTATION_FLOOR_AUDIT_COMPLETE',
        failed_prefixes=rows, checked_synthetic_conflict_floor=expected,
        original_teacher_KL_gate=.005, gate_changed=False,
        input_equality_uses_actual_normalized_float32_network_inputs=True,
        quality_predictions_or_manual_GT_read=0, new_neural_teacher_encoder_fits=0,
        no_future_prefix_inputs=True, current_study_or_outputs_changed=False,
        limitations='A lower bound, not a fitted richer baseline or an RMOT benefit. Geometry and frozen coarse features need not expose all information in teacher images.',
        wall_seconds=time.perf_counter()-begin, goal_status='ACTIVE')
    atomic_json(OUT/'completion.json', report)
    (PUBLIC/'frame_fit_representation_audit.json').write_text(
        json.dumps(report, ensure_ascii=False, separators=(',', ':'))+'\n')
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
