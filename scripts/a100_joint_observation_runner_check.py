"""CPU integration checks only; no new RMOT quality evaluation or GPU job."""
import copy
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from flex_tta.runtime.pf_receipts import atomic_json, file_sha
from flex_uda.a100.mechanism import generation_guard, metadata
from flex_uda.active.joint_observation_inputs import JointObservationVideo, use_verified_local_features
from flex_uda.active.joint_observation_study import (
    video_records, prefix_arrays, coalesce_exact_inputs, make_model, fit_prefix)
from scripts.a100_joint_observation_collect import OUT as COLLECTION
from scripts.a100_joint_observation_six_setting import RECIPES, METHODS, infer_video, OUT


def main():
    guard = generation_guard()
    torch.set_num_threads(1)
    use_verified_local_features()
    video = JointObservationVideo('C1', sorted(metadata('Dance')['videos'])[0])
    common = video_records(video, COLLECTION)
    mean = common['norm_sum']/common['norm_rows']
    std = np.sqrt(np.maximum(common['norm_square']/common['norm_rows']-mean**2, 0)).clip(min=1e-3)
    arrays = prefix_arrays([common['records']['DESIGN']], mean, std)
    projected, projection = coalesce_exact_inputs(arrays, dict(family='generic'))
    assert np.isfinite(projected['first']).all()
    # Tiny model fits exercise the deployed optimizer API. These are not
    # RMOT trained heads, candidate selection, or another experimental seed.
    rng = np.random.default_rng(0)
    toy = dict(anchors=rng.normal(size=(4, 3, 8)).astype('float32'),
               present=np.ones((4, 3), dtype=bool),
               first=rng.normal(size=(4, 8)).astype('float32'),
               second=rng.normal(size=(4, 8)).astype('float32'),
               joint_target=np.tile([.1, .2, .3, .4], (4, 1)),
               binary_target=np.tile([.15, .15, .35, .35], (4, 1)),
               coarse_inputs=rng.normal(size=(4, 8)).astype('float32'),
               coarse_target=np.asarray([.15, .85, .25, .75]))
    fits = {}
    for name in ('FULL_CONDITIONAL_REFERENCE_DESIGN', 'JOINT_KD_DESIGN',
                 'MARGINAL_KD_DESIGN', 'COARSE_UNIT_SOFT_DESIGN', 'COARSE_CONF_HARD_DESIGN'):
        _, _, _, receipt = fit_prefix(toy, RECIPES[name], np.zeros(8), np.ones(8), device='cpu')
        fits[name] = dict(steps=receipt['actual_final_Adam_steps'],
                          excess_KL=receipt['metrics']['excess_KL'],
                          gate=receipt['teacher_fit_gate_passed'])
        assert receipt['teacher_fit_gate_passed']
    # Use real descriptors with just two queries / eight existing observations
    # to exercise all 33 inference paths, including nonzero logits and masks.
    proxy = copy.copy(video)
    proxy.queries = video.queries[:2]
    proxy.old_queries = video.old_queries[:2]
    proxy.raw_bags = video.raw_bags[:2]
    proxy.obs = video.obs[:8]
    proxy.columns = video.columns[:8]
    small = dict(common, pool_values=common['pool_values'][:2], coarse=common['coarse'][:2])
    models = {}
    for name, recipe in RECIPES.items():
        torch.manual_seed(0)
        model = make_model(recipe, video.dimension, 4).eval()
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(.003)
        early = copy.deepcopy(model)
        with torch.no_grad():
            for parameter in early.parameters():
                parameter.mul_(.9)
        models[name] = model, early
    OUT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='cpu_inference_check_', dir=OUT) as path:
        scores = infer_video(proxy, small, models, mean, std, Path(path), 'cpu')
        assert set(scores) == set(METHODS) and len(scores) == 33
        assert all(s.shape == (2, 8) and s.dtype == np.float64 and np.isfinite(s).all() for s in scores.values())
    assert not guard
    report = dict(status='PASS_JOINT_STUDY_CPU_INPUT_FIT_AND_33_PATH_INFERENCE',
                  real_input_video=video.video, real_input_dimension=video.dimension,
                  projection=projection, toy_API_fit_metrics=fits,
                  inference_methods=len(METHODS), inference_rows_per_method=16,
                  actual_GPU_teacher_backbone_calls=0, GT_open_attempts=0,
                  new_RMOT_HOTA_or_quality_claim=False,
                  source_sha256={n: file_sha(ROOT/n) for n in (
                      'scripts/a100_joint_observation_runner_check.py',
                      'scripts/a100_joint_observation_six_setting.py',
                      'flex_uda/active/joint_observation_study.py')}, goal_status='ACTIVE')
    atomic_json(OUT/'cpu_integration_check.json', report)
    atomic_json(ROOT/'docs/a100_six_direction_20261122/joint_observation_runner_check.json', report)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
