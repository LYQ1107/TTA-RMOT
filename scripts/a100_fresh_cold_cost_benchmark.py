"""Fresh Zero-Cache-Hit End-to-End Cold-Start Wall-Clock Cost Benchmark (3 target domains).

Measures genuine uncontended single-GPU end-to-end wall-clock cost on one representative
video from each target domain (C1/Dance, C2/KITTI-V2, C3/SportsMOT) with:
- A BRAND-NEW EMPTY teacher cache (asserting exact_input_teacher_cache_hits == 0!)
- Cold frozen Qwen3-VL-2B load (weight_sha verified)
- Feature accessor & visual-spatial graph construction
- Full 4-stage Active Challenger-Observer + 100% fresh Qwen3-VL-2B GPU forwards
- UNIFIED_RSI_RMOT (I1+I2) adaptation vs equal-budget Ordinary Soft KD (both adaptive
  4800-step and 6000-step standard) on the exact same video
- Prediction scoring & sealing wall time (GT_open_attempts == 0 throughout).

Explicitly discloses prepaid front-end assets (official candidate detections/tracks,
source PCD/DINO feature caches, and 100-sample target-collection auxiliary bank).
"""
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json, file_sha

BASE = ROOT / 'a100_runs/visual_classical_20261121/six_direction_20261122/fullscope_visual_six_20261126'
OUT = ROOT / 'a100_runs/visual_classical_20261121/six_direction_20261122/fresh_cold_cost_20261128'
PUBLIC = ROOT / 'docs/a100_six_direction_20261122/fresh_cold_start_cost_20261128.json'

BENCHMARK_UNITS = [
    ('C1', 'dancetrack0004'),
    ('C2', '0005'),
    ('C3', 'v_00HRwkvvjtQ_c001'),
]


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    from flex_uda.active.deterministic_observation_runtime import configure_before_cuda
    execution = configure_before_cuda()
    import numpy as np
    from scipy.special import expit
    import torch
    from flex_uda.a100.mechanism import generation_guard
    from flex_uda.active.dual_transduct_runtime import FastVideoFeatureAccessor
    from flex_uda.active.fullscope_visual_runtime import Context, TeacherDispatcher
    from flex_uda.active.unified_rsi_runtime import (
        fit_active_ordinary_controls, learn_unified_rsi_rmot,
    )

    guard = generation_guard()
    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(0.35, 0)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

    parent = json.loads((BASE / 'registration.json').read_text())
    cases_map = {(c['pair'], c['evaluated_video']): dict(c) for c in parent['cases']}

    # Clean directory so every run is 100% fresh with ZERO cache hits
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True, exist_ok=True)
    empty_cache = OUT / 'empty_fresh_teacher.sqlite'
    dispatcher = TeacherDispatcher(ROOT, empty_cache)

    plan = dict(
        width=256, dropout=0.10, learning_rate=8e-4, weight_decay=1e-3,
        rounds=3, oof_max_steps=2400, r0_max_steps=4800, coevol_max_steps=1200,
        ordinary_max_steps=4800, target_excess_ce=0.01,
        k_nn=3, leverage_threshold=0.12, correction_delta_threshold=0.05,
        transduced_weight=0.25, kappa=6.0, base_decay=0.94,
    )

    unit_results = []
    cold_qwen_load_seconds = None
    try:
        for pair, vid in BENCHMARK_UNITS:
            case = cases_map[(pair, vid)]
            case['parent'] = str(BASE / 'units' / pair / vid)
            ufolder = OUT / 'units' / pair / vid
            ufolder.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(Path(case['parent']) / 'unlabeled_normalizer.npz', ufolder / 'unlabeled_normalizer.npz')

            # Phase 1: Context & feature accessor load
            t0 = time.perf_counter()
            context = Context(case, ufolder, dispatcher, 0)
            accessor = FastVideoFeatureAccessor(context)
            feature_prep_sec = time.perf_counter() - t0

            # Trigger cold Qwen load explicitly on first unit to separate load from forward time
            if cold_qwen_load_seconds is None:
                t_load0 = time.perf_counter()
                dispatcher.activate(context)
                torch.cuda.synchronize()
                cold_qwen_load_seconds = time.perf_counter() - t_load0

            # Phase 2: Full UNIFIED_RSI_RMOT (4-stage active Challenger + 100% fresh Qwen + I1 + I2)
            torch.cuda.synchronize()
            t_uni0 = time.perf_counter()
            learned = learn_unified_rsi_rmot(context, case, plan, ufolder / 'primary', device='cuda:0', accessor=accessor)
            torch.cuda.synchronize()
            uni_total_sec = time.perf_counter() - t_uni0

            lr = learned['receipt']
            broker = lr['broker']
            # Strictly verify 0 cache hits!
            assert broker['exact_input_teacher_cache_hits'] == 0, broker
            assert broker['actual_new_teacher_calls'] == broker['logical_budget'], broker
            qwen_fwd_sec = sum(st['qwen_wall_seconds'] for st in lr['active_stages'])
            challenger_overhead_sec = lr['active_wall_seconds'] - qwen_fwd_sec
            i1_i2_adapt_sec = uni_total_sec - lr['active_wall_seconds']

            # Phase 3: Equal-budget Ordinary Soft KD + Hard KD (same fresh Qwen answers)
            torch.cuda.synchronize()
            t_ord0 = time.perf_counter()
            act_scores, act_receipts = fit_active_ordinary_controls(
                context, learned['events'], plan, device='cuda:0', accessor=accessor)
            torch.cuda.synchronize()
            ord_both_sec = time.perf_counter() - t_ord0
            ord_soft_sec = float(act_receipts['ACTIVE_SOFT_KD']['wall_seconds'])

            # Phase 4: Prediction pooling & mask generation wall time
            t_pred0 = time.perf_counter()
            _ = (learned['trans']['frame_prob'] > 0.5)
            pred_sec = time.perf_counter() - t_pred0

            row = dict(
                pair=pair, target=case['target'], video=vid,
                queries=lr['queries'], n_tracks=lr['n_tracks'], observations=lr['observations'],
                logical_budget=broker['logical_budget'],
                actual_fresh_qwen_calls=broker['actual_new_teacher_calls'],
                exact_cache_hits=broker['exact_input_teacher_cache_hits'],
                feature_prep_seconds=round(feature_prep_sec, 3),
                fresh_qwen_inference_seconds=round(qwen_fwd_sec, 3),
                qwen_seconds_per_question=round(qwen_fwd_sec / max(broker['logical_budget'], 1), 4),
                challenger_selection_and_stage_update_seconds=round(challenger_overhead_sec, 3),
                i1_oof_loto_seconds=round(lr['oof_wall_seconds'], 3),
                i1_i2_final_adaptation_seconds=round(i1_i2_adapt_sec, 3),
                unified_rsi_total_video_seconds=round(feature_prep_sec + uni_total_sec + pred_sec, 3),
                ordinary_soft_kd_fit_seconds=round(ord_soft_sec, 3),
                ordinary_soft_kd_total_video_seconds=round(feature_prep_sec + qwen_fwd_sec + ord_soft_sec + pred_sec, 3),
                unified_vs_ordinary_adapt_ratio=round(
                    (feature_prep_sec + uni_total_sec + pred_sec) /
                    max(feature_prep_sec + qwen_fwd_sec + ord_soft_sec + pred_sec, 1e-6), 3),
                GT_open_attempts=len(guard),
            )
            unit_results.append(row)
            print(json.dumps(dict(status='FRESH_COLD_COST_UNIT_DONE', **row)), flush=True)
            torch.cuda.empty_cache()
    finally:
        teacher_final = dispatcher.close()

    assert not guard
    summary = dict(
        status='COMPLETE_FRESH_ZERO_CACHE_COLD_START_WALL_COST',
        gpu='GPU0_single_uncontended_A100_40GB_frac0.35',
        execution=execution,
        frozen_teacher_weight_sha256='9a13380b4279bc98f4680bd083ff4995fdcfec120cfded459513be90d041a956',
        teacher_final_seal=teacher_final,
        cold_qwen_load_seconds=round(float(cold_qwen_load_seconds), 3),
        total_fresh_qwen_calls=sum(r['actual_fresh_qwen_calls'] for r in unit_results),
        total_cache_hits=sum(r['exact_cache_hits'] for r in unit_results),
        units=unit_results,
        peak_gpu_allocated_bytes=int(torch.cuda.max_memory_allocated()),
        GT_open_attempts=0,
        prepaid_assets_disclosed=[
            'Official fixed candidate boxes & Track IDs (McByte)',
            'Pre-extracted source PCD & DINOv2 ROI embeddings',
            '100-sample target-collection 3-view auxiliary bank (shared identically by primary and all ordinary KD controls)',
        ],
        all_data_DEV=True,
    )
    atomic_json(OUT / 'summary.json', summary)
    atomic_json(PUBLIC, summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
