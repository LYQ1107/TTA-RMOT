"""A schedule-matched ordinary KD control for the fixed completed primary.

Only one extra ordinary recipe, not a broad module matrix. Preserve every
original opponent and the original primary's failed strict screening result.
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json, file_sha
from scripts import a100_visual_constraint_primary as parent

OUT = parent.OUT.parent/'visual_constraint_schedule_KD_20261125'
PUBLIC = parent.PUBLIC
PLAN = OUT/'registration.json'
STEM = 'visual_constraint_schedule_KD_20261125'
PRIMARY = parent.PRIMARY
ORDINARY = 'SCHEDULE_MATCHED_MEAN_SOFT_KD'


def prepare():
    plan = parent.registration()
    assert json.loads((PUBLIC/(parent.STEM+'_reproduction.json')).read_text())['records'] == 84
    refs = {}
    for case in plan['cases']:
        for leaf in ('scores.npy', 'mask.npy', 'predictions_seal.json'):
            path = parent.OUT/case['pair']/'predictions'/PRIMARY/leaf; refs[str(path)] = file_sha(path)
    record = dict(status='REGISTERED_ONE_SCHEDULE_MATCHED_ADEQUATE_ORDINARY_CONTROL',
        primary=PRIMARY, comparators=[ORDINARY], cases=plan['cases'],
        source_sha256={'scripts/a100_visual_constraint_schedule_kd.py': file_sha(Path(__file__))},
        parent_registration_sha256=file_sha(parent.PLAN), parent_prediction_sha256=refs,
        unchanged_primary=True, all_original9_controls_retained=True, original_strict_screen_still_failed=True,
        hypothesis='Separate visual-constraint learning from the changed optimization/data schedule of the new primary.',
        schedule='Same seed0/init9352/128/128 head; exact500-step subset bootstrap, freshAdam,1500steps on same nonprobe subset,4000steps on ALLteacher responses. Samebatch512/lr.001/gradclip10. No graph or learnedobserver.',
        ordinary_adequacy='Minimum6000 total steps; continue all-guidance fitting if proper excessKL>.005, original maximum16000 retained.',
        information='Exact same target images, features, responses, final complete labels and source models as parent. No teacher target changed or new teacher call.',
        metric_scope='Three existing first videos, allDEV; raw and BOTH CountMatched. All9 new outputs plus3 fixed primary references sealed before new-control GT evaluation.',
        official_statistical_records=12, paired_effects=9,
        implementation='One GPU2 worker, memory fraction.10, sharing only after sufficient-free-memory check; all existing processes preserved.',
        no_new_neural_seed_or_main_acceptance_claim=True, goal_status='ACTIVE')
    atomic_json(PLAN, record); atomic_json(PUBLIC/(STEM+'_registration.json'), record)
    print(record['status'], flush=True)


def registration():
    plan = json.loads(PLAN.read_text()); parent.registration()
    assert file_sha(parent.PLAN) == plan['parent_registration_sha256']
    for name, sha in plan['source_sha256'].items(): assert file_sha(ROOT/name) == sha, name
    for name, sha in plan['parent_prediction_sha256'].items(): assert file_sha(name) == sha, name
    return plan


def run():
    from flex_uda.active.deterministic_observation_runtime import configure_before_cuda
    execution = configure_before_cuda()
    import numpy as np
    import torch
    from torch import nn
    from flex_uda.active.visual_constraint_learning import make_solver, probe_mask, teacher_loss, cpu_state, save
    from flex_uda.active.joint_teacher_noise import fit_solver, risk_floor
    from flex_uda.a100.mechanism import generation_guard, metadata, PAIRS
    from flex_uda.presence.predictions import seal_mask, verify_predictions
    from scripts.a100_factor_rsi_primary_quality import subset_metadata, projection
    plan = registration(); guard = generation_guard()
    torch.set_num_threads(2); torch.cuda.set_device(0); torch.cuda.set_per_process_memory_fraction(.10)
    tasks = []
    for case in plan['cases']:
        pair = case['pair']; folder = OUT/pair; folder.mkdir(parents=True, exist_ok=True)
        checkpoint = folder/'ordinary.pt'
        with np.load(parent.DATA/pair/'teacher_inputs.npz') as data: features, probability = data['features'], data['probability']
        keys = json.loads((parent.DATA/pair/'row_keys.json').read_text())['keys']
        target = np.nanmean(probability, axis=1); hold = probe_mask(keys, case['evaluated_video'])
        model = make_solver('cuda:0')
        if checkpoint.exists():
            saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
            assert saved['registration_sha256'] == file_sha(PLAN) and saved['adequate_passed']
            model.load_state_dict(saved['model'])
        else:
            started = time.perf_counter()
            bootstrap = fit_solver(model, features[~hold], target[~hold], 'cuda:0', steps=500)
            optimizer = torch.optim.Adam(model.parameters(), lr=.001)
            x = torch.as_tensor(features, device='cuda:0'); y = torch.as_tensor(target, dtype=torch.float32, device='cuda:0')
            selected = torch.as_tensor(~hold, device='cuda:0')
            floor = risk_floor(features, target, np.ones(len(target)))
            for step in range(15500):
                optimizer.zero_grad(set_to_none=True)
                loss = teacher_loss(model, x[selected], y[selected]) if step < 1500 else teacher_loss(model, x, y)
                loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 10.); optimizer.step()
                total = 500+step+1
                if total >= 6000 and total % 1000 == 0:
                    with torch.no_grad():
                        logits = torch.cat([model(x[b:b+512]).double() for b in range(0, len(x), 512)])
                        ce = float(nn.functional.binary_cross_entropy_with_logits(logits, torch.as_tensor(target, device='cuda:0')))
                    if ce-floor <= .005: break
            assert ce-floor <= .005
            saved = dict(model=cpu_state(model), total_steps=total, bootstrap=bootstrap,
                proper_teacher_CE=ce, attainable_floor=floor, excess_KL=ce-floor, adequate_passed=True,
                registration_sha256=file_sha(PLAN), data_sha256=case['data_sha256'],
                wall_seconds=time.perf_counter()-started, seed=0)
            save(checkpoint, saved)
        dense = np.load(parent.OUT/pair/'graph/features.npy', mmap_mode='r')
        primary_folder = parent.OUT/pair/'predictions'/PRIMARY
        original_scores = np.load(primary_folder/'scores.npy'); original_mask = np.load(primary_folder/'mask.npy')
        scores = np.empty_like(original_scores)
        with torch.no_grad():
            for begin in range(0, len(dense), 512):
                x = torch.as_tensor(np.asarray(dense[begin:begin+512]), device='cuda:0')
                scores.reshape(-1)[begin:begin+len(x)] = model(x).cpu().numpy()
        mask = scores > 0
        obs = np.asarray(metadata(case['target'])['videos'][case['evaluated_video']]['observations'], dtype=int)
        md = subset_metadata(case['target'], case['evaluated_video'])
        root = folder/'predictions'/ORDINARY; root.mkdir(parents=True, exist_ok=True)
        np.save(root/'scores.npy', scores); np.save(root/'mask.npy', mask)
        assert np.array_equal(projection(scores, mask, obs), mask)
        seal_mask(root, md, mask.flatten(), target=case['target'], model=PAIRS[pair][0], method=ORDINARY,
                  GT_open_attempts=len(guard), extra=dict(same_primary_schedule=True, same_features_and_all_teacher_responses=True))
        tasks.extend([dict(pair=pair, method=PRIMARY, folder=str(primary_folder)), dict(pair=pair, method=ORDINARY, folder=str(root))])
        for rank, count, value, donor in ((PRIMARY, ORDINARY, original_scores, mask), (ORDINARY, PRIMARY, scores, original_mask)):
            name = rank+'__AT_'+count+'_COUNT'; destination = folder/'predictions'/name
            selected_mask = projection(value, donor, obs)
            seal_mask(destination, md, selected_mask.flatten(), target=case['target'], model=PAIRS[pair][0], method=name,
                      GT_open_attempts=len(guard), extra=dict(rank_method=rank, count_method=count, counts_from_GT=False, counts_equal_per_query_frame=True))
            tasks.append(dict(pair=pair, method=name, folder=str(destination)))
        atomic_json(folder/'completion.json', dict(status='SEALED_NEW_SCHEDULE_MATCHED_ORDINARY_NO_GT',
            fit={k: v for k, v in saved.items() if k != 'model'}, GT_open_attempts=0, execution=execution))
        print(json.dumps(dict(status='SCHEDULE_MATCHED_KD_ADEQUATE_AND_SEALED', pair=pair, steps=saved['total_steps'], excess_KL=saved['excess_KL'])), flush=True)
    assert len(tasks) == 12 and not guard
    for task in tasks: verify_predictions(task['folder'])
    atomic_json(OUT/'all12_sealed_before_GT.json', dict(tasks=tasks, GT_open_attempts=0, parent_primary_already_evaluated=True))
    from scripts.a100_joint_teacher_noise_pilot import evaluate_one
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool: completed = list(pool.map(evaluate_one, tasks))
    results = {c['pair']: {} for c in plan['cases']}
    for task, value in completed: results[task['pair']][task['method']] = value
    comparisons = []
    for pair, value in results.items():
        a, b = value[PRIMARY]['HOTA_unrounded'], value[ORDINARY]['HOTA_unrounded']
        comparisons.append(dict(pair=pair, comparator=ORDINARY, primary_HOTA=a, ordinary_HOTA=b,
            raw_delta_HOTA=a-b, primary_at_ordinary_count_delta_HOTA=value[PRIMARY+'__AT_'+ORDINARY+'_COUNT']['HOTA_unrounded']-b,
            ordinary_at_primary_count_delta_HOTA=a-value[ORDINARY+'__AT_'+PRIMARY+'_COUNT']['HOTA_unrounded']))
    eligible = all(min(x['raw_delta_HOTA'], x['primary_at_ordinary_count_delta_HOTA'], x['ordinary_at_primary_count_delta_HOTA']) > 0 for x in comparisons)
    atomic_json(OUT/'quality_results.json', dict(status='COMPLETE_EXTRA_SCHEDULE_MATCHED_ORDINARY', results=results,
        comparisons=comparisons, candidate_eligible_for_larger_screen=eligible,
        original_main_screen_not_overridden=True, not_fullsix_or_three_contribution_success=True, all_data_DEV=True, goal_status='ACTIVE'))
    from scripts.a100_primary_statistics_common import collect, reproduce
    collect(OUT, PUBLIC, STEM, 12); reproduce(PUBLIC, STEM, 12, 9)
    print(json.dumps(dict(status='COMPLETE_SCHEDULE_MATCHED_KD_ALL12_REPRODUCED', comparisons=comparisons)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--prepare', action='store_true'); args = parser.parse_args()
    if args.prepare: prepare()
    else:
        import subprocess
        assert os.environ.get('CUDA_VISIBLE_DEVICES') == '2'
        # nvidia-smi indexes physical GPUs despite CUDA_VISIBLE_DEVICES.
        used = int(subprocess.check_output(['nvidia-smi', '-i', '2', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True).strip())
        total = int(subprocess.check_output(['nvidia-smi', '-i', '2', '--query-gpu=memory.total', '--format=csv,noheader,nounits'], text=True).strip())
        assert total-used >= 8192, (used, total)
        atomic_json(OUT/'shared_GPU2_preflight.json', dict(used_MiB=used, total_MiB=total, memory_fraction=.10, all_existing_processes_preserved=True))
        run()
