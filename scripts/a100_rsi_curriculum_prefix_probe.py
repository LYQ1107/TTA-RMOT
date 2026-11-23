"""Same-seed prefix experiment for raw versus prior-controlled RSI feedback.

Reuse the completed raw/frozen first video. No method may import that full
video head to predict an earlier round. This script never generates RMOT
predictions or opens evaluation labels; task quality remains unproven.
"""
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from flex_tta.runtime.pf_receipts import atomic_json, file_sha
from flex_uda.a100.mechanism import metadata, generation_guard
from flex_uda.active.joint_observation_inputs import JointObservationVideo, use_verified_local_features
from flex_uda.active.joint_observation_study import fit_prefix, prefix_arrays
from flex_uda.active.rsi_curriculum import question_inventory
from flex_uda.active.rsi_curriculum_learning import run_video_curriculum
from flex_uda.active.rsi_observation_inputs import RSIVisualInputs
from flex_uda.active.rsi_progress_reward import skill_progress
from scripts.a100_joint_observation_collect import OUT as COLLECTION
from scripts.a100_six_direction import PUBLIC, OUT as SIX

OUT = SIX/'rsi_curriculum_prefix_20261123'
RAW = SIX/'rsi_curriculum_engineering_20261123'
PLAN = PUBLIC/'rsi_curriculum_prefix_plan.json'
STATE = OUT/'state.json'


def reward_contract():
    rows = [
        ('FROZEN_SOLVER_NULL_IMPROVES', (1., 1., 1., .1), 0),
        ('FROZEN_SOLVER_NULL_WORSENS', (1., 1., 1., 2.), 0),
        ('SAME_PRIOR_ONLY_GAIN', (1., .4, 1., .4), 0),
        ('LESS_THAN_PRIOR_GAIN', (1., .7, 1., .4), 0),
        ('BEYOND_PRIOR_GAIN', (1., .4, 1., .8), 1),
        ('SOLVER_REGRESSION_BAD_NULL', (1., 1.2, 1., 2.), -1)]
    results = {}
    for name, values, expected in rows:
        result = skill_progress(*values)
        assert np.sign(result['reward']) == expected, (name, result)
        results[name] = result
    return results


def main():
    plan = json.loads(PLAN.read_text())
    for name, sha in plan['source_sha256'].items():
        assert file_sha(ROOT/name) == sha, name
    assert plan['neural_seeds'] == [0] and plan['device'] == 'cpu'
    guard = generation_guard()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    use_verified_local_features()
    started = time.perf_counter()
    contract = reward_contract()
    all_reports, selected = {}, {}
    for branch in ('FROZEN', 'RAW', 'PRIOR_CONTROLLED'):
        policy = 'FROZEN_ACTOR' if branch == 'FROZEN' else 'ADAPTIVE'
        reward_mode = 'PRIOR_CONTROLLED' if branch == 'PRIOR_CONTROLLED' else 'RAW_CE'
        records, previous, actor_state, ordinary_previous = [], None, None, {}
        total, square, number = np.zeros(1101), np.zeros(1101), 0
        own = []
        selected[branch] = {}
        for vi, video in enumerate(plan['videos']):
            v = JointObservationVideo(plan['pair'], video)
            manifest = json.loads((COLLECTION/'inputs'/v.target/video/'manifest.json').read_text())
            actions, quotas = question_inventory(manifest, plan['pair'])
            visual = RSIVisualInputs(v, actions)
            total += visual.norm_sum
            square += visual.norm_square
            number += visual.norm_rows
            mean = total/number
            std = np.sqrt(np.maximum(square/number-mean**2, 0)).clip(min=1e-3)
            folder = OUT/branch/video
            folder.mkdir(parents=True, exist_ok=True)
            atomic_json(STATE, dict(status='RUNNING_RSI_SAME_SEED_PREFIX_TEACHER_PROBE',
                        pid=os.getpid(), branch=branch, video=video, completed_videos=vi,
                        total_videos=len(plan['videos']), goal_status='ACTIVE'))
            if vi == 0 and branch in ('FROZEN', 'RAW'):
                initial = RAW/policy
                saved = torch.load(initial/'final.pt', map_location='cpu', weights_only=False)
                previous, actor_state = saved['solver'], saved['actor']
                assert np.array_equal(previous['mean'], mean) and np.array_equal(previous['std'], std)
                paid = json.loads((initial/'paid_answers.json').read_text())['received']
                record = visual.teacher_dataset(paid)
                receipt = json.loads((initial/'receipt.json').read_text())
                receipt = dict(receipt, reused_exact_completed_raw_reward_first_video=True,
                               raw_source_commit=plan['raw_source_commit'])
                for name in ('GENERIC_JOINT', 'COARSE_UNIT_SOFT'):
                    ordinary_previous[name] = torch.load(initial/(name+'.pt'), map_location='cpu', weights_only=False)
            elif (folder/'completion.pt').is_file():
                saved = torch.load(folder/'completion.pt', map_location='cpu', weights_only=False)
                previous, actor_state, ordinary_previous = saved['solver'], saved['actor'], saved['ordinary']
                paid = json.loads((folder/'paid_answers.json').read_text())['received']
                record, receipt = visual.teacher_dataset(paid), json.loads((folder/'receipt.json').read_text())
            else:
                progress = folder/'round_latest.pt'
                resume = torch.load(progress, map_location='cpu', weights_only=False) if progress.is_file() else None
                def callback(stage, receipt, saved):
                    temporary = progress.with_suffix('.tmp')
                    torch.save(saved, temporary)
                    temporary.replace(progress)
                    atomic_json(STATE, dict(status='RUNNING_RSI_SAME_SEED_PREFIX_TEACHER_PROBE',
                                pid=os.getpid(), branch=branch, video=video, completed_round=stage,
                                actor_update=receipt['actor_update'], goal_status='ACTIVE'))
                    print(json.dumps(dict(branch=branch, video=video, stage=stage,
                                          reward=receipt['actor_update']['reward'],
                                          reward_mode=reward_mode)), flush=True)
                result = run_video_curriculum(visual, actions, quotas, COLLECTION, mean, std,
                    past_records=records, previous=previous, actor_state=actor_state,
                    policy=policy, reward_mode=reward_mode, device='cpu', callback=callback, resume=resume)
                previous, actor_state = result['checkpoint'], result['actor_state']
                paid, record, receipt = result['received'], result['record'], result['report']
                common = prefix_arrays([*records, record], mean, std)
                ordinary = {}
                for name, recipe in (
                    ('GENERIC_JOINT', dict(family='generic', risk='joint', policy=policy)),
                    ('COARSE_UNIT_SOFT', dict(family='coarse', confidence=False, hard=False, policy=policy))):
                    _, _, checkpoint, fit = fit_prefix(common, recipe, mean, std,
                        previous=ordinary_previous.get(name), device='cpu')
                    assert fit['teacher_fit_gate_passed']
                    ordinary[name], ordinary_previous[name] = fit, checkpoint
                receipt['adequate_same_all_paid_ordinary'] = ordinary
            assert receipt['teacher_budget']['unique_teacher_bundles'] == sum(quotas.values())
            atomic_json(folder/'paid_answers.json', dict(received=paid))
            atomic_json(folder/'receipt.json', receipt)
            temporary = folder/'completion.tmp'
            torch.save(dict(solver=previous, actor=actor_state, ordinary=ordinary_previous), temporary)
            temporary.replace(folder/'completion.pt')
            records.append(record)
            own.append(dict(video=video, **receipt))
            selected[branch][video] = {r['uid'] for r in paid}
            print(json.dumps(dict(branch=branch, video=video, completed=True,
                                  actor_actual_updates=receipt['actor_actual_updates'],
                                  teacher_budget=sum(quotas.values()))), flush=True)
        all_reports[branch] = own
    assert not guard
    difference = {video: {branch: len(selected[branch][video] ^ selected['FROZEN'][video])
                         for branch in ('RAW', 'PRIOR_CONTROLLED')} for video in plan['videos']}
    report = dict(status='PASS_SAME_SEED_RSI_PREFIX_FEEDBACK_DIAGNOSIS', pair=plan['pair'],
                  videos=plan['videos'], neural_seeds=[0], branches=all_reports,
                  selected_question_difference_from_frozen=difference,
                  reward_contract=contract, source_sha256=plan['source_sha256'],
                  GT_open_attempts=0, new_RMOT_HOTA_or_predictions=False,
                  actual_new_teacher_or_backbone_calls=0, research_replay_not_deployment_cost=True,
                  no_semantic_RSI_gain_proven_by_this_experiment=True,
                  wall_seconds_this_invocation=time.perf_counter()-started, goal_status='ACTIVE')
    atomic_json(OUT/'analysis.json', report)
    atomic_json(PUBLIC/'rsi_curriculum_prefix_receipt.json', report)
    atomic_json(STATE, dict(status=report['status'], pid=os.getpid(), goal_status='ACTIVE'))
    print(json.dumps(dict(status=report['status'], question_difference=difference,
                          new_RMOT_HOTA=False)), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        atomic_json(STATE, dict(status='FAILED_RSI_PREFIX_REQUIRES_ROOT_CAUSE',
                               error=repr(error), goal_status='ACTIVE'))
        raise
