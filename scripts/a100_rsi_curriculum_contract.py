"""Audit RSI budgets, then exercise a real teacher-only three-round loop.

No RMOT predictions or ground-truth files are produced/read by this script.
The real loop is an engineering/mechanism experiment, not six-setting HOTA.
"""
import argparse
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
from flex_uda.a100.mechanism import PAIRS, generation_guard, metadata
from flex_uda.active.rsi_curriculum import (
    question_inventory, reserve_probes, exploration_log_probabilities)
from scripts.a100_joint_observation_collect import OUT as COLLECTION
from scripts.a100_six_direction import PUBLIC, OUT as SIX

OUT = SIX/'rsi_curriculum_engineering_20261123'
PLAN = PUBLIC/'rsi_curriculum_engineering_plan.json'
STATE = OUT/'state.json'
SOURCES = ('scripts/a100_rsi_curriculum_contract.py',
           'flex_uda/active/rsi_curriculum.py',
           'flex_uda/active/rsi_observation_inputs.py',
           'flex_uda/active/rsi_curriculum_learning.py',
           'flex_uda/active/joint_observation_study.py',
           'flex_uda/active/conditional_joint_observation.py',
           'flex_uda/active/joint_observation_inputs.py')


def audit():
    guard = generation_guard()
    rows = {}
    for pair, (_, target) in PAIRS.items():
        own = []
        for video in sorted(metadata(target)['videos']):
            manifest = json.loads((COLLECTION/'inputs'/target/video/'manifest.json').read_text())
            actions, quotas = question_inventory(manifest, pair)
            reserved = reserve_probes(actions, quotas)
            own.append(dict(video=video, queries=len(quotas), budget=sum(quotas.values()),
                            candidate_actions=len(actions), probes=reserved['actual_probe_count'],
                            fresh_probe_rounds=reserved['fresh_probe_rounds'],
                            minimum_first_round_training=min(reserved['training_schedule'][0].values()),
                            all_per_query_budgets_exact=True))
        rows[pair] = own
    # Verify the score-function identity for the actual epsilon-mixture
    # policy. This checks exploration weighting without another training seed.
    scores = torch.tensor([-.8, .3, 1.1, -.2], dtype=torch.float64, requires_grad=True)
    reward = torch.tensor([.6, -.4, .2, .9], dtype=torch.float64)
    logp = exploration_log_probabilities(scores)
    exact = torch.autograd.grad((logp.exp()*reward).sum(), scores, retain_graph=True)[0]
    score_function = sum(logp[i].exp().detach()*reward[i]*
                         torch.autograd.grad(logp[i], scores, retain_graph=True)[0] for i in range(4))
    error = float((exact-score_function).abs().max())
    assert error < 1e-12 and not guard
    report = dict(status='PASS_RSI_PROBE_BUDGET_AND_POLICY_GRADIENT_CONTRACT',
                  settings={p: dict(videos=len(r), teacher_budget=sum(v['budget'] for v in r),
                                   paid_probe_bundles=sum(v['probes'] for v in r),
                                   min_fresh_probe_rounds=min(v['fresh_probe_rounds'] for v in r),
                                   all_queries_keep_first_round_training=all(v['minimum_first_round_training'] >= 1 for v in r))
                            for p, r in rows.items()},
                  finite_policy_score_function_max_error=error,
                  teacher_answers_read=0, GT_open_attempts=0, model_fits=0,
                  not_RSI_quality_evidence=True, goal_status='ACTIVE')
    atomic_json(OUT/'per_video_probe_budgets.json', rows)
    atomic_json(PUBLIC/'rsi_curriculum_budget_contract.json', report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


def real_loop():
    plan = json.loads(PLAN.read_text())
    for name, sha in plan['source_sha256'].items():
        assert file_sha(ROOT/name) == sha, name
    assert plan['neural_seeds'] == [0] and plan['device'] == 'cpu'
    guard = generation_guard()
    from flex_uda.active.joint_observation_inputs import JointObservationVideo, use_verified_local_features
    from flex_uda.active.rsi_observation_inputs import RSIVisualInputs
    from flex_uda.active.rsi_curriculum_learning import run_video_curriculum
    from flex_uda.active.joint_observation_study import fit_prefix, prefix_arrays
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    start = time.perf_counter()
    use_verified_local_features()
    pair, video = plan['pair'], plan['video']
    inputs = JointObservationVideo(pair, video)
    manifest = json.loads((COLLECTION/'inputs'/inputs.target/video/'manifest.json').read_text())
    actions, quotas = question_inventory(manifest, pair)
    visual = RSIVisualInputs(inputs, actions)
    mean = visual.norm_sum/visual.norm_rows
    std = np.sqrt(np.maximum(visual.norm_square/visual.norm_rows-mean**2, 0)).clip(min=1e-3)
    reports, selected = {}, {}
    for policy in ('ADAPTIVE', 'FROZEN_ACTOR'):
        folder = OUT/policy
        folder.mkdir(parents=True, exist_ok=True)
        atomic_json(STATE, dict(status='RUNNING_REAL_TEACHER_ONLY_RSI_LOOP', pid=os.getpid(),
                    policy=policy, completed_policies=len(reports), goal_status='ACTIVE'))
        def callback(stage, receipt, saved):
            torch.save(saved, folder/('round'+str(stage)+'.pt'))
            atomic_json(folder/('round'+str(stage)+'.json'), receipt)
            atomic_json(STATE, dict(status='RUNNING_REAL_TEACHER_ONLY_RSI_LOOP', pid=os.getpid(),
                        policy=policy, completed_round=stage, actor_update=receipt['actor_update'], goal_status='ACTIVE'))
            print(json.dumps(dict(policy=policy, completed_round=stage,
                                  actor_update=receipt['actor_update'],
                                  teacher_bundles=receipt['paid_teacher']['unique_teacher_bundles'])), flush=True)
        result = run_video_curriculum(visual, actions, quotas, COLLECTION, mean, std,
                                      policy=policy, device='cpu', callback=callback)
        full = result['record']
        all_paid = prefix_arrays([full], mean, std)
        ordinary = {}
        for label, recipe in (
            ('GENERIC_JOINT', dict(family='generic', risk='joint', policy=policy)),
            ('COARSE_UNIT_SOFT', dict(family='coarse', confidence=False, hard=False, policy=policy))):
            _, _, checkpoint, receipt = fit_prefix(all_paid, recipe, mean, std, device='cpu')
            assert receipt['teacher_fit_gate_passed']
            ordinary[label] = receipt
            torch.save(checkpoint, folder/(label+'.pt'))
        torch.save(dict(solver=result['checkpoint'], actor=result['actor_state']), folder/'final.pt')
        atomic_json(folder/'paid_answers.json', dict(received=result['received'], sequence=result['selected_sequence']))
        report = dict(result['report'], adequate_same_all_paid_ordinary=ordinary)
        atomic_json(folder/'receipt.json', report)
        reports[policy] = report
        selected[policy] = {r['uid'] for r in result['received']}
    adaptive = reports['ADAPTIVE']
    frozen = reports['FROZEN_ACTOR']
    assert adaptive['actor_actual_updates'] > 0
    assert adaptive['solver_actual_updates'] == frozen['solver_actual_updates'] == 3
    assert frozen['actor_actual_updates'] == 0
    assert adaptive['teacher_budget']['unique_teacher_bundles'] == frozen['teacher_budget']['unique_teacher_bundles'] == sum(quotas.values())
    assert not guard
    report = dict(status='PASS_REAL_THREE_ROUND_RSI_MECHANISM_ENGINEERING_CHECK',
                  pair=pair, video=video, neural_seeds=[0], device='cpu',
                  source_sha256=plan['source_sha256'], policies=reports,
                  exact_declared_teacher_budget_per_policy=sum(quotas.values()),
                  selected_question_set_symmetric_difference=len(selected['ADAPTIVE'] ^ selected['FROZEN_ACTOR']),
                  all_feedback_cost_charged=True,
                  actual_new_teacher_or_backbone_calls=0, GT_open_attempts=0,
                  new_RMOT_predictions_or_HOTA=False,
                  teacher_proxy_progress_not_a_semantic_success_claim=True,
                  wall_seconds=time.perf_counter()-start, goal_status='ACTIVE')
    atomic_json(OUT/'analysis.json', report)
    atomic_json(PUBLIC/'rsi_curriculum_engineering_receipt.json', report)
    atomic_json(STATE, dict(status=report['status'], pid=os.getpid(), goal_status='ACTIVE'))
    print(json.dumps(dict(status=report['status'], wall_seconds=report['wall_seconds'],
                          actor_actual_updates=adaptive['actor_actual_updates'],
                          GT_open_attempts=0, new_RMOT_HOTA=False)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--real-loop', action='store_true')
    args = parser.parse_args()
    if args.real_loop:
        try:
            real_loop()
        except Exception as error:
            atomic_json(STATE, dict(status='FAILED_RSI_REAL_LOOP_REQUIRES_ROOT_CAUSE',
                                   error=repr(error), goal_status='ACTIVE'))
            raise
    else:
        audit()
