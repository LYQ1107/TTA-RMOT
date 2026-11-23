"""A complete three-round paid-feedback learning loop for one target video.

Past/current data only; the caller supplies this video's unlabelled normalizer.
All probe groups remain out of solver fitting until a final common-data fit.
Actor sampling state and actual student snapshots are returned for resumption.
"""
import time

import numpy as np
import torch

from flex_uda.active.joint_observation_study import fit_prefix, make_model, prefix_arrays
from flex_uda.active.rsi_curriculum import (
    ProgressChallenger, TeacherReplayBroker, challenger_features,
    reserve_probes, select_round, update_challenger)
from flex_uda.active.rsi_observation_inputs import predict_joint, probe_cross_entropy


def initial_solver(recipe, dimension, mean, std, previous, device):
    torch.manual_seed(0)
    width = previous['width'] if previous else 64
    model = make_model(recipe, dimension, width).to(device).eval()
    if previous:
        assert previous['recipe'] == recipe
        model.load_state_dict(previous['model'])
        model.renormalize(previous['mean'], previous['std'], mean, std)
    return model


def run_video_curriculum(visual, actions, quotas, collection, mean, std, *,
                         past_records=(), previous=None, actor_state=None,
                         policy='ADAPTIVE', device='cpu', callback=None):
    assert policy in ('ADAPTIVE', 'FROZEN_ACTOR')
    started = time.perf_counter()
    reservation = reserve_probes(actions, quotas)
    broker = TeacherReplayBroker(collection, visual.video.pair, visual.video.video,
                                  actions, quotas, reservation)
    recipe = dict(family='structured', conditional=True, reference=True, policy=policy)
    model = initial_solver(recipe, visual.video.dimension, mean, std, previous, device)
    order = sorted(actions)
    raw = visual.arrays(order)
    logq = predict_joint(model, raw, mean, std, device)
    initial_features = challenger_features(logq, raw, actions, [])
    torch.manual_seed(0)
    actor = ProgressChallenger(initial_features.shape[-1]).cpu()
    generator = torch.Generator(device='cpu').manual_seed(0)
    baseline = 0.
    if actor_state:
        actor.load_state_dict(actor_state['model'])
        generator.set_state(actor_state['sampling_generator'])
        baseline = actor_state['baseline']
    optimizer = torch.optim.Adam(actor.parameters(), lr=.03)
    if actor_state:
        optimizer.load_state_dict(actor_state['optimizer'])
    training, received, receipts, solver_checkpoints = [], [], [], []
    for stage in range(3):
        stamp = time.perf_counter()
        logq = predict_joint(model, raw, mean, std, device)
        features = challenger_features(logq, raw, actions, [r['uid'] for r in received])
        chosen, log_policy = select_round(actor, features, actions, quotas, reservation,
                                           [r['uid'] for r in received], stage, generator)
        rows = broker.request(chosen, role='train', stage=stage)
        training.extend(rows)
        received.extend(rows)
        # Action choices precede access to this round's new teacher feedback.
        probe = broker.request(reservation['probes'][stage], role='probe', stage=stage)
        received.extend(probe)
        before = probe_cross_entropy(model, probe, visual, mean, std, device)
        record = visual.teacher_dataset(training)
        # Fresh probe query/track groups are absent from ALL this video's
        # ordinary solver training updates, including previous rounds.
        train_groups = {(r['local_query'], r['local_track']) for r in training}
        assert not train_groups.intersection(reservation['groups'])
        data = prefix_arrays([*past_records, record], mean, std)
        before_parameters = torch.cat([p.detach().cpu().flatten().double() for p in model.parameters()])
        model, _, checkpoint, fitted = fit_prefix(data, recipe, mean, std, previous=previous, device=device)
        after_parameters = torch.cat([p.detach().cpu().flatten().double() for p in model.parameters()])
        previous = checkpoint
        after = probe_cross_entropy(model, probe, visual, mean, std, device)
        update = update_challenger(actor, optimizer, log_policy, before, after, baseline,
                                    frozen=policy == 'FROZEN_ACTOR')
        baseline = update['baseline_after']
        receipt = dict(stage=stage, selected_new_training=len(chosen),
                       fresh_probe_questions=len(probe), training_bundles=len(training),
                       solver_parameter_L2_change=float((after_parameters-before_parameters).norm()),
                       actor_update=update, solver_fit=fitted,
                       paid_teacher=broker.receipt(), wall_seconds=time.perf_counter()-stamp,
                       target_GT_or_RMOT_quality_used=False)
        receipts.append(receipt)
        solver_checkpoints.append(checkpoint)
        state = dict(model={k: v.detach().cpu().clone() for k, v in actor.state_dict().items()},
                     sampling_generator=generator.get_state(), baseline=baseline,
                     optimizer=optimizer.state_dict())
        if callback:
            callback(stage, receipt, dict(solver=checkpoint, actor=state,
                training_UIDs=[r['uid'] for r in training], received_UIDs=[r['uid'] for r in received],
                teacher_sequence=broker.sequence))
    # The final method and adequate ordinary controls can fit every paid
    # response, including probes. This final fit has no further actor reward.
    common = visual.teacher_dataset(received)
    data = prefix_arrays([*past_records, common], mean, std)
    model, early, checkpoint, final_fit = fit_prefix(data, recipe, mean, std, previous=previous, device=device)
    state = dict(model={k: v.detach().cpu().clone() for k, v in actor.state_dict().items()},
                 sampling_generator=generator.get_state(), baseline=baseline,
                 optimizer=optimizer.state_dict())
    report = dict(policy=policy, stages=receipts, final_all_paid_fit=final_fit,
                  teacher_budget=broker.receipt(complete=True),
                  probe_groups=len(reservation['groups']), fresh_probe_rounds=reservation['fresh_probe_rounds'],
                  requested_probe_count=reservation['requested_probe_count'],
                  actual_probe_count=reservation['actual_probe_count'],
                  actor_actual_updates=sum(r['actor_update']['actor_parameter_L2_change'] > 0 for r in receipts),
                  solver_actual_updates=sum(r['solver_parameter_L2_change'] > 0 for r in receipts),
                  neural_seed=0, no_new_teacher_or_backbone_calls=True,
                  teacher_library_not_deployment_cost=True,
                  teacher_probe_reused_for_final_training_only_after_last_actor_update=True,
                  source_GroundTruth_or_HOTA_used=False, wall_seconds=time.perf_counter()-started)
    return dict(model=model, early=early, checkpoint=checkpoint, actor_state=state,
                record=common, received=received, selected_sequence=broker.sequence,
                round_solver_checkpoints=solver_checkpoints, report=report)
