"""Learn visual correspondences as a training constraint, not new true labels.

The external teacher responses are immutable. A small observer learns which
image/state correspondences help predict held teacher responses after a
virtual student update. This is finite, RSI-inspired feedback; the observer is
not Qwen and noisy teacher probes are not a clean validation set.
"""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.func import functional_call

from flex_tta.runtime.pf_receipts import atomic_json, file_sha
from flex_uda.active.appearance_learning_inputs import DIMENSION
from flex_uda.active.joint_teacher_noise import fit_solver, risk_floor

BOOTSTRAP_STEPS = 500
META_STEPS = 1500
FINAL_STEPS = 4000
EDGE_BATCH = 256
META_EVERY = 10
GRAPH_STRENGTH = .1
VIRTUAL_RATE = .001


class Solver(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(DIMENSION, 128), nn.SiLU(),
                                 nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 1))

    def forward(self, value):
        return self.net(value).flatten()


def make_solver(device):
    torch.manual_seed(0)
    return Solver().to(device)


class Observer(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(10, 32), nn.Tanh(), nn.Linear(32, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, values):
        return 2*self.net(values).flatten().sigmoid()


def probe_mask(keys, target_video):
    """Auxiliary whole-track groups only; main-video guidance is never split."""
    groups = sorted({(str(k[0]), int(k[2])) for k in keys if k[0] != target_video},
                    key=lambda g: hashlib.sha256(json.dumps(g).encode()).hexdigest())
    assert len(groups) >= 5
    chosen = set(groups[:max(1, len(groups)//5)])
    mask = np.asarray([(str(k[0]), int(k[2])) in chosen for k in keys])
    assert mask.any() and (~mask).any()
    assert all(k[0] != target_video for k, hold in zip(keys, mask) if hold)
    return mask


def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def save(path, state):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    torch.save(state, temporary)
    temporary.replace(path)


def teacher_loss(model, x, target, parameters=None):
    loss = torch.zeros((), device=x.device)
    for begin in range(0, len(x), 512):
        value = (model(x[begin:begin+512]) if parameters is None else
                 functional_call(model, parameters, (x[begin:begin+512],)))
        loss = loss + nn.functional.binary_cross_entropy_with_logits(
            value, target[begin:begin+512], reduction='sum')/len(x)
    return loss


def edge_values(model, dense, graph, indices, device):
    left = graph['left'][indices]
    right = graph['right'][indices]
    x = torch.as_tensor(np.asarray(dense[np.r_[left, right]]), device=device)
    logits = model(x)
    first, last = logits.chunk(2)
    static = torch.as_tensor(graph['context'][indices], device=device)
    # Current solver feedback is detached: A cannot change its inputs to
    # manufacture agreement. S receives gradients through the relation loss.
    p, q = first.detach().sigmoid(), last.detach().sigmoid()
    dynamic = torch.stack((p, q, (p-q).abs(), 4*((p+q)/2)*(1-(p+q)/2)), 1)
    context = torch.cat((static, dynamic), 1)
    robust = nn.functional.huber_loss(first, last, reduction='none', delta=1.)
    base = torch.as_tensor(graph['weight'][indices], device=device)
    return robust, context, base


def train(features, probability, keys, target_video, dense, graph, folder,
          device, registration_sha256, data_sha256):
    """One registered complete method; no module or parameter grid."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    final = folder/'final.pt'
    if final.exists():
        saved = torch.load(final, map_location='cpu', weights_only=False)
        assert saved['registration_sha256'] == registration_sha256
        model = make_solver(device)
        model.load_state_dict(saved['model'])
        return model, saved
    started = time.perf_counter()
    target = np.nanmean(probability, axis=1)
    target_sha = hashlib.sha256(target.tobytes()).hexdigest()
    hold = probe_mask(keys, target_video)
    x = torch.as_tensor(features, device=device)
    y = torch.as_tensor(target, dtype=torch.float32, device=device)
    training = torch.as_tensor(~hold, device=device)
    probing = torch.as_tensor(hold, device=device)
    torch.manual_seed(0)
    model = make_solver(device)
    observer = Observer().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    actor_optimizer = torch.optim.Adam(observer.parameters(), lr=.001)
    checkpoint = folder/'closed.pt'
    rounds, first_step, bootstrap, initial_actor = [], 0, None, cpu_state(observer)
    generator = np.random.default_rng(0)
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        assert saved['registration_sha256'] == registration_sha256
        assert saved['data_sha256'] == data_sha256 and saved['teacher_target_sha256'] == target_sha
        model.load_state_dict(saved['model']); observer.load_state_dict(saved['observer'])
        optimizer.load_state_dict(saved['optimizer']); actor_optimizer.load_state_dict(saved['observer_optimizer'])
        generator.bit_generator.state = saved['generator_state']
        first_step, rounds, bootstrap = saved['steps'], saved['feedback'], saved['bootstrap']
        initial_actor = saved['initial_observer']
    else:
        bootstrap = fit_solver(model, features[~hold], target[~hold], device, steps=BOOTSTRAP_STEPS)
        print(json.dumps(dict(status='VISUAL_CONSTRAINT_BOOTSTRAP', case=folder.parent.name,
                              labels=int((~hold).sum()), probes=int(hold.sum()))), flush=True)
    normalizer = max(float(np.mean(graph['weight'])), 1e-8)
    for step in range(first_step, META_STEPS+FINAL_STEPS):
        indices = generator.integers(0, len(graph['left']), size=EDGE_BATCH)
        in_meta = step < META_STEPS
        if in_meta and step % META_EVERY == 0:
            model.zero_grad(set_to_none=True); observer.zero_grad(set_to_none=True)
            relation, context, base = edge_values(model, dense, graph, indices, device)
            weights = observer(context)
            support = teacher_loss(model, x[training], y[training])
            constraint = (weights*base*relation).mean()/normalizer
            parameters = dict(model.named_parameters())
            gradients = torch.autograd.grad(support+GRAPH_STRENGTH*constraint,
                                             tuple(parameters.values()), create_graph=True)
            virtual = {name: value-VIRTUAL_RATE*gradient for (name, value), gradient
                       in zip(parameters.items(), gradients)}
            before = teacher_loss(model, x[probing], y[probing]).detach()
            after = teacher_loss(model, x[probing], y[probing], virtual)
            # A weak magnitude prior keeps the observer from learning merely
            # to erase all image relations. It does not certify relation truth.
            objective = (after-before)/VIRTUAL_RATE+.01*(weights-1).square().mean()
            actor_gradient = torch.autograd.grad(objective, tuple(observer.parameters()))
            actor_optimizer.zero_grad(set_to_none=True)
            for parameter, gradient in zip(observer.parameters(), actor_gradient):
                parameter.grad = gradient
            actor_optimizer.step()
            rounds.append(dict(step=step, held_teacher_CE_before=float(before),
                               virtual_held_teacher_CE_after=float(after.detach()),
                               virtual_rate=VIRTUAL_RATE,
                               mean_relation_multiplier=float(weights.detach().mean()),
                               observer_gradient_norm=float(torch.sqrt(sum(g.detach().square().sum() for g in actor_gradient))),
                               noisy_external_teacher_probe_not_GT_or_accuracy=True))
        optimizer.zero_grad(set_to_none=True)
        relation, context, base = edge_values(model, dense, graph, indices, device)
        with torch.no_grad(): weights = observer(context)
        active_x, active_y = (x[training], y[training]) if in_meta else (x, y)
        support = teacher_loss(model, active_x, active_y)
        constraint = (weights*base*relation).mean()/normalizer
        (support+GRAPH_STRENGTH*constraint).backward()
        nn.utils.clip_grad_norm_(model.parameters(), 10.)
        optimizer.step()
        done = step+1
        if done % 500 == 0 or done == META_STEPS+FINAL_STEPS:
            with torch.no_grad(): risk = float(teacher_loss(model, x, y))
            saved = dict(model=cpu_state(model), observer=cpu_state(observer),
                         optimizer=optimizer.state_dict(), observer_optimizer=actor_optimizer.state_dict(),
                         steps=done, total_solver_steps=BOOTSTRAP_STEPS+done, feedback=rounds,
                         bootstrap=bootstrap, initial_observer=initial_actor,
                         generator_state=generator.bit_generator.state, teacher_CE=risk,
                         teacher_target_sha256=target_sha, data_sha256=data_sha256,
                         registration_sha256=registration_sha256, seed=0,
                         teacher_targets_unchanged=True, target_manual_GT_used=False)
            save(checkpoint, saved)
            print(json.dumps(dict(status='VISUAL_CONSTRAINT_LEARNING_PROGRESS', case=folder.parent.name,
                                  solver_steps=BOOTSTRAP_STEPS+done, feedback_updates=len(rounds),
                                  teacher_CE=risk, phase='META' if in_meta else 'ALL_TEACHER_FINAL')), flush=True)
    saved['observer_parameter_change'] = max(float((value-initial_actor[k]).abs().max())
                                            for k, value in cpu_state(observer).items())
    saved['attainable_teacher_floor'] = risk_floor(features, target, np.ones(len(target)))
    saved['teacher_excess_CE'] = saved['teacher_CE']-saved['attainable_teacher_floor']
    saved['wall_seconds'] = time.perf_counter()-started
    saved['same_pseudo_targets_not_a_pure_KD_objective'] = True
    save(final, saved)
    atomic_json(folder/'receipt.json', {k: v for k, v in saved.items() if k not in
                ('model', 'observer', 'optimizer', 'observer_optimizer', 'generator_state', 'initial_observer')})
    assert target_sha == hashlib.sha256(np.nanmean(probability, axis=1).tobytes()).hexdigest()
    return model.eval(), saved
