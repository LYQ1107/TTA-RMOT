"""Train only the new source presence head on sealed frozen feature sets."""
from argparse import ArgumentParser
from functools import partial
import json
from pathlib import Path
import random
import sys
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flex_uda.presence.runtime import RUN, PUBLIC, IN_DOMAIN
from flex_uda.presence.data import FeatureUnits, collate_units, head_loss
from flex_uda.presence.head import PresenceHead
from flex_tta.runtime.pf_receipts import atomic_json, append_json, file_sha


def epoch(model, loader, device, weight, optimizer=None):
    model.train(optimizer is not None)
    totals = np.zeros(4)
    with torch.set_grad_enabled(optimizer is not None):
        for batch in loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            output = model(**{k: batch[k] for k in ('pcd', 'margin', 'box', 'sent', 'padding_mask')})
            loss, bce, ce = head_loss(output, batch['count'], weight)
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite source presence head loss')
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            n = len(batch['count'])
            totals += np.array([float(loss.detach()), float(bce.detach()), float(ce.detach()), 1])*n
    return dict(loss=totals[0]/totals[3], BCE=totals[1]/totals[3], count_CE=totals[2]/totals[3], units=int(totals[3]))


def main():
    p = ArgumentParser(description=__doc__)
    p.add_argument('--target', choices=list(IN_DOMAIN), required=True)
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    from scripts.multidomain_runtime import guard
    access = guard()  # only the already-exported own-source numerical labels are needed
    name = IN_DOMAIN[a.target]
    train_root = RUN / 'features/source_train' / a.target / name
    val_root = RUN / 'features' / a.target / name
    train, val = FeatureUnits(train_root), FeatureUnits(val_root, validation=True)
    for root in (train_root, val_root):
        if not (root / 'export_receipt.json').exists(): raise ValueError('Incomplete source feature export')
    out = RUN / 'heads' / name; out.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda:0'); model = PresenceHead().to(device)
    weight = torch.tensor(min(10., (len(train)-train.positive_units)/train.positive_units), device=device)
    generator = torch.Generator().manual_seed(0)
    train_loader = DataLoader(train, batch_size=256, shuffle=True, generator=generator, num_workers=0,
        collate_fn=partial(collate_units, augment=True), pin_memory=True)
    val_loader = DataLoader(val, batch_size=256, shuffle=False, num_workers=0, collate_fn=collate_units, pin_memory=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    latest = out / 'checkpoint_latest.pt'; start_epoch = 0; best = float('inf'); stale = 0
    prior_wall = 0.
    if a.resume:
        saved = torch.load(latest, map_location=device, weights_only=False)
        model.load_state_dict(saved['head']); optimizer.load_state_dict(saved['optimizer']); scheduler.load_state_dict(saved['scheduler'])
        start_epoch, best, stale = saved['epoch'], saved['best'], saved['stale']
        torch.set_rng_state(saved['rng'].cpu()); generator.set_state(saved['loader_rng'].cpu())
        torch.cuda.set_rng_state(saved['cuda_rng'].cpu()); prior_wall = saved['wall_seconds']
    started = time.perf_counter()
    for e in range(start_epoch, 20):
        training = epoch(model, train_loader, device, weight, optimizer)
        validation = epoch(model, val_loader, device, weight)
        scheduler.step(); improved = validation['loss'] < best
        if improved: best, stale = validation['loss'], 0
        else: stale += 1
        row = dict(epoch=e+1, train=training, validation=validation, best=best, stale=stale,
            lr=optimizer.param_groups[0]['lr'], selected=improved, wall_seconds=prior_wall+time.perf_counter()-started)
        append_json(out / 'curve.jsonl', row); print(json.dumps(row), flush=True)
        state = dict(head=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
            epoch=e+1, best=best, stale=stale, target=a.target, model=name, seed=0,
            train_units=len(train), validation_units=len(val), pos_weight=float(weight),
            parameters=sum(p.numel() for p in model.parameters()), rng=torch.get_rng_state(),
            loader_rng=generator.get_state(), cuda_rng=torch.cuda.get_rng_state(), wall_seconds=row['wall_seconds'])
        temporary = latest.with_suffix('.partial'); torch.save(state, temporary); temporary.replace(latest)
        if improved: torch.save(state, out / 'checkpoint_best_source.pt')
        if stale >= 3: break
    torch.cuda.synchronize()
    result = dict(status='HEAD_SOURCE_TRAINING_COMPLETE', target=a.target, model=name,
        best_source_loss=best, epochs=e+1, train_units=len(train), validation_units=len(val),
        parameters=sum(p.numel() for p in model.parameters()), head_only=True, frozen_FlexHook_loaded=False,
        target_GT_open_attempts=len(access), wall_seconds=prior_wall+time.perf_counter()-started,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(), defaults_changed=False,
        checkpoint=str(out / 'checkpoint_best_source.pt'), checkpoint_sha256=file_sha(out / 'checkpoint_best_source.pt'))
    atomic_json(out / 'training_receipt.json', result)


if __name__ == '__main__': main()
