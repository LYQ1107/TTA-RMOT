"""Three real source-cache optimizer steps per domain, isolated from all heads.

Functional GPU check only: no source validation/early stopping, no official
checkpoint overwrite, no target labels or teacher. It is not a trained result.
"""
import json
import os
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import torch
    import numpy as np
    from flex_uda.presence.runtime import RUN, PUBLIC, IN_DOMAIN
    from flex_uda.presence.head import PresenceHead
    from flex_uda.presence.data import FeatureUnits, collate_units, head_loss
    from scripts.multidomain_runtime import guard
    from scripts.select_gpu import query
    from flex_tta.runtime.pf_receipts import atomic_json, file_sha
    assigned = os.environ['CUDA_VISIBLE_DEVICES']
    gpu = next(g for g in query() if g['uuid'] == assigned)
    registered = json.loads((RUN / 'launchers/1793607153759691902/admission.json').read_text())['selected']
    if assigned not in {g['uuid'] for g in registered} or gpu['free_mib'] < 8192:
        raise RuntimeError('Real head smoke must fit on the same two registered GPUs')
    access = guard(); torch.manual_seed(0); device = torch.device('cuda:0')
    torch.cuda.set_per_process_memory_fraction(min(.9, (4 << 30)/torch.cuda.get_device_properties(0).total_memory))
    results = []; begin = time.perf_counter()
    for target, name in IN_DOMAIN.items():
        data = FeatureUnits(RUN / 'features/source_train' / target / name, shard_subdir='smoke')
        model = PresenceHead().to(device); optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
        pos_weight = min(10., (len(data)-data.positive_units)/data.positive_units)
        indices = np.linspace(0, len(data)-1, 256, dtype=int)
        values = []; started = time.perf_counter()
        for step in range(3):
            batch = collate_units([data[int(i)] for i in indices], augment=True)
            batch = {k:v.to(device) for k,v in batch.items()}
            output = model(**{k:batch[k] for k in ('pcd', 'margin', 'box', 'sent', 'padding_mask')})
            loss, _, ce = head_loss(output, batch['count'], torch.tensor(pos_weight, device=device))
            optimizer.zero_grad(set_to_none=True); loss.backward()
            norm = torch.stack([p.grad.square().sum() for p in model.parameters() if p.grad is not None]).sum().sqrt()
            if not torch.isfinite(loss) or not torch.isfinite(norm) or not norm > 0:
                raise RuntimeError('Real source head gradient/loss check failed')
            optimizer.step()
            values.append(dict(step=step+1, loss=float(loss.detach()), count_CE=float(ce.detach()),
                               gradient_norm=float(norm), positive_units=int((batch['count'] > 0).sum())))
        torch.cuda.synchronize()
        results.append(dict(target=target, model=name, steps=values, seconds=time.perf_counter()-started,
            parameters=sum(p.numel() for p in model.parameters()), batch_size=256,
            source_smoke_seal_path=str(RUN / 'features/source_train' / target / name / 'smoke_receipt.json'),
            source_smoke_seal_sha256=file_sha(RUN / 'features/source_train' / target / name / 'smoke_receipt.json')))
        print(json.dumps(results[-1]), flush=True)
        del model, optimizer, batch, output, loss
        torch.cuda.empty_cache()
    if access: raise RuntimeError('Head smoke opened raw target/source GT')
    atomic_json(PUBLIC / 'head_gpu_smoke.json', dict(status='THREE_REAL_SOURCE_HEAD_OPTIMIZER_SMOKES_PASS',
        records=results, total_seconds=time.perf_counter()-begin, physical_gpu=gpu['index'],
        peak_allocated_bytes=torch.cuda.max_memory_allocated(), target_GT_open_attempts=0,
        numerical_labels_are_own_source_train_only=True, official_heads_changed=False, teacher_calls=0,
        full_training_or_RMOT_result=False))


if __name__ == '__main__': main()
