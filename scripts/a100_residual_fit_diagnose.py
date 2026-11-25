"""Reproduce the failed ordinary C4 fit without opening task GT or HOTA.

Preserves the original model, labels, schedule, batches, rate and clipping.
Checkpoints include optimizer state so this diagnosis need not be repeated.
"""
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flex_uda.active.deterministic_observation_runtime import configure_before_cuda
configure_before_cuda()
import numpy as np
import torch
from flex_tta.runtime.pf_receipts import atomic_json, file_sha
from flex_uda.active.residual_rsi_learning import (
    data, fit, make, logits, paired_training_table, state, save, STEPS)
from scripts.a100_residual_rsi_primary import OUT as PARENT, PLAN, registration

OUT = PARENT/'fit_failure_diagnosis'


def measurement(model, x, y, weight, device):
    z = logits(model, x, device).astype('float64')
    h = -y*np.log(np.clip(y, 1e-15, 1))-(1-y)*np.log(np.clip(1-y, 1e-15, 1))
    per = np.logaddexp(0, z)-y*z-h
    with torch.no_grad():
        a = torch.as_tensor(x.copy(), device=device)
        raw = model.output(model.hidden((a-model.mean)/model.std)).flatten().cpu().numpy()
    order = np.argsort(weight*per)[::-1][:12]
    return dict(proper_teacher_excess_CE=float(np.sum(weight*per)/weight.sum()),
                raw_logit_min=float(raw.min()), raw_logit_max=float(raw.max()),
                numerically_saturated_count=int(np.sum(np.abs(z)>=7.999999)),
                worst=[dict(index=int(i), label=float(y[i]), logit=float(z[i]),
                            raw_logit=float(raw[i]), weight=float(weight[i]), KL=float(per[i])) for i in order])


def main():
    from flex_uda.a100.mechanism import generation_guard
    guard = generation_guard()
    plan = registration()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '3'
    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(.35)
    OUT.mkdir(exist_ok=True)
    case = next(c for c in plan['cases'] if c['pair']=='C4')
    features, query, doc, quotas, auxiliary, ay, mean, std = data(case)
    draws = json.loads((PARENT/'C4/primary/draws.json').read_text())['draws']
    model = make(mean, std, 'cuda:0')
    opt, _ = fit(model, auxiliary, ay, np.full(len(ay), 1/len(ay)), 500, 'cuda:0')
    records = []
    for stage, steps in enumerate(STEPS):
        prefix = [r for r in draws if r['stage']<=stage]
        x, y, w = paired_training_table(features, query, prefix, quotas, auxiliary, ay, 'SOFT_KD')
        opt, receipt = fit(model, x, y, w, steps, 'cuda:0', opt)
        record = dict(stage=stage, total_steps=500+sum(STEPS[:stage+1]), **measurement(model,x,y,w,'cuda:0'))
        records.append(record)
        print(json.dumps(record), flush=True)
        save(OUT/f'phase_{stage}.pt', dict(model=state(model), optimizer=opt.state_dict(), record=record))
    total = 6000
    while total<16000:
        opt, _ = fit(model, x, y, w, 500, 'cuda:0', opt)
        total += 500
        record = dict(total_steps=total, **measurement(model,x,y,w,'cuda:0'))
        records.append(record)
        print(json.dumps({k:v for k,v in record.items() if k!='worst'}), flush=True)
    save(OUT/'failed_fit_16000.pt',dict(model=state(model),optimizer=opt.state_dict(),record=record))
    assert not guard
    result = dict(status='COMPLETE_ORIGINAL_C4_SOFT_FIT_REPRODUCTION',
                  records=records, original_registration_sha256=file_sha(PLAN),
                  self_source_sha256=file_sha(__file__), GT_open_attempts=0,
                  no_HOTA_or_checkpoint_selection=True)
    atomic_json(OUT/'diagnosis.json', result)
    print(json.dumps(dict(status=result['status'],final=record)),flush=True)


if __name__=='__main__':
    main()
