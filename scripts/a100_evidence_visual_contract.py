"""Targeted GT-free checks for the failure-driven new learning objective."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from flex_uda.active.evidence_visual_learning import (
    make,bounded,CURRENT,content_loss,teacher_table,data)
from scripts.a100_residual_rsi_primary import registration as previous,OUT as PREVIOUS
from scripts.a100_evidence_visual_inputs import OUT as PIXEL


def main():
    from flex_uda.a100.mechanism import generation_guard
    from flex_uda.active.full_appearance_learning_inputs import FullAppearanceBank
    guard=generation_guard();torch.set_num_threads(2);plan=previous();records=[]
    raw=torch.tensor([-389.,-140.,0.,140.,389.],requires_grad=True)
    bounded(raw).sum().backward();assert torch.isfinite(raw.grad).all() and (raw.grad>0).all()
    derivative=raw.grad.detach().tolist()
    for case in plan['cases']:
        f,q,doc,quotas,auxiliary,ay,mean,std=data(case)
        bank=FullAppearanceBank(case['target'],case['evaluated_video'])
        chosen=np.arange(min(len(f),32));rows=[]
        for i in chosen:
            r=doc['actions'][doc['order'][i]]['case'];rows.append(bank.by_observation[int(r['frame_id']),int(r['track_id'])])
        assert np.array_equal(f[chosen,CURRENT],bank.normalized(np.asarray(rows)))
        with np.load(PIXEL/case['pair']/'pixel_pairs.npz') as p:
            pixel={k:torch.from_numpy(p[k].copy()) for k in ('left','right','weight')}
        s=make(mean,std,'cpu');r=make(mean,std,'cpu',outputs=2)
        ordinary=make(mean,std,'cpu',kind='plain',width=288)
        counts=[sum(p.numel() for p in m.parameters()) for m in (s,r,ordinary)]
        assert counts[2]>=counts[0]+counts[1]
        x=torch.from_numpy(f[chosen].copy());z=s(x)
        synthetic=torch.linspace(.05,.95,len(x))
        teacher=torch.nn.functional.binary_cross_entropy_with_logits(z,synthetic)
        pretext=content_loss(r,pixel);loss=teacher+.1*pretext;loss.backward()
        assert torch.isfinite(loss) and any(p.grad is not None and p.grad.abs().sum()>0 for p in s.parameters())
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in r.app.parameters())
        draws=json.loads((PREVIOUS/case['pair']/'primary/draws.json').read_text())['draws']
        _,targets,weight=teacher_table(f,q,draws,quotas,auxiliary,ay)
        assert np.array_equal(targets[:len(draws)],np.asarray([a['probability'] for a in draws]))
        assert ((targets>=0)&(targets<=1)).all() and (weight>=0).all()
        records.append(dict(pair=case['pair'],real_current_image_coordinates_exact=True,teacher_targets_exact=True,
            nonnegative_weights=True,independent_image_pretext_finite=True,
            actual_cpu_backward_two_models=True,optimizer_steps=0,parameter_counts=counts,
            synthetic_teacher_term=float(teacher.detach()),image_pretext=float(pretext.detach())))
    assert not guard
    result=dict(status='PASS_GT_FREE_NEW_VISUAL_LEARNING_CONTRACT',records=records,
        polynomialcap_derivative_at_real_failure_scale=derivative,
        no_GT_neural_optimizer_teacher_or_backbone_calls=True,
        backward_checks_not_semantic_quality_or_RSI_success=True,
        self_source_sha256=file_sha(__file__),goal_status='ACTIVE')
    atomic_json(ROOT/'a100_runs/visual_classical_20261121/supervisor/evidence_visual_contract_20261125.json',result)
    atomic_json(ROOT/'docs/a100_six_direction_20261122/evidence_visual_contract_20261125.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
