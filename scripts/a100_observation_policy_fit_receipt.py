"""Reconcile final CE fields for I1's continued ordinary heads, on CPU only.

The frozen capacity helper reports the correct deployed KL; the caller also
retains older CE/entropy text. This appendix verifies the actual checkpoints
without changing the registered helper, weights, predictions or evaluation.
"""
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch import nn
from flex_uda.a100.mechanism import generation_guard
from scripts.a100_observation_design_quality import conditions, PUBLIC


def main():
    started = time.perf_counter(); guard = generation_guard(); torch.set_num_threads(1)
    rows = []; prefixes = passed = actual_fits = continued = 0
    for case in conditions():
        root = Path(case['folder']); document = json.loads((root/'completion.json').read_text())
        actual_fits += document['actual_head_fits']; continued += document['actual_capacity_calls']
        stream = None
        for index, receipt in enumerate(document['receipts']):
            for policy, fit in receipt['fits'].items():
                prefixes += 1; passed += bool(fit['teacher_fit_within_0_005'])
                if 'original_64_unit_teacher_KL' not in fit:
                    continue
                # This study retained only each policy's latest head. All
                # continued cases must be final prefixes to audit that head.
                assert index+1 == len(document['receipts'])
                if stream is None:
                    stream = torch.load(root/'stream.pt', map_location='cpu', weights_only=False)
                checkpoint = torch.load(root/(policy+'_latest_head.pt'), map_location='cpu', weights_only=False)
                x = np.concatenate(stream['bx'][policy]); raw_y = np.concatenate(stream['by'][policy])
                width, dimension = checkpoint['model']['0.weight'].shape
                model = nn.Sequential(nn.Linear(dimension, width), nn.GELU(), nn.Linear(width, 1))
                model.load_state_dict(checkpoint['model']); model.eval()
                xx = torch.as_tensor(((x-checkpoint['mean'])/checkpoint['std']).astype('float32'))
                target = torch.as_tensor(np.clip(raw_y, .02, .98).astype('float32')).double()
                weight = torch.as_tensor((1.+3.*np.abs(2.*raw_y-1.)).astype('float32')).double()
                with torch.no_grad():
                    logits = model(xx).squeeze(-1).double()
                entropy = float(weight@(-target*torch.log(target)-(1-target)*torch.log1p(-target))/weight.sum())
                ce = float(weight@torch.nn.functional.binary_cross_entropy_with_logits(logits, target, reduction='none')/weight.sum())
                kl = ce-entropy; error = abs(kl-fit['teacher_KL'])
                assert kl <= .005 and error < 1e-5, (case, policy, kl, error)
                rows.append(dict(pair=case['pair'], seed=case['seed'], video=receipt['video'], policy=policy,
                    prefix_videos=index+1, examples=len(raw_y), actual_architecture=f'{dimension}->{width} GELU->1',
                    inherited_original_CE=fit['teacher_CE'], inherited_original_entropy=fit['teacher_entropy'],
                    original_64_unit_teacher_KL=fit['original_64_unit_teacher_KL'],
                    reported_final_deployed_teacher_KL=fit['teacher_KL'],
                    independently_recomputed_CPU_CE=ce, independently_recomputed_entropy=entropy,
                    independently_recomputed_CPU_KL=kl, CPU_vs_GPU_KL_error=error,
                    original_teacher_fit_gate_pass=True))
    assert prefixes == passed == 1776 and len(rows) == continued == 12 and not guard
    report = dict(status='PASS_OBSERVATION_POLICY_FINAL_LOSS_RECEIPT_RECONCILIATION',
        ordinary_policy_prefix_receipts=prefixes, passed_original_KL_gate=passed,
        actual_head_fits=actual_fits, actual_capacity_continuations=continued, continued_heads=rows,
        explanation='Final deployed KL was correct and used for the gate. Inherited older CE/entropy/architecture fields are reconciled here without rewriting original records.',
        no_weight_prediction_or_metric_change=True, no_manual_GT_or_quality_predictions_read=True,
        new_teacher_or_training_calls=0, wall_seconds=time.perf_counter()-started, goal_status='ACTIVE')
    (PUBLIC/'observation_policy_fit_receipt_audit.json').write_text(json.dumps(report, separators=(',', ':'))+'\n')
    print(json.dumps(dict(status=report['status'], prefixes=prefixes, continued_heads=len(rows),
        actual_widths=sorted({r['actual_architecture'] for r in rows}),
        maximum_CPU_vs_GPU_KL_error=max(r['CPU_vs_GPU_KL_error'] for r in rows))))


if __name__ == '__main__':
    main()
