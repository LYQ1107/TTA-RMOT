"""Publish source-only Stage A gate from already sealed official evaluations."""
from argparse import ArgumentParser
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flex_uda.presence.runtime import RUN, PUBLIC, IN_DOMAIN
from flex_tta.runtime.pf_receipts import atomic_json


def stage_a_gate():
    from scripts.multidomain_diagnostics import reusable, base
    from scripts.pf_diag_common import official_metrics
    rows = []
    for target, model in IN_DOMAIN.items():
        source_model = {'KITTI-V2': 'Dance-best', 'Dance': 'V2-best', 'SportsMOT': 'V2-best'}[target]
        native = reusable(target, 'IN_DOMAIN', source_model) or base(target, source_model) / 'IN_DOMAIN'
        baseline = official_metrics(native / 'results')
        folder = RUN / 'stage_A' / target / model
        for method in ('PRES_TOPK', 'PRES_TOPK_SMOOTH'):
            receipt = json.loads((folder / method / 'official_completion.json').read_text())
            metrics = receipt['metrics']; delta = metrics['HOTA']-baseline['HOTA']
            rows.append(dict(target=target, model=model, method=method, native_HOTA=baseline['HOTA'],
                             delta_HOTA=delta, **metrics))
    failures = [row['target'] for row in rows if row['method'] == 'PRES_TOPK' and row['delta_HOTA'] < -1]
    from scripts.diag_attrib_v2_common import write_csv
    write_csv(PUBLIC / 'stage_A_same_domain.csv', rows)
    result = dict(status='FAIL_SAME_DOMAIN_INVESTIGATE_BEFORE_BC' if failures else 'PASS_ALL_THREE_SAME_DOMAIN',
        failed_targets=failures, criterion='Each own-source PRES_TOPK HOTA >= native HOTA -1',
        metrics_path=str(PUBLIC / 'stage_A_same_domain.csv'), published_commit=None,
        target_GT_is_not_runtime_supervision=True)
    atomic_json(PUBLIC / 'stage_A_gate.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = ArgumentParser(); parser.add_argument('--stage-a-gate', action='store_true'); args = parser.parse_args()
    if args.stage_a_gate: stage_a_gate()
