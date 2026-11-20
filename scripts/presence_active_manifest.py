"""Seal an explicit, already-published complete Stage A for the B/C workers."""
from argparse import ArgumentParser
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flex_uda.presence.runtime import RUN,PUBLIC,IN_DOMAIN
from flex_tta.runtime.pf_receipts import atomic_json,file_sha


def main():
    p=ArgumentParser(description=__doc__);p.add_argument('--publication',required=True)
    mode=p.add_mutually_exclusive_group()
    mode.add_argument('--fast-map-pilot-only',action='store_true')
    mode.add_argument('--fast-map-main-critical',action='store_true');a=p.parse_args()
    record=PUBLIC / 'redesign_v5';out=RUN / 'redesign_v5'
    gate_path=record / 'gate.json';gate=json.loads(gate_path.read_text())
    artifacts=record / 'artifacts.json';info=json.loads(artifacts.read_text())
    expected={(t,m) for t in IN_DOMAIN for m in IN_DOMAIN.values()}
    if gate['status']!='PASS_SAME_DOMAIN' or not gate['completed_full_A'] or gate['failed_targets']:
        raise PermissionError('Do not issue execution manifest before the complete gate')
    if {(r['target'],r['model']) for r in info['decoded_arrays']}!=expected:
        raise ValueError('The nine Stage-A cells are not complete')
    result=dict(version='v5',published_commit=a.publication,gate_path=str(gate_path),gate_sha256=file_sha(gate_path),
        artifacts_path=str(artifacts),artifacts_sha256=file_sha(artifacts),
        active_root=str(out / 'active'),stage_A_root=str(out / 'stage_A'),
        public_root=str(record / 'active'),launcher_root=str(out / 'active_launchers'),
        source_heads=info['source_heads'],probability_rule='unchanged source-calibrated multi-Bernoulli posterior',
        count_decision='expected IoU, not the probabilities used in the B/C prior',
        old_stage_A_gate_unchanged=True,teacher_revision_unchanged=True,
        original_BC_hyperparameters_unchanged=True,first_jobs='C1/C3 FULL rho=.3 first-video resumable pilots',
        maximum_GPUs=2,matrix_GPU_hour_gate=40)
    if a.fast_map_pilot_only or a.fast_map_main_critical:
        prereg=record / 'active_map_v1/PREREG.md'
        result.update(fast_mode='source_map_v1',pilot_only=True,maximum_pilot_videos=1,
            active_root=str(out / 'active_map_v1'),public_root=str(record / 'active_map_v1'),
            launcher_root=str(out / 'active_map_launchers'),prereg_path=str(prereg),prereg_sha256=file_sha(prereg),
            original_BC_hyperparameters_unchanged=False,
            registered_change='source-anchored MAP fast calibration with teacher-supported two-class evidence only',
            original_teacher_source_head_and_LoRA_unchanged=True,full_matrix_authorized=False)
    if a.fast_map_main_critical:
        prereg=record / 'active_map_v1_full/PREREG.md'
        result.update(pilot_only=False,maximum_pilot_videos=None,
            public_root=str(record / 'active_map_v1_full'),
            launcher_root=str(out / 'active_map_full_launchers'),
            prereg_path=str(prereg),prereg_sha256=file_sha(prereg),
            continuation_of=str(record / 'execution_map_v1.json'),
            first_jobs='C1/C3 complete FULL rho=.3, then four fixed critical controls per pair',
            registered_tier='MAIN_AND_CRITICAL_ONLY',
            allowed_registered_jobs=[dict(pair=pair,method=method,rho=.3)
                for pair in ('C1','C3') for method in ('FULL','MINUS_SLOW','UNIFORM_K',
                    'YESTRACK_BUDGET_RANDOM','YESTRACK_BUDGET_PRIOR')],
            resume_completed_pilot_videos=True,full_matrix_authorized=False)
    name=('execution_map_v1_full.json' if a.fast_map_main_critical else
          'execution_map_v1.json' if a.fast_map_pilot_only else 'execution.json')
    path=record / name
    if path.exists():
        if json.loads(path.read_text())!=result:raise ValueError('Do not replace an existing execution manifest')
    else:atomic_json(path,result)
    print(json.dumps(dict(path=str(path),sha256=file_sha(path))),flush=True)


if __name__=='__main__':main()
