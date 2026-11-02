"""One explicit publication snapshot, not a polling agent/experiment loop."""
import json
from pathlib import Path
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flex_uda.presence.runtime import RUN, PUBLIC, IN_DOMAIN
from flex_tta.runtime.pf_receipts import atomic_json, file_sha


def main():
    smoke = []
    for target, model in IN_DOMAIN.items():
        for kind in ('target', 'source_train'):
            folder = RUN / 'features' / ('' if kind == 'target' else kind) / target / model
            path = folder / 'smoke_receipt.json'; result = json.loads(path.read_text())
            smoke.append(dict(target=target, model=model, kind=kind, path=str(path), sha256=file_sha(path), result=result))
    atomic_json(PUBLIC / 'smoke.json', dict(status='ALL_SIX_REAL_SMOKES_PASS', records=smoke,
        target_gt_read_by_target_exports=False, source_gt_is_legal_for_own_source_only=True))
    references = []
    for pair in ('C1', 'C6'):
        folder = RUN / 'active' / pair / 'GATE_REASSIGN_REF'
        if (folder / 'official_completion.json').exists():
            completion = json.loads((folder / 'official_completion.json').read_text())
            seal = json.loads((folder / 'predictions_seal.json').read_text())
            references.append(dict(pair=pair, metrics=completion['metrics'],
                path=str(folder), prediction_seal_sha256=file_sha(folder / 'predictions_seal.json'),
                detailed_sha256=completion['detailed_sha256'], logical_teacher_calls=seal['teacher']['logical_calls'],
                new_teacher_calls=0, exact_old_result_reuse=pair == 'C1',
                CPU_derived_using_same_old_gates=pair == 'C6'))
    atomic_json(PUBLIC / 'reference_controls.json', dict(records=references,
        current_method_results=False, inherited_gates_shared_not_double_counted=True,
        GT_only_after_prediction_seals=True))
    tests = subprocess.run([sys.executable, '-m', 'pytest', '-q', 'tests/test_presence_active.py'],
        cwd=ROOT, capture_output=True, text=True)
    atomic_json(PUBLIC / 'validation.json', dict(returncode=tests.returncode, stdout=tests.stdout, stderr=tests.stderr,
        tested_files={str(path.relative_to(ROOT)): file_sha(path) for root in ('flex_uda/presence', 'flex_uda/active')
                      for path in sorted((ROOT / root).glob('*.py'))},
        teacher_backend_sha256=file_sha(ROOT / 'flex_uda/teacher/yestrack_backend.py'),
        teacher_visual_builder_sha256=file_sha(ROOT / 'flex_uda/teacher/visual_input_builder.py'),
        full_RMOT_complete=False, teacher_called_by_new_method=False, LoRA_adaptation_started=False))
    if tests.returncode: raise RuntimeError(tests.stdout+tests.stderr)
    print(tests.stdout, flush=True)


if __name__ == '__main__': main()
