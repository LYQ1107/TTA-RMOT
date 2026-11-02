"""Once-only completed-trial integrity/coverage check, not a process monitor."""
import json
from pathlib import Path
import subprocess
import sys
from argparse import ArgumentParser

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from flex_uda.presence.runtime import target_metadata, IN_DOMAIN
from flex_uda.presence.predictions import verify_predictions
from flex_tta.runtime.pf_receipts import file_sha, atomic_json
from scripts.presence_redesign import OUT, RECORD
from flex_uda.presence.runtime import RUN,PUBLIC


def main():
    global OUT,RECORD
    p=ArgumentParser(description=__doc__);p.add_argument('--version',choices=['v3','v4','v5'],default='v3')
    a=p.parse_args();OUT=RUN / ('redesign_'+a.version);RECORD=PUBLIC / ('redesign_'+a.version)
    choice = json.loads((RECORD / 'selection.json').read_text())['models']
    rows = []
    gate=json.loads((RECORD / 'gate.json').read_text())
    jobs=list(IN_DOMAIN.items()) if not gate['completed_full_A'] else [(t,m) for t in IN_DOMAIN for m in IN_DOMAIN.values()]
    for target, name in jobs:
        meta = target_metadata(target)
        expected = {(q['video_id'],q['expression_id']) for q in meta['episodes']}
        allowed = {v:set(bank['lines']) for v,bank in meta['videos'].items()}
        if file_sha(choice[name]['checkpoint']) != choice[name]['checkpoint_sha256']:
            raise ValueError('Source-selected checkpoint changed')
        for method in ('PRES_TOPK','PRES_TOPK_SMOOTH'):
            folder = OUT / 'stage_A' / target / name / method
            seal = verify_predictions(folder)
            if {(q['video'],q['expression']) for q in seal['episodes']} != expected:
                raise ValueError('Incomplete episode coverage')
            if seal['head_checkpoint_sha256'] != choice[name]['checkpoint_sha256']:
                raise ValueError('Wrong checkpoint in prediction seal')
            observed = 0
            for q in seal['episodes']:
                with Path(q['prediction_path']).open() as handle:
                    for line in handle:
                        if line.rstrip('\n') not in allowed[q['video']]:
                            raise ValueError('Changed/foreign MOT frame, ID or box')
                        observed += 1
            if observed != seal['prediction_rows']: raise ValueError('Prediction row count mismatch')
            rows.append(dict(target=target,model=name,method=method,videos=len(meta['videos']),
                episodes=len(expected),prediction_rows=observed,all_lines_original_MOT=True,
                GT_open_attempts=seal['GT_open_attempts'],seal_sha256=file_sha(folder / 'predictions_seal.json')))
    protected = ['results/presence_active/REPORT.md','vendor/FlexHook',
                 'flex_uda/teacher/yestrack_backend.py','flex_uda/teacher/visual_input_builder.py']
    changed = subprocess.check_output(['git','diff','--name-only',
        'efce389200aa5b93d47f6ef387f7c54a5e8ef890','--',*protected],cwd=ROOT,text=True).splitlines()
    if changed: raise ValueError('Historical report, official vendor or teacher unexpectedly changed')
    if a.version in ('v4','v5'):
        old=subprocess.check_output(['git','diff','--name-only','77cd790f1a1a9cb4fe581f31d51b20af965dc436',
            '--','results/presence_active/redesign_v3'],cwd=ROOT,text=True)
        if old.strip():raise ValueError('Do not modify v3 historical results')
    if a.version=='v5':
        old=subprocess.check_output(['git','diff','--name-only','f0fc286893fc97821d502f59c5b9d75620776b01',
            '--','results/presence_active/redesign_v4'],cwd=ROOT,text=True)
        if old.strip():raise ValueError('Do not modify v4 historical results')
    names = {'presence_train.py','presence_redesign.py','presence_eval_offline.py','active_eval.py',
             'active_run.py','active_teacher_worker.py','presence_bernoulli.py'}
    running = []
    for directory in Path('/proc').iterdir():
        if not directory.name.isdigit(): continue
        try:
            argv = (directory / 'cmdline').read_bytes().split(b'\0')
            cwd = (directory / 'cwd').resolve()
        except (OSError,ProcessLookupError): continue
        if cwd != ROOT: continue
        if len(argv)>1 and Path(argv[1].decode(errors='replace')).name in names:
            running.append(dict(pid=int(directory.name),script=argv[1].decode(errors='replace')))
    if running: raise RuntimeError('Wait for this task own workers; do not kill: '+str(running))
    result = dict(status='PASS_COMPLETE_SOURCE_HEAD_TRIAL',cells=rows,
        protected_historical_report_vendor_and_teacher_unchanged=True,
        new_teacher_calls=0,target_LoRA_updates=0,task_scientific_workers_remaining=running,
        tests_passed=json.loads((RECORD / 'functional_checks.json').read_text())['tests_passed'],
        stage_A_scientific_gate=json.loads((RECORD / 'gate.json').read_text())['status'],
        integrity_is_not_scientific_success=True,validation_no_new_model_or_teacher_calls=True)
    atomic_json(RECORD / 'validation.json',result)
    print(json.dumps(result),flush=True)


if __name__ == '__main__':main()
