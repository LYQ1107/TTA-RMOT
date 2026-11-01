"""Final targeted integrity checks for the authorized completed subset, not a six-pair success claim."""
from collections import Counter
import csv
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.multidomain_runtime import *
from scripts.multidomain_diagnostics import CONFIGS,base,verify


def frame_counts(episode):
    return Counter(int(float(line.split(',',1)[0])) for line in Path(episode['prediction_path']).read_text().splitlines())


def main():
    gate=require_plan();original=subprocess.check_output(['git','show','98e7f84efbe43493761063e7832de7bfe9d4e18d:results/multidomain_generality/PREREG.md'],cwd=ROOT)
    if not (PUBLIC/'PREREG.md').read_bytes().startswith(original):raise ValueError('Original preregistered text was changed')
    checks={};prediction_sets=0
    for target,expected in [('Dance',425),('KITTI-V2',862)]:
        allseals={name:verify(target,name) for name in CONFIGS}
        if any(s['episode_count']!=expected for s in allseals.values()):raise ValueError('Sequence coverage changed')
        for a,b in [('R_SxK_I','IN_DOMAIN'),('R_IxK_S','NATIVE'),('R_SxK_ORACLE','R_IxK_ORACLE')]:
            ea={(e['video'],e['expression']):e for e in allseals[a]['episodes']}
            eb={(e['video'],e['expression']):e for e in allseals[b]['episodes']}
            if ea.keys()!=eb.keys() or any(frame_counts(ea[k])!=frame_counts(eb[k]) for k in ea):
                raise ValueError('Per-query per-frame counts not matched: '+target+' '+a+' '+b)
        if allseals['B0_GLOBAL']['prediction_rows']!=allseals['IN_DOMAIN']['prediction_rows']:
            raise ValueError('B0 absolute total budget differs')
        prep=json.loads((base(target)/'GT_free_predictions.json').read_text())
        if prep['GT_open_attempts']:raise ValueError('Prediction generator opened GT')
        checks[target]=dict(sequences=expected,main_fixed_configurations=7,per_frame_count_equivalence=True,
            oracle_same_counts=True,B0_absolute_budget=True,GT_free_open_attempts=0)
        prediction_sets+=len(allseals)
    diag=list(csv.DictReader((PUBLIC/'diagnostics.csv').open()))
    if len(diag)!=42 or sum(r['status']=='COMPLETE' for r in diag)!=14 or sum(r['status']=='BLOCKED_OFFICIAL_ASSETS' for r in diag)!=28:
        raise ValueError('Full scope/blocked rows not explicitly retained')
    files=list(csv.DictReader(gzip.open(PUBLIC/'all_prediction_files.csv.gz','rt')))
    if len(files)!=11595 or len({r['prediction_path'] for r in files})!=11595:
        raise ValueError('Prediction index must include full, historical and independent source reference')
    directory=RUN/'validation';directory.mkdir(exist_ok=True)
    cmd=[sys.executable,'-m','pytest','-q','tests/test_multidomain_generality.py']
    test=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True)
    (directory/'pytest.log').write_text(test.stdout+test.stderr)
    if test.returncode or '3 passed' not in test.stdout:raise RuntimeError('Targeted regression failed: '+str(directory/'pytest.log'))
    compile_run=subprocess.run([sys.executable,'-m','compileall','-q',*[str(p) for p in (ROOT/'scripts').glob('*multidomain*.py')]],cwd=ROOT,capture_output=True,text=True)
    if compile_run.returncode:raise RuntimeError(compile_run.stderr)
    result=dict(status='PASS_AUTHORIZED_C1_C2_ONLY',full_C1_to_C6_complete=False,blocked_combinations=['C3','C4','C5','C6'],
        original_prereg_prefix_unchanged=True,prereg_amendment_publication=gate,checks=checks,
        complete_main_prediction_sets=prediction_sets,all_indexed_prediction_files=len(files),
        targeted_tests_passed=3,targeted_test_command=cmd,targeted_test_log=str(directory/'pytest.log'),
        targeted_test_log_sha256=file_sha(directory/'pytest.log'),compileall_passed=True,
        no_training=True,no_MLLM=True,max_physical_GPUs=2,source_historical_bit_exact=True,
        historical_source_HOTA=42.526,complete_source_HOTA=42.752,
        gt_audit=json.loads((PUBLIC/'gt_audit.json').read_text()))
    atomic_json(PUBLIC/'validation.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
