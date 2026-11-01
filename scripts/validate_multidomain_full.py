"""Scientific integrity gates for all six pairs; no fitting or new inference."""
from collections import Counter
import csv
import gzip
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.multidomain_runtime import RUN,PUBLIC,OLD_SOURCE,require_plan,require_remaining_p0
from scripts.multidomain_diagnostics import CONFIGS,base,verify,inputs
from scripts.report_multidomain_generality import PAIRS
from flex_tta.runtime.pf_receipts import atomic_json,file_sha


def counts(episode):
    return Counter(int(float(s.split(',',1)[0])) for s in Path(episode['prediction_path']).read_text().splitlines() if s.strip())


def main():
    require_plan();p0=require_remaining_p0();checks={}
    original=subprocess.check_output(['git','show','98e7f84efbe43493761063e7832de7bfe9d4e18d:results/multidomain_generality/PREREG.md'],cwd=ROOT)
    if not (PUBLIC/'PREREG.md').read_bytes().startswith(original):raise ValueError('Original PREREG changed')
    for combo,source,target in PAIRS:
        meta,arrays,groups,_=inputs(target,source)
        seals={m:verify(target,m,source) for m in CONFIGS}
        for a,b in [('R_SxK_I','IN_DOMAIN'),('R_IxK_S','NATIVE'),('R_SxK_ORACLE','R_IxK_ORACLE')]:
            left={(x['video'],x['expression']):x for x in seals[a]['episodes']}
            right={(x['video'],x['expression']):x for x in seals[b]['episodes']}
            if left.keys()!=right.keys() or any(counts(left[k])!=counts(right[k]) for k in left):
                raise ValueError('Per-query/frame count mismatch: '+combo+' '+a)
        if seals['B0_GLOBAL']['prediction_rows']!=seals['IN_DOMAIN']['prediction_rows']:
            raise ValueError('Global budget mismatch')
        for name,array in zip(('NATIVE','IN_DOMAIN'),arrays):
            if len(array)!=meta['candidate_query_rows'] or not np.isfinite(array).all():
                raise ValueError('Full score coverage mismatch')
            if int(np.count_nonzero(array>0))!=seals[name]['prediction_rows']:
                raise ValueError('Native threshold/row mismatch')
        record=json.loads((base(target,source)/'GT_free_predictions.json').read_text())
        if record['GT_open_attempts']:raise PermissionError('Decoder target GT open')
        checks[combo]=dict(episodes=len(meta['episodes']),videos=len(meta['videos']),
            complete_score_rows=meta['candidate_query_rows'],fixed_main_configurations=7,
            per_query_frame_count_equivalence=True,global_budget=True,GT_open_attempts=0)
    masks={k:np.load(RUN/k/'historical.npy') for k in ('kitti','dance','sports')}
    source_seals=[]
    for key in ('kitti','dance','sports'):
        for p in (RUN/key).glob('*/scores_seal.json'):
            s=json.loads(p.read_text());full=np.load(s['complete_path'],mmap_mode='r');old=np.load(s['historical_path'],mmap_mode='r')
            if file_sha(s['complete_path'])!=s['complete_sha256'] or file_sha(s['historical_path'])!=s['historical_sha256']:
                raise ValueError('Changed score seal')
            if not np.array_equal(full[masks[key]],old) or s['GT_open_attempts']:
                raise ValueError('Historical subset/GT audit mismatch')
            source_seals.append(dict(target=key,model=s['model'],history_exact_subset=True,GT_open_attempts=0))
    source=np.load(RUN/'kitti/V2-best/complete.npy',mmap_mode='r')
    idx=np.load(RUN/'kitti/historical_indices.npy')
    if not np.array_equal(source[masks['kitti']],np.load(OLD_SOURCE/'FROZEN.npy',mmap_mode='r')[idx]):
        raise ValueError('Original V2 historical scores no longer bit-exact')
    gt=json.loads((PUBLIC/'sports_GT_completion.json').read_text())
    if gt['status']!='PASS_418_EXACT_48_COMPLETED' or gt['exact_passed']!=418 or gt['exact_failed']:
        raise ValueError('GT completion exact gate not passed')
    for row in gt['files']:
        if file_sha(row['path'])!=row['sha256']:raise ValueError('Evaluation GT changed')
    diag=list(csv.DictReader((PUBLIC/'diagnostics.csv').open()))
    if len(diag)!=42 or any(x['status']!='COMPLETE' for x in diag):
        raise ValueError('Do not omit a fixed comparison')
    files=list(csv.DictReader(gzip.open(PUBLIC/'all_prediction_files.csv.gz','rt')))
    if len({x['prediction_path'] for x in files})!=len(files):
        raise ValueError('Duplicate prediction index path')
    for row in files:
        if file_sha(row['prediction_path'])!=row['prediction_sha256']:
            raise ValueError('Prediction changed after sealing')
    directory=RUN/'validation_full';directory.mkdir(exist_ok=True)
    command=[sys.executable,'-m','pytest','-q','tests/test_multidomain_generality.py','tests/test_multidomain_download.py']
    test=subprocess.run(command,cwd=ROOT,capture_output=True,text=True)
    (directory/'pytest.log').write_text(test.stdout+test.stderr)
    if test.returncode:raise RuntimeError('Targeted regression failed')
    compiled=subprocess.run([sys.executable,'-m','compileall','-q',*[str(p) for p in (ROOT/'scripts').glob('*multidomain*.py')]],cwd=ROOT,capture_output=True,text=True)
    if compiled.returncode:raise RuntimeError(compiled.stderr)
    result=dict(status='PASS_ALL_SIX_PREREGISTERED_PAIRS',full_C1_to_C6_complete=True,original_PREREG_prefix_unchanged=True,
        remaining_p0_publication=p0,checks=checks,score_seals=source_seals,
        complete_main_prediction_sets=42,all_indexed_prediction_files=len(files),
        Sports_GT_original_exact=418,Sports_GT_completed_missing=48,
        targeted_test_command=command,targeted_test_output=test.stdout.strip(),
        targeted_test_log=str(directory/'pytest.log'),targeted_test_log_sha256=file_sha(directory/'pytest.log'),
        compileall_passed=True,no_training=True,no_MLLM=True,max_physical_GPUs=2,
        original_source_historical_scores_bit_exact=True,gt_audit=json.loads((PUBLIC/'gt_audit.json').read_text()))
    atomic_json(PUBLIC/'validation.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
