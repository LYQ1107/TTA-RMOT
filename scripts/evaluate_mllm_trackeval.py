#!/usr/bin/env python3
"""Independent parallel official TrackEval jobs; no training/model imports."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json

METHODS=['FROZEN','SOURCE_REHEARSAL','MLLM_DISTILL','FROZEN_AT_MLLM_COUNT','MLLM_AT_FROZEN_COUNT']

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-root',type=Path,required=True);p.add_argument('--max-jobs',type=int,default=4);a=p.parse_args()
    root=a.output_root.resolve();children=[];started=time.perf_counter()
    if not (root/'controls_sealed.json').exists():raise ValueError('all GT-free predictions/controls must be sealed before official GT access')
    registered=json.loads((root/'controls_sealed.json').read_text());methods=registered.get('methods',METHODS)
    for method in methods:
        audit=json.loads((root/method/'full_audit.json').read_text())
        if audit['status']!='PASS' or audit['valid_episodes']!=425:raise ValueError('all425 predictions required')
    def run_method(method):
        job=root/'trackeval_jobs'/method;job.mkdir(parents=True,exist_ok=True);link=job/method
        if not link.exists():link.symlink_to(root/method,target_is_directory=True)
        log=(job/'driver_stdout.txt').open('w')
        command=[sys.executable,str(ROOT/'scripts/evaluate_flexhook_dance_tta.py'),'--output-root',str(job),
            '--dataset-root','/data2/user/TTA-RMOT-assets/flexhook/refer-dance-official/Refer-Dance',
            '--vendor-root',str(ROOT/'vendor/FlexHook'),'--methods',method]
        process=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        code=process.wait();log.close()
        return dict(method=method,exit_code=code)
    results=[]
    with ThreadPoolExecutor(max_workers=a.max_jobs) as pool:
        for future in as_completed([pool.submit(run_method,method) for method in methods]):
            result=future.result();results.append(result)
            print(json.dumps(dict(event='official_TrackEval_finished',**result)),flush=True)
    atomic_json(root/'official_evaluation_completion.json',dict(status='COMPLETE' if all(r['exit_code']==0 for r in results) else 'FAILED_RETAINED',
        methods=results,wall_seconds=time.perf_counter()-started,GT_processes_independent_from_student=True))
    if any(r['exit_code'] for r in results):raise RuntimeError('inspect first actionable official evaluation error; predictions remain unchanged')
