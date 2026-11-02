"""Proportionate numerical completion checks, not new method experiments."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.track_ranking_common import *


def main():
    require_plan();start=time.perf_counter();plan=json.loads((RUN/'plan.json').read_text())
    for r in plan['inputs']:checked(r)
    done=json.loads((RUN/'summary_completion.json').read_text());machine=done['machine_files']
    for r in machine.values():checked(r)
    q=pd.read_csv(machine['sequence_metrics.csv']['path'],dtype={'video':str})
    expected={'KITTI-V2':862,'Dance':425,'SportsMOT':466};videos={'KITTI-V2':4,'Dance':25,'SportsMOT':45}
    for target,n in expected.items():
        d=q[q.target==target]
        if len(d)!=n*3 or d.video.nunique()!=videos[target] or set(d.model)!=set(MODELS):raise ValueError('Incomplete target/model/query matrix')
        if d.groupby(['video','query']).size().ne(3).any():raise ValueError('Not exactly three official models per q')
        if d.groupby(['video','query']).common_AUROC_q.nunique().ne(1).any():raise ValueError('Model-dependent AUROC cohort')
    units=pd.read_csv(machine['units.csv']['path'],dtype={'video':str})
    if (units.observations<3).any() or not np.array_equal(units.Y.to_numpy(),(2*units.matched_observations>=units.observations).astype(int).to_numpy()):
        raise ValueError('Wrong track filter or majority label')
    var=(10*units.matched_observations>units.observations)&(10*units.matched_observations<9*units.observations)
    if not np.array_equal(units.time_varying.to_numpy(),var.astype(int).to_numpy()):raise ValueError('Wrong strict temporal-mixing bounds')
    frame_rows=0
    for name in ('auroc_frame.csv','auroc_segment.csv','auroc_track.csv'):
        count=0
        for part in pd.read_csv(machine[name]['path'],usecols=['objects','positives','AUROC'],chunksize=100000):
            valid=(part.positives>0)&(part.positives<part.objects)
            if not np.array_equal(valid.to_numpy(),part.AUROC.notna().to_numpy()):raise ValueError('Missing/misdefined binary AUROC units')
            if ((part.loc[valid,'AUROC']<0)|(part.loc[valid,'AUROC']>1)).any():raise ValueError('AUROC outside [0,1]')
            count+=len(part)
        if count!=machine[name]['rows']:raise ValueError('Combined CSV lost a row')
        if name=='auroc_frame.csv':frame_rows=count
    if frame_rows!=int(q.frames.sum()) or machine['auroc_segment.csv']['rows']!=len(q) or machine['auroc_track.csv']['rows']!=4*len(q):
        raise ValueError('Granularity matrix is incomplete')
    b=pd.read_csv(machine['bisect_track.csv']['path'])
    if len(b)!=4*len(q):raise ValueError('Incomplete four verifier/decoder controls')
    dense=b[(b.verifier=='ORACLE')&(b.method=='DENSE')]
    if not (dense.F1==1).all():raise ValueError('Perfect dense F1 must be one')
    for verifier in ('ORACLE','NOISY_SIMULATION'):
        dd=b[(b.verifier==verifier)&(b.method=='DENSE')]
        if not (dd.questions==dd.tracks).all() or dd.groupby(['target','video','query']).F1.nunique().ne(1).any():
            raise ValueError('Dense answers changed with the ranking model')
    bis=b[b.method=='BISECT_EXACT']
    if (bis.questions>np.ceil(np.log2(bis.tracks+1))).any():raise ValueError('Binary-search question bound violated')
    if (q.oracle_bisect_F1-q.TRACK_prefix>1e-12).any():raise ValueError('A prefix output exceeds the best-prefix ceiling')
    noise=json.loads((RUN/'noise_rates.json').read_text())
    if [noise['rates'][t]['candidate_pairs'] for t in TARGETS]!=[9180,13540,6426]:raise ValueError('Noise rates must use all original CROP probes')
    receipts=[]
    for target in TARGETS:
        d=json.loads((RUN/'targets'/target/'completion.json').read_text())
        for rec in d['video_receipts']:
            rr=json.loads(checked(rec).read_text());layout=rr['layout'];receipts.append(rr)
            if layout['raw_tracks']!=layout['retained_tracks']+layout['excluded_tracks']:raise ValueError('Excluded tracks not conserved')
            if layout['raw_observations']!=layout['retained_observations']+layout['excluded_observations']:raise ValueError('Excluded observations not conserved')
            if sum(layout['segment_observation_counts'])!=layout['retained_observations']:raise ValueError('Segments lost or duplicated observations')
    logs=RUN/'validation';logs.mkdir(exist_ok=True)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    tests=subprocess.run([sys.executable,'-m','pytest','-q','tests/test_track_ranking.py','tests/test_search_feasibility.py','tests/test_decompose.py'],
        cwd=ROOT,env=env,capture_output=True,text=True)
    (logs/'pytest.log').write_text(tests.stdout+tests.stderr)
    compile_result=subprocess.run([sys.executable,'-m','compileall','-q',*[str(p) for p in sorted((ROOT/'scripts').glob('*track_ranking*.py'))]],
        cwd=ROOT,env=env,capture_output=True,text=True)
    (logs/'compileall.log').write_text(compile_result.stdout+compile_result.stderr)
    if tests.returncode or compile_result.returncode:raise RuntimeError('Targeted regression failed: '+str(logs))
    result=dict(status='PASS',role=ROLE,tests_passed=int(re.search(r'(\d+) passed',tests.stdout).group(1)),compileall_passed=True,
        videos=len(receipts),sequences=sum(expected.values()),model_sequence_rows=len(q),frame_AUROC_rows=frame_rows,
        raw_machine_rows={k:r['rows'] for k,r in machine.items()},all_nine_margin_SHAs_unchanged=True,
        all_granularities_share_track_filter=True,segments_conserve_observations=True,
        track_labels_and_temporal_mixing_exact=True,common_q_is_model_independent=True,
        perfect_dense_F1_is_one=True,noisy_dense_answers_shared_across_models=True,
        all_binary_search_bounds_pass=True,prefix_upper_bounds_pass=True,
        new_teacher_calls=0,new_model_calls=0,GPU_hours=0,optimizer_updates=0,CPU_wall_seconds=time.perf_counter()-start,
        logs=[record(logs/'pytest.log'),record(logs/'compileall.log')])
    dump(PUBLIC/'validation.json',result)
    print(json.dumps(dict(event='TRACK_VALIDATION_PASSED',tests_passed=result['tests_passed'],sequences=result['sequences'],frame_rows=frame_rows)),flush=True)


if __name__=='__main__':main()
