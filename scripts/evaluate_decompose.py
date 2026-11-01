"""Official TrackEval and exact metric reuse, only in this offline GT process."""
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decompose_common import *
from scripts.decompose_oracle import assert_same


def reuse_path(combo,name,target,model):
    if name=='G4_F_FRAME':return previous.base(target,model)/'R_SxK_ORACLE'
    if combo=='C1':
        old={'G2_V_MEAN':'ORACLE__V_MEAN','G3_F_MEAN':'ORACLE__F_MEAN','TRACK_ORACLE':'ORACLE__TRACK_ORACLE',
             'C1_TRACK_MEAN':'FROZEN__C1_TRACK_MEAN'}
        if name in old:return COUNT/'evaluation'/old[name]
    return None


def evaluate(combo,name):
    require_plan();_,model,target=next(p for p in PAIRS if p[0]==combo)
    gate=json.loads((RUN/'oracle'/f'{combo}.json').read_text())
    if gate['status']!='ORACLE_SEALED_G4_EXACT_ROWS_PASS':raise ValueError('Pass mandatory row equality before any evaluation')
    s=verify(combo,name);out=folder(combo,name);done=out/'official_completion.json'
    if done.exists():
        r=json.loads(done.read_text())
        if file_sha(out/'results/pedestrian_detailed.csv')!=r['detailed_sha256']:raise ValueError('Changed official result')
        return r
    from scripts.pf_diag_common import official_metrics
    start=time.perf_counter();old=reuse_path(combo,name,target,model);reused=False
    if old is not None:
        assert_same(s,old)
        old_done=json.loads((old/'official_completion.json').read_text())
        if target=='SportsMOT' and old_done.get('GT_protocol')!=previous.SPORTS_GT_PROTOCOL:
            raise ValueError('Cannot mix SportsMOT protocols')
        for f,key in [('pedestrian_summary.txt','summary_sha256'),('pedestrian_detailed.csv','detailed_sha256')]:
            if file_sha(old/'results'/f)!=old_done[key]:raise ValueError('Changed reused official evaluation')
            shutil.copy2(old/'results'/f,out/'results'/f)
        reused=True
    else:
        root=previous.target_root(target)
        for e in s['episodes']:
            p=out/'results'/e['video']/e['expression']/'gt.txt'
            if not p.exists():p.symlink_to(root/'gt_template'/e['video']/e['expression']/'gt.txt')
        seq={'KITTI-V2':'kitti-2','Dance':'dance','SportsMOT':'SportsMOT'}[target]
        command=[sys.executable,'./TrackEval/scripts/run_mot_challenge.py','--METRICS','HOTA','CLEAR','Identity',
            '--SEQMAP_FILE',str(previous.VENDOR/'seqmaps'/(seq+'.txt')),'--SKIP_SPLIT_FOL','True','--GT_FOLDER',str(root),
            '--TRACKERS_FOLDER',str(out/'results'),'--GT_LOC_FORMAT','{gt_folder}{video_id}/{expression_id}/gt.txt',
            '--TRACKERS_TO_EVAL',str(out/'results'),'--USE_PARALLEL','True','--NUM_PARALLEL_CORES','2','--PLOT_CURVES','False']
        with (out/'trackeval.log').open('w') as h:
            code=subprocess.call(command,cwd=previous.VENDOR,stdout=h,stderr=subprocess.STDOUT)
        if code:raise RuntimeError('Official TrackEval failed: '+str(out/'trackeval.log'))
    m=official_metrics(out/'results')
    if int(m['Dets'])!=s['prediction_rows']:raise ValueError('Official evaluation differs from sealed row count')
    r=dict(status='PASS',combo=combo,method=name,target=target,source=model,metrics=m,
           prediction_seal_sha256=file_sha(out/'predictions_seal.json'),reused_exact_raw_rows=reused,
           reused_from=str(old) if reused else None,wall_seconds=time.perf_counter()-start,
           GT_protocol=previous.SPORTS_GT_PROTOCOL if target=='SportsMOT' else 'OFFICIAL_UNCHANGED',
           summary_sha256=file_sha(out/'results/pedestrian_summary.txt'),detailed_sha256=file_sha(out/'results/pedestrian_detailed.csv'))
    atomic_json(done,r)
    print(json.dumps(dict(event='DECOMPOSE_OFFICIAL_DONE',combo=combo,method=name,HOTA=m['HOTA'],
                         rows=s['prediction_rows'],seconds=r['wall_seconds'],reused=reused)),flush=True)
    return r


def main(combos):
    jobs=[(c,n) for c in combos for n in (*ORACLES,*CALIBRATORS)]
    results=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for f in as_completed([pool.submit(evaluate,c,n) for c,n in jobs]):results.append(f.result())
    atomic_json(RUN/'evaluation'/('_'.join(combos)+'.json'),dict(status='ALL_REQUESTED_OFFICIAL_EVALUATIONS_COMPLETE',methods=results))


if __name__=='__main__':
    p=ArgumentParser();p.add_argument('--combos',nargs='+',choices=[r[0] for r in PAIRS],required=True);a=p.parse_args();main(a.combos)
