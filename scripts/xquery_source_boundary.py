"""Source boundary: seal native/X0/X2/X4, then independent official TrackEval."""
from concurrent.futures import ThreadPoolExecutor,as_completed
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.xquery_common import *

SOURCE_METHODS=('FROZEN','T2V_D1_s17');RULES=('NATIVE','X0_C3','X2_C3_Z','X4_C3_Z_REJECT')

def generate():
    setup();require_plan();attempts=v2.deny_gt();start=time.perf_counter();dest=RUN/'source';meta=json.loads((dest/'metadata.json').read_text())
    thresholds=json.loads((RUN/'thresholds.json').read_text())['models'];output=RUN/'source_boundary/evaluation';methods=[]
    for m in SOURCE_METHODS:
        array=np.load(dest/(m+'.npy'),mmap_mode='r')
        for rule in RULES:methods.append(dict(method=m+'__'+rule,margin=m,decoder=rule,episodes=[]))
        for video,bank in meta['videos'].items():
            queries=[q for q in meta['episodes'] if q['video_id']==video];obs=np.asarray(bank['observations']);frames=obs[:,0];ids=obs[:,1];unique,inverse=np.unique(ids,return_inverse=True)
            matrix=np.stack([array[q['start']:q['stop']] for q in queries]);masks=decode(matrix,frames,inverse,thresholds[m]['threshold']);masks['NATIVE']=matrix>0
            for rule in RULES:
                record=next(r for r in methods if r['margin']==m and r['decoder']==rule)
                for j,q in enumerate(queries):
                    p=output/(m+'__'+rule)/'results'/video/q['expression_id']/'predict.txt'
                    chosen=[bank['lines'][k] for k in np.flatnonzero(masks[rule][j])];v2.write_new(p,''.join(x+'\n' for x in chosen))
                    if m=='FROZEN' and rule=='NATIVE':
                        old=ROOT/'results/runs/official_refer_kitti_v2_2gpu/results'/video/q['expression_id']/'predict.txt'
                        if v2.prediction(p)!=v2.prediction(old):raise ValueError('STOP: source frozen native differs from official original')
                    record['episodes'].append(dict(video=video,expression=q['expression_id'],prediction_path=str(p),prediction_sha256=file_sha(p),selected_rows=len(chosen)))
        for record in [r for r in methods if r['margin']==m]:
            if len(record['episodes'])!=862:raise ValueError('source requires862 sequences')
            record['prediction_rows']=sum(e['selected_rows'] for e in record['episodes'])
            atomic_json(output/record['method']/'predictions_seal.json',record)
    result=dict(status='ALL8_SOURCE_BOUNDARY_PREDICTIONS_SEALED',methods=methods,target_GT_open_attempts=len(attempts),
        source_frame_rule='image0-based; predicted evaluation rows original MOT1-based',CPU_seconds=time.perf_counter()-start)
    if attempts:raise RuntimeError('source boundary generator read Dance GT')
    atomic_json(RUN/'source_boundary/predictions_seal.json',result)
    atomic_json(PUBLIC/'source_boundary_seal.json',{k:v for k,v in result.items() if k!='methods'}|dict(methods=[{k:v for k,v in r.items() if k!='episodes'} for r in methods]))
    print(json.dumps(dict(event='XQUERY_SOURCE_BOUNDARY_SEALED',sets=8)),flush=True)

def evaluate():
    setup();require_plan();seal=json.loads((RUN/'source_boundary/predictions_seal.json').read_text());gt=json.loads((RUN/'source/GT_seal.json').read_text())
    if seal['status']!='ALL8_SOURCE_BOUNDARY_PREDICTIONS_SEALED':raise ValueError('seal source predictions first')
    from scripts.pf_diag_common import official_metrics
    started=time.perf_counter()
    def job(r):
        name=r['method'];folder=RUN/'source_boundary/evaluation'/name;done=folder/'official_completion.json'
        for e in r['episodes']:
            if file_sha(e['prediction_path'])!=e['prediction_sha256']:raise ValueError('source prediction changed')
        if done.exists():
            old=json.loads(done.read_text())
            if old['status']=='PASS':return old
        for e in r['episodes']:
            p=folder/'results'/e['video']/e['expression']/'gt.txt'
            if not p.exists():p.symlink_to(SOURCE/'gt_template'/e['video']/e['expression']/'gt.txt')
        begin=time.perf_counter();log=folder/'trackeval.log'
        command=[sys.executable,'./TrackEval/scripts/run_mot_challenge.py','--METRICS','HOTA','CLEAR','Identity',
            '--SEQMAP_FILE','./seqmaps/kitti-2.txt','--SKIP_SPLIT_FOL','True','--GT_FOLDER',str(SOURCE/'KITTI/training/image_02'),
            '--TRACKERS_FOLDER',str(folder/'results'),'--GT_LOC_FORMAT','{gt_folder}{video_id}/{expression_id}/gt.txt',
            '--TRACKERS_TO_EVAL',str(folder/'results'),'--USE_PARALLEL','True','--NUM_PARALLEL_CORES','2','--PLOT_CURVES','False']
        with log.open('w') as h:code=subprocess.call(command,cwd=ROOT/'vendor/FlexHook',stdout=h,stderr=subprocess.STDOUT)
        result=dict(status='PASS' if code==0 else 'FAIL',exit_code=code,method=name,margin=r['margin'],decoder=r['decoder'],
            wall_seconds=time.perf_counter()-begin,stdout_path=str(log),prediction_rows=r['prediction_rows'])
        if code:atomic_json(folder/'failed.json',result);raise RuntimeError('source official TrackEval failed: '+str(log))
        result.update(metrics=official_metrics(folder/'results'),summary_sha256=file_sha(folder/'results/pedestrian_summary.txt'),detailed_sha256=file_sha(folder/'results/pedestrian_detailed.csv'))
        atomic_json(done,result);print(json.dumps(dict(event='XQUERY_SOURCE_OFFICIAL_PASS',method=name,metrics=result['metrics'])),flush=True);return result
    with ThreadPoolExecutor(max_workers=2) as pool:results=[f.result() for f in as_completed([pool.submit(job,r) for r in seal['methods']])]
    results.sort(key=lambda r:r['method']);rows=[dict(method=r['method'],margin=r['margin'],decoder=r['decoder'],prediction_rows=r['prediction_rows'],**r['metrics']) for r in results]
    v2.write_csv(PUBLIC/'source_boundary.csv',rows)
    record=dict(status='ALL8_SOURCE_BOUNDARY_OFFICIAL_COMPLETE',methods=results,actual_batch_wall_seconds=time.perf_counter()-started,
        source_track_query_histogram=gt['track_positive_query_histogram'],source_track_fractions=gt['fractions'],physical_tracks=gt['tracks'],
        does_not_participate_in_Dance_decisions=True,source_coverage='2355528 official historical candidate-query scores/862 queries',
        caveat='same source validation split used to fit thresholds; not independent rejection generalization')
    atomic_json(RUN/'source_boundary/completion.json',record);atomic_json(PUBLIC/'source_boundary.json',record)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--evaluate',action='store_true');a=p.parse_args();evaluate() if a.evaluate else generate()
