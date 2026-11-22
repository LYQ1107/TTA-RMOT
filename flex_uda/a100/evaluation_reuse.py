"""Evaluation-only reuse of identical per-query predictions and official stats.

No matcher or metric formula changes. New queries run the vendor eval_sequence;
all query statistics combine with the vendor metric.combine_sequences methods.
The cache is under offline_GT and is forbidden to adaptation by its IO guard.
"""
import contextlib
import csv
from functools import partial
import hashlib
import json
from multiprocessing.pool import Pool
import os
from pathlib import Path
import pickle
import sqlite3
import sys
import time
import numpy as np
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from flex_uda.a100.runtime import ROOT

RUN=ROOT/'a100_runs/visual_classical_20261121'
sys.path.insert(0,str(ROOT/'vendor/FlexHook/TrackEval'))


def metrics():
    import trackeval
    return [trackeval.metrics.HOTA(),trackeval.metrics.CLEAR(dict(PRINT_CONFIG=False,THRESHOLD=.5)),
        trackeval.metrics.Identity(dict(PRINT_CONFIG=False,THRESHOLD=.5)),trackeval.metrics.Count()]


def cache_path():
    code=ROOT/'vendor/FlexHook/TrackEval/trackeval'
    files=[code/'eval.py',code/'datasets/mot_challenge_2d_box.py',code/'datasets/_base_dataset.py']
    files += [code/'metrics'/name for name in ('hota.py','clear.py','identity.py','count.py','_base_metric.py')]
    digest=hashlib.sha256(''.join(file_sha(p) for p in files).encode()).hexdigest()[:16]
    path=RUN/'offline_GT'/('official_episode_cache_'+digest+'.sqlite');path.parent.mkdir(parents=True,exist_ok=True)
    return path


def connect():
    db=sqlite3.connect(cache_path(),timeout=120)
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('CREATE TABLE IF NOT EXISTS episodes(pair TEXT,seq TEXT,prediction TEXT,value BLOB,reference TEXT,PRIMARY KEY(pair,seq,prediction))')
    db.execute('CREATE TABLE IF NOT EXISTS imported(folder TEXT PRIMARY KEY,seal TEXT,csv TEXT)')
    return db


def decode_row(row,metric_list):
    result={}
    for metric in metric_list:
        fields={field:float(row[field]) for field in metric.float_fields}
        fields.update({field:int(float(row[field])) for field in metric.integer_fields})
        for field in metric.float_array_fields+metric.integer_array_fields:
            fields[field]=np.asarray([float(row[field+'___'+str(int(100*x))]) for x in metric.array_labels],dtype=np.float64)
        result[metric.get_name()]=fields
    return dict(pedestrian=result)


def bootstrap():
    """Import completed immutable official tables, once per folder."""
    db=connect();metric_list=metrics();count=0
    for family in sorted(RUN.iterdir()):
        if not family.is_dir() or family.name in ('offline_GT','supervisor'):continue
        for pair in ('C4','C1'):
            for completion in sorted((family/pair).glob('*/official_completion.json')):
                folder=completion.parent
                if db.execute('SELECT 1 FROM imported WHERE folder=?',(str(folder),)).fetchone():continue
                receipt=json.loads(completion.read_text());seal_path=folder/'predictions_seal.json'
                if not seal_path.is_file() or receipt.get('offline_oracle_uses_GT'):continue
                seal=json.loads(seal_path.read_text());detail=folder/'results/pedestrian_detailed.csv'
                if seal['GT_open_attempts'] or not detail.is_file():continue
                assert file_sha(seal_path)==receipt['prediction_seal_sha256']
                assert file_sha(detail)==receipt['detailed_sha256']
                signatures={r['video']+'+'+r['expression']:r['prediction_sha256'] for r in seal['episodes']}
                entries=[]
                with detail.open() as handle:
                    for row in csv.DictReader(handle):
                        seq=row['seq']
                        if seq not in signatures:continue
                        value=pickle.dumps(decode_row(row,metric_list),protocol=5)
                        entries.append((pair,seq,signatures[seq],value,str(folder)))
                assert len(entries)==len(signatures)
                db.executemany('INSERT OR IGNORE INTO episodes VALUES(?,?,?,?,?)',entries)
                db.execute('INSERT OR IGNORE INTO imported VALUES(?,?,?)',(str(folder),file_sha(seal_path),file_sha(detail)))
                db.commit();count+=len(entries)
    total=db.execute('SELECT COUNT(*) FROM episodes').fetchone()[0];db.close()
    return dict(imported_rows=count,unique_episode_predictions=total,path=str(cache_path()),
        context='unchanged read-only C4/C1 evaluation GT and versioned vendor metric+dataset code')


def write_results(folder,res,metric_list):
    import trackeval
    summaries=[];details=[]
    for metric in metric_list:
        name=metric.get_name();table={seq:r['pedestrian'][name] for seq,r in res.items()}
        table['COMBINED_SEQ']=metric.combine_sequences(table)
        summaries.append(metric.summary_results(table));details.append(metric.detailed_results(table))
    trackeval.utils.write_summary_results(summaries,'pedestrian',str(Path(folder)/'results'))
    trackeval.utils.write_detailed_results(details,'pedestrian',str(Path(folder)/'results'))


def dataset_for(folder,pair,seal):
    import trackeval
    from flex_uda.a100.mechanism import metadata,PAIRS
    from scripts.a100_mechanism_offline import gt_root
    _,target=PAIRS[pair];m=metadata(target);truth=gt_root(target)/'gt_template';folder=Path(folder)
    if target=='KITTI-V2' and not truth.is_dir():
        from flex_uda.a100.kitti_gt_restore import recover
        truth=recover(folder)/'gt_template'
    wanted={(r['video'],r['expression']) for r in seal['episodes']};seq_info={}
    for q in m['episodes']:
        v,e=q['video_id'],q['expression_id']
        if (v,e) not in wanted:continue
        seq_info[v+'+'+e]=int(q['frame_count']);p=folder/'results'/v/e/'gt.txt'
        if not p.exists():p.symlink_to(truth/v/e/'gt.txt')
    assert len(seq_info)==len(wanted)
    cfg=dict(GT_FOLDER=str(truth),TRACKERS_FOLDER=str(folder/'results'),TRACKERS_TO_EVAL=[str(folder/'results')],
        SKIP_SPLIT_FOL=True,GT_LOC_FORMAT='{gt_folder}/{video_id}/{expression_id}/gt.txt',SEQ_INFO=seq_info,PRINT_CONFIG=False)
    return trackeval.datasets.MotChallenge2DBox(cfg),seq_info


def _evaluate(folder,pair):
    from flex_uda.presence.predictions import verify_predictions
    from trackeval.eval import eval_sequence
    from scripts.pf_diag_common import official_metrics
    from scripts.active_eval import official_stats
    folder=Path(folder)
    if (folder/'official_completion.json').is_file():return json.loads((folder/'official_completion.json').read_text())
    begin=time.perf_counter();seal=verify_predictions(folder);dataset,seq_info=dataset_for(folder,pair,seal)
    signatures={r['video']+'+'+r['expression']:r['prediction_sha256'] for r in seal['episodes']}
    db=connect();res={};missing=[];references={};metric_list=metrics()
    for seq in seq_info:
        row=db.execute('SELECT value,reference FROM episodes WHERE pair=? AND seq=? AND prediction=?',(pair,seq,signatures[seq])).fetchone()
        if row is None:missing.append(seq)
        else:res[seq]=pickle.loads(row[0]);references[seq]=row[1]
    with (folder/'trackeval.log').open('w') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        if missing:
            job=partial(eval_sequence,dataset=dataset,tracker=str(folder/'results'),class_list=['pedestrian'],
                metrics_list=metric_list,metric_names=[m.get_name() for m in metric_list])
            with Pool(2) as pool:computed=pool.map(job,missing)
            for seq,value in zip(missing,computed):res[seq]=value
        ordered={seq:res[seq] for seq in seq_info};write_results(folder,ordered,metric_list)
    for seq in missing:
        db.execute('INSERT OR IGNORE INTO episodes VALUES(?,?,?,?,?)',(pair,seq,signatures[seq],pickle.dumps(res[seq],protocol=5),str(folder)))
    db.commit();db.close();stats=official_stats(folder);values=official_metrics(folder/'results')
    assert int(values['Dets'])==seal['prediction_rows']
    result=dict(status='PASS',pair=pair,metrics=values,HOTA_unrounded=stats['HOTA_unrounded'],offline_oracle_uses_GT=False,
        wall_seconds=time.perf_counter()-begin,prediction_seal_sha256=file_sha(folder/'predictions_seal.json'),
        detailed_sha256=file_sha(folder/'results/pedestrian_detailed.csv'),
        official_episode_reuse=dict(cached=len(references),new=len(missing),same_sequence_and_prediction_sha_required=True,
            cache=str(cache_path()),references=references,matching_and_combination='unmodified vendor TrackEval',
            no_metric_approximation=True))
    atomic_json(folder/'official_completion.json',result)
    print(json.dumps(dict(event='MECHANISM_OFFICIAL_DONE',folder=str(folder),HOTA=stats['HOTA_unrounded'],
        reused=len(references),new=len(missing))),flush=True)
    return result


def evaluate(folder,pair):
    """Official evaluation in a fresh interpreter, separate from IO guards.

    Quantity generation can install a permanent Python audit hook. A fork
    inherits it; an exec does not. Keep that guard intact in the generating
    process and give only the sealed prediction path to an evaluation worker.
    The vendor matching, metric functions and exact episode cache are unchanged.
    """
    import subprocess
    folder=Path(folder)
    receipt=folder/'official_completion.json'
    if receipt.is_file():return json.loads(receipt.read_text())
    log=folder/'official_eval_process.log'
    with log.open('w') as handle:
        process=subprocess.run([sys.executable,'-m','flex_uda.a100.evaluation_reuse',
            '--folder',str(folder),'--pair',pair],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'))
    if process.returncode:
        raise RuntimeError('Isolated official evaluation failed: '+str(log)+'\n'+log.read_text()[-6000:])
    return json.loads(receipt.read_text())


if __name__=='__main__':
    import argparse
    from flex_uda.a100.mechanism import PAIRS
    parser=argparse.ArgumentParser();parser.add_argument('--folder',required=True);parser.add_argument('--pair',choices=tuple(PAIRS),required=True)
    args=parser.parse_args();_evaluate(args.folder,args.pair)
