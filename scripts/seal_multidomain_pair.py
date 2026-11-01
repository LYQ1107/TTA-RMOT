"""Publishable GT-free score/prediction receipts, before independent GT stages."""
from argparse import ArgumentParser
import csv
import gzip
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.multidomain_diagnostics import base,inputs,verify
from scripts.multidomain_runtime import PUBLIC,RUN,guard
from scripts.report_multidomain_generality import PAIRS
from flex_tta.runtime.pf_receipts import atomic_json,file_sha


def main():
    parser=ArgumentParser();parser.add_argument('--combo',required=True,choices=['C3','C4','C5','C6'])
    args=parser.parse_args();combo,model,target=next(x for x in PAIRS if x[0]==args.combo)
    attempts=guard();meta,_,_,paths=inputs(target,model)
    folder=base(target,model);manifest=folder/'GT_free_predictions.json'
    record=json.loads(manifest.read_text());methods={};files=[]
    if record['GT_open_attempts']:raise PermissionError('Prediction generator attempted GT')
    for name in record['methods']+record['historical_methods']:
        seal=verify(target,name,model);path=folder/name/'predictions_seal.json'
        if seal['oracle']:raise ValueError('Pre-GT receipt cannot contain oracle predictions')
        methods[name]=dict(seal_path=str(path),seal_sha256=file_sha(path),
            episodes=seal['episode_count'],prediction_rows=seal['prediction_rows'])
        files.extend(dict(configuration=name,**e) for e in seal['episodes'])
    index=PUBLIC/(combo+'_prediction_files.csv.gz');output=PUBLIC/(combo+'_P1_SEALS.json')
    if output.exists():
        previous=json.loads(output.read_text())
        if previous['methods']!=methods or previous['prediction_index_sha256']!=file_sha(index):
            raise ValueError('Published pre-GT receipt changed')
        print(json.dumps(dict(status='ALREADY_SEALED_NOT_OVERWRITTEN',combo=combo)),flush=True);return
    with gzip.open(index,'wt') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(files[0]));writer.writeheader();writer.writerows(files)
    key={'KITTI-V2':'kitti','Dance':'dance','SportsMOT':'sports'}[target]
    score_path=RUN/key/model/'scores_seal.json';scores=json.loads(score_path.read_text())
    if attempts or scores['GT_open_attempts']:raise PermissionError('GT opened by pre-GT sealing')
    result=dict(status=combo+'_ALL_SCORE_AND_GT_FREE_PREDICTION_SEALS_COMPLETE_BEFORE_GT',
        source=model,target=target,videos=len(meta['videos']),episodes=len(meta['episodes']),
        complete_rows=scores['complete_rows'],historical_rows=scores['historical_rows'],
        score_seal_path=str(score_path),score_seal_sha256=file_sha(score_path),
        complete_margin_sha256=scores['complete_sha256'],historical_margin_sha256=scores['historical_sha256'],
        input_score_paths=[dict(path=str(p),sha256=file_sha(p)) for p in paths],
        methods=methods,indexed_files=len(files),prediction_index_sha256=file_sha(index),
        GT_open_attempts=0,training_updates=0,teacher_calls=0,formal_metrics='NOT_COMPUTED_YET')
    atomic_json(output,result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('methods','input_score_paths')}),flush=True)


if __name__=='__main__':main()
