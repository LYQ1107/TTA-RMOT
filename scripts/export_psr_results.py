#!/usr/bin/env python3
"""Export complete prediction/log artifacts, with large score/weight recovery index."""
import argparse
import gzip
import json
from pathlib import Path
import shutil
import sys
import tarfile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from scripts.snapshot_psr_stage import snapshot


def archive(path,files):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('wb') as out,gzip.GzipFile(filename='',fileobj=out,mode='wb',mtime=0) as gz,tarfile.open(fileobj=gz,mode='w|') as tar:
        for source,name in sorted(files,key=lambda x:x[1]):
            info=tar.gettarinfo(str(source),arcname=name)
            info.mtime=info.uid=info.gid=0;info.uname=info.gname='';info.mode=0o644
            with source.open('rb') as stream:tar.addfile(info,stream)
    return dict(path=str(path),sha256=file_sha(path),bytes=path.stat().st_size,files=len(files))


def export(run,public):
    run,public=Path(run).resolve(),Path(public).resolve();root=run/'evaluation/merged'
    seal=json.loads((root/'controls_sealed.json').read_text());complete=json.loads((root/'official_evaluation_completion.json').read_text())
    if complete['status']!='COMPLETE':raise ValueError('complete official results required')
    snapshot();index={};logs=[]
    for method in seal['methods']:
        folder=root/method;dest=public/method;dest.mkdir(parents=True,exist_ok=True)
        for name in ('full_audit.json','run_metadata.json','reference_metadata.json','inference_start_provenance.json'):
            if (folder/name).exists():shutil.copy2(folder/name,dest/name)
        for name in ('pedestrian_summary.txt','pedestrian_detailed.csv'):
            shutil.copy2(folder/'results'/name,dest/name)
        audit=json.loads((folder/'full_audit.json').read_text());files=[];episodes=[]
        for row in audit['episodes']:
            video=row['video'];query=row.get('expression',row.get('expression_id'));episode=folder/'results'/video/query
            prediction=episode/'predict.txt';score=episode/'candidate_margins.jsonl.gz'
            files.append((prediction,str(Path(video)/query/'predict.txt')))
            episodes.append(dict(video=video,query=query,prediction_path=str(prediction),prediction_sha256=file_sha(prediction),
                score_stream_path=str(score),score_stream_sha256=file_sha(score)))
            for name in ('completion.json','reference_receipt.json'):
                if (episode/name).exists():files.append((episode/name,str(Path(video)/query/name)))
        if len(episodes)!=425:raise ValueError('missing episode export')
        tar=archive(public/'archives'/(method+'_425_predictions.tar.gz'),files)
        atomic_json(dest/'episode_recovery.json',dict(episodes=episodes,score_streams_stay_on_server=True,predictions_public=tar))
        index[method]=tar
        for source in folder.rglob('*.log'):
            if 'results' not in source.relative_to(folder).parts:logs.append((source,str(Path('inference')/method/source.relative_to(folder))))
        for source in (root/'trackeval_jobs'/method).glob('*.txt'):logs.append((source,str(Path('official_TrackEval')/method/source.name)))
        for source in (folder/'parity').rglob('*.json'):
            target=dest/'parity'/source.relative_to(folder/'parity');target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    for pattern in ('teacher_full/*/*.txt','teacher_smoke/*/*.txt','matrix_launchers/*/*.json','cpu_pipeline/*/*.txt','cpu_pipeline/*/*.json','*stdout.txt'):
        logs.extend((p,str(p.relative_to(run))) for p in run.glob(pattern) if p.is_file())
    for phase in ('students','interface','smoke'):
        logs.extend((p,str(p.relative_to(run))) for p in (run/phase).rglob('*.log'))
    stdout=ROOT/'results/runs/psr_matrix_20261029_stdout.txt'
    if stdout.exists():logs.append((stdout,'psr_matrix_stdout.txt'))
    logs=[(p,name) for name,p in {name:p for p,name in logs}.items()]
    index['runtime_logs']=archive(public/'logs/runtime_logs.tar.gz',logs)
    for name in ('controls_sealed.json','official_evaluation_completion.json'):
        shutil.copy2(root/name,public/name)
    if (run/'mean_teacher_prefetch_completion.json').exists():shutil.copy2(run/'mean_teacher_prefetch_completion.json',public/'mean_teacher_prefetch_completion.json')
    atomic_json(public/'artifact_index.json',dict(status='COMPLETE_PREDICTIONS_LOGS_AND_RECOVERY',methods=index,
        private_run_root=str(run),large_score_streams='per-method episode_recovery.json: exact paths and SHA256',
        checkpoints='results/psr/training/*/training_receipt.json: exact paths and SHA256',
        raw_teacher_root=str(run/'teacher_full'),pseudo_root='/data2/user/TTA-RMOT-assets/psr/val_v1',
        raw_teacher_cache='/data2/user/TTA-RMOT-assets/psr/cache/raw.sqlite',
        no_images_weights_or_complete_raw_MLLM_cache_uploaded=True))
    print(json.dumps(dict(status='PSR_ARTIFACT_EXPORT_COMPLETE',methods=len(seal['methods']),runtime_log_files=len(logs))))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-root',type=Path,required=True)
    p.add_argument('--output',type=Path,default=ROOT/'results/psr/full');a=p.parse_args();export(a.run_root,a.output)
