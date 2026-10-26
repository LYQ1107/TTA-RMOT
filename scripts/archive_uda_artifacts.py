#!/usr/bin/env python3
"""Compact immutable experiment evidence, no images/weights/GT in publication."""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import tarfile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from flex_uda.training import METHODS
from flex_tta.runtime.pf_receipts import atomic_json,file_sha


def compressed_copy(source,dest):
    with Path(source).open('rb') as src,Path(dest).open('wb') as out:
        with gzip.GzipFile(filename='',fileobj=out,mode='wb',mtime=0) as zipped:shutil.copyfileobj(src,zipped)


def pack_logs(training,evaluation,public):
    training,evaluation,public=map(lambda p:Path(p).resolve(),(training,evaluation,public))
    for method in METHODS:
        audit=json.loads((evaluation/method/'full_audit.json').read_text())
        if audit['status']!='PASS' or audit['valid_episodes']!=425:
            raise ValueError('publish runtime archive only after all four425 audits')
    roots=[training,evaluation,ROOT/'results/runs/uda_val_smoke_20261026_v1',ROOT/'results/runs/uda_val_smoke_20261026_v2']
    record_names={'gpu_admission.json','launcher_status.json','evaluation_launcher_status.json',
        'parity_gpu_admission.json','evaluation_gpu_admission.json','parity_launcher_status.json',
        'full_suite_completion.json','evaluation_suite_completion.json'}
    paths=set((ROOT/'results/uda/assets').glob('*.txt'))
    for root in roots:
        paths.update(p for p in root.rglob('*') if p.is_file() and
            (p.suffix=='.log' or p.name.endswith('_stdout.txt') or
             (p.name.startswith('log_rank') and p.suffix=='.txt') or p.name in record_names))
    dest=public/'logs';dest.mkdir(parents=True,exist_ok=True);archive=dest/'runtime_logs.tar.gz';records=[]
    with archive.open('wb') as out,gzip.GzipFile(filename='',fileobj=out,mode='wb',mtime=0) as zipped:
        with tarfile.open(fileobj=zipped,mode='w') as tar:
            for path in sorted(paths):
                data=path.read_bytes();name=path.relative_to(ROOT).as_posix()
                member=tarfile.TarInfo(name);member.size=len(data);member.mtime=0;member.mode=0o644
                tar.addfile(member,io.BytesIO(data))
                records.append(dict(path=name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
    with tarfile.open(archive,'r:gz') as tar:
        members=tar.getmembers()
        if len(members)!=len(records):raise ValueError('runtime archive count changed')
        for member,record in zip(members,records):
            if member.name!=record['path'] or hashlib.sha256(tar.extractfile(member).read()).hexdigest()!=record['sha256']:
                raise ValueError('runtime archive differs from original log')
    result=dict(status='VERIFIED_COMPLETE_UDA_TEXT_LOG_ARCHIVE',files=len(records),
        uncompressed_bytes=sum(r['bytes'] for r in records),archive_bytes=archive.stat().st_size,
        archive_sha256=file_sha(archive),archive=str(archive.relative_to(ROOT)),records=records,
        scope='preparation tests; revoked v1 and valid v2 smoke; full training and failures/recovery; GPU admission/launcher status; parity/inference; all official evaluations',
        protocol='TRANSDUCTIVE_VAL_DIAGNOSTIC',raw_images_checkpoints_predictions_GT_margin_streams_excluded=True)
    atomic_json(dest/'runtime_log_index.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))


def export(training,evaluation,public,methods,training_only=False):
    training,evaluation,public=map(lambda p:Path(p).resolve(),(training,evaluation,public))
    indexes={}
    for method in methods:
        folder=training/method;dest=public/method;dest.mkdir(parents=True,exist_ok=True)
        receipt=json.loads((folder/'training_receipt.json').read_text())
        if receipt['status']!='COMPLETE_FIXED_COVERAGE_DIAGNOSTIC' or receipt['steps']!=8973:raise ValueError('archive only completed coverage')
        for name in ('training_receipt.json','source_validation.jsonl','provenance.json','config.json','numerical_retries.jsonl','resume_trace_recovery.json','training_finite_audit.json'):
            if (folder/name).exists():shutil.copy2(folder/name,dest/name)
        for name in ('metrics_train.jsonl',):compressed_copy(folder/name,dest/(name+'.gz'))
        compressed_copy(training/(method+'_stdout.txt'),dest/'training_stdout.txt.gz')
        provenance=[json.loads(p.read_text()) for p in sorted((folder/'provenance_attempts').glob('*.json'))]
        provenance.append(json.loads((folder/'provenance.json').read_text()))
        atomic_json(dest/'training_provenance_history.json',provenance)
        config=json.loads((folder/'config.json').read_text())
        import yaml
        (dest/'config.yaml').write_text(yaml.safe_dump(config,sort_keys=False,allow_unicode=True))
        index=dict(training_root=str(folder),evaluation_root=str(evaluation/method),
            checkpoints={n:dict(path=str(folder/n),sha256=file_sha(folder/n)) for n in ('checkpoint_latest.pt','checkpoint_best_source_selected.pt')},
            plan_sha256=provenance[-1]['plan_sha256'],training_trace=str(folder/'metrics_train.jsonl'),
            checkpoint_state='student/EMA/optimizer/constant-LR/scaler/RNG/step/coverage/memory',
            raw_data_and_weights_not_in_Git=True,original_failed_suffixes=str(folder/'resume_discarded_traces'))
        sampling={}
        for domain,name in (('source','source_train_plan.jsonl'),('target','target_plan.jsonl')):
            plan=[json.loads(s) for s in (folder/'plans'/name).read_text().splitlines()]
            sampling[domain]=dict(windows=len(plan),videos=len({r['video_id'] for r in plan}),
                unique_anchor_frames=len({(r['video_id'],r['frame']) for r in plan}),
                query_pairs=len({(r['video_id'],r['expression_id']) for r in plan}))
        index['sampling']=dict(domains=sampling,all_frame_optimization=False,
            note='coverage means all videos/queries/eligible source tracks; each anchor also reads causal track histories; SOURCE_REHEARSAL does not optimize target windows')
        if not training_only:
            out=evaluation/method;audit=json.loads((out/'full_audit.json').read_text())
            if audit['valid_episodes']!=425 or audit['status']!='PASS':raise ValueError('archive all425 predictions only')
            metadata=json.loads((out/'run_metadata.json').read_text());archive=public/'archives'/(method+'_425_predictions.tar.gz')
            archive.parent.mkdir(parents=True,exist_ok=True)
            if not archive.exists():
                with tarfile.open(archive,'w:gz') as tar:
                    for e in metadata['episodes']:
                        path=out/'results'/e['video']/e['expression_id']
                        if file_sha(path/'predict.txt')!=e['prediction_sha256']:raise ValueError('prediction receipt hash mismatch')
                        for name in ('predict.txt','completion.json'):tar.add(path/name,arcname=str((path/name).relative_to(out)),recursive=False)
            with tarfile.open(archive,'r:gz') as tar:
                names=tar.getnames()
                if len(names)!=850 or sum(n.endswith('/predict.txt') for n in names)!=425 or any(n.endswith('/gt.txt') for n in names):
                    raise ValueError('prediction archive coverage/GT violation')
            index.update(prediction_archive=dict(path=str(archive),sha256=file_sha(archive),episodes=425),
                score_streams='per-episode candidate_margins.jsonl.gz on server; hashes in episode_receipts.json')
            if (folder/'pseudo_evidence.jsonl').exists():index['pseudo_evidence']=dict(path=str(folder/'pseudo_evidence.jsonl'),sha256=file_sha(folder/'pseudo_evidence.jsonl'))
        atomic_json(dest/'raw_recovery_index.json',index);indexes[method]=index
    atomic_json(public/('training_stage_recovery_index.json' if training_only else 'full_recovery_index.json'),indexes)
    print(json.dumps(dict(methods=methods,training_only=training_only,public_root=str(public)),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--training-root',type=Path,required=True)
    p.add_argument('--evaluation-root',type=Path,required=True);p.add_argument('--public-root',type=Path,default=ROOT/'results/uda/full')
    p.add_argument('--methods',nargs='+',choices=METHODS,default=list(METHODS));p.add_argument('--training-only',action='store_true')
    p.add_argument('--logs-only',action='store_true')
    a=p.parse_args()
    if a.logs_only:pack_logs(a.training_root,a.evaluation_root,a.public_root)
    else:export(a.training_root,a.evaluation_root,a.public_root,a.methods,a.training_only)
