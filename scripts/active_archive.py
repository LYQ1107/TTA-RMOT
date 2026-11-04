"""Package completed active logs and exact cached teacher responses, CPU only."""
import gzip
import json
from pathlib import Path
import sqlite3
import tarfile
import shutil
from flex_tta.runtime.pf_receipts import atomic_json,file_sha
from flex_uda.presence.runtime import RUN


def archive_transport_parts(archive_path,chunk_bytes=16 << 20):
    """Byte-preserving transport after GitHub rejects the large blob (422)."""
    archive_path=Path(archive_path);folder=archive_path.with_name(archive_path.name+'.parts')
    folder.mkdir(parents=True,exist_ok=True);parts=[]
    with archive_path.open('rb') as source:
        index=0
        while payload:=source.read(chunk_bytes):
            path=folder / ('part_%03d.bin' % index)
            path.write_bytes(payload)
            parts.append(dict(name=path.name,bytes=len(payload),sha256=file_sha(path)))
            index+=1
    manifest=folder / 'manifest.json'
    record=dict(status='BYTE_EXACT_SPLIT_FOR_GITHUB_API',original_name=archive_path.name,
        original_bytes=archive_path.stat().st_size,original_sha256=file_sha(archive_path),
        chunk_bytes=chunk_bytes,parts=parts,
        reason='GitHub create-blob API returned 422: input too large; original archive retained on server')
    atomic_json(manifest,record)
    return dict(manifest=str(manifest),manifest_sha256=file_sha(manifest),parts=parts,
        original_archive_retained=True,all_bytes_in_order=True)


def reassemble_archive(manifest_path,output):
    manifest_path=Path(manifest_path);output=Path(output)
    record=json.loads(manifest_path.read_text())
    if output.exists():
        if file_sha(output)!=record['original_sha256']:
            raise ValueError('Do not overwrite a different existing archive: '+str(output))
        return record['original_sha256']
    for row in record['parts']:
        part=manifest_path.parent / row['name']
        if part.stat().st_size!=row['bytes'] or file_sha(part)!=row['sha256']:
            raise ValueError('Archive transport part changed: '+str(part))
    with output.open('xb') as destination:
        for row in record['parts']:
            with (manifest_path.parent / row['name']).open('rb') as source:
                shutil.copyfileobj(source,destination)
    if output.stat().st_size!=record['original_bytes'] or file_sha(output)!=record['original_sha256']:
        raise ValueError('Reassembled archive does not match the original SHA256')
    return record['original_sha256']


def prepare_existing_archive(logs_manifest):
    """Split an already-closed archive without repacking/changing its bytes."""
    logs_manifest=Path(logs_manifest);record=json.loads(logs_manifest.read_text())
    archive=record['archive'];path=Path(archive['path'])
    if path.stat().st_size!=archive['bytes'] or file_sha(path)!=archive['sha256']:
        raise ValueError('Existing log archive changed after sealing')
    archive['github_transport']=archive_transport_parts(path)
    atomic_json(logs_manifest,record)
    return archive


def package(execution,rows):
    roots=[Path(row['path']) for row in rows if Path(row['path']).is_relative_to(execution.active)]
    keys=set();files=[];checkpoints=[];unpackaged_launcher_logs=[]
    for root in roots:
        seal=json.loads((root / 'predictions_seal.json').read_text())
        if seal['GT_open_attempts'] or seal.get('teacher',{}).get('GT_open_attempts',0):
            raise ValueError('Runtime/teacher GT isolation failed')
        for path in (root / 'videos').glob('*/questions.json'):
            keys.update(q['request_id'] for q in json.loads(path.read_text()))
        for path in (root / 'videos').glob('*/receipt.json'):
            keys.update(json.loads(path.read_text()).get('question_keys',[]))
        for path in root.rglob('*'):
            if not path.is_file() or path.is_symlink():continue
            if path.suffix in ('.log','.json','.jsonl') or path.name in ('pedestrian_detailed.csv','pedestrian_summary.txt'):
                files.append((path,'methods/'+str(path.relative_to(execution.active))))
            elif path.suffix=='.pt':
                checkpoints.append(dict(path=str(path),bytes=path.stat().st_size,sha256=file_sha(path)))
    for path in execution.launchers.rglob('*'):
        if path.is_file() and path.suffix in ('.log','.json'):
            if path.suffix=='.log':
                receipts=[path.with_name(path.stem+'_completion.json'),path.with_suffix('.json')]
                if path.name=='coordinator.log':receipts.append(path.parent/'completion.json')
                terminal=any(p.exists() and (json.loads(p.read_text()).get('exit_code') is not None or
                    json.loads(p.read_text()).get('all_task_pids_confirmed_absent',False)) for p in receipts)
                if not terminal:
                    unpackaged_launcher_logs.append(str(path));continue
            files.append((path,'launchers/'+str(path.relative_to(execution.launchers))))
    cache=RUN / 'teacher/raw.sqlite'
    raw=execution.public / 'teacher_responses.jsonl.gz'
    if keys:
        # A read-only snapshot of only this stage's actual input keys. Never
        # invoke a backend or claim new cache leases in an offline reporter.
        with sqlite3.connect(cache.resolve().as_uri()+'?mode=ro',uri=True) as database,gzip.open(raw,'wt') as output:
            for key in sorted(keys):
                row=database.execute('SELECT payload FROM outputs WHERE key=?',(key,)).fetchone()
                if row is None or row[0] is None:raise ValueError('Missing completed teacher response '+key)
                output.write(json.dumps(dict(request_id=key,response=json.loads(row[0])),sort_keys=True)+'\n')
    manifest=[]
    archive_path=execution.public / 'experiment_logs.tar.gz'
    with tarfile.open(archive_path,'w:gz') as archive:
        for path,name in sorted(files,key=lambda row:row[1]):
            manifest.append(dict(path=str(path),archive_path=name,bytes=path.stat().st_size,sha256=file_sha(path)))
            archive.add(path,arcname=name,recursive=False)
    for row in manifest:
        if file_sha(row['path'])!=row['sha256']:raise RuntimeError('Log owner has not finished: '+row['path'])
    record=dict(status='COMPLETED_SEALED_METHOD_LOGS_PACKAGED',completed_active_methods=len(roots),
        full108_matrix_completion_not_claimed=True,files=manifest,checkpoints=checkpoints,
        unpackaged_launcher_logs=unpackaged_launcher_logs,
        launcher_log_scope='only after worker/evaluator wait() receipt; unfinished paths retained, not claimed completed',
        archive=dict(path=str(archive_path),bytes=archive_path.stat().st_size,sha256=file_sha(archive_path),
            github_transport=archive_transport_parts(archive_path) if archive_path.stat().st_size>32 << 20 else None),
        raw_teacher_response_keys=len(keys),raw_teacher_cache_source=str(cache),
        raw_response_cost_note='one immutable original response per input; per-method actual/cache/equivalent costs are in costs.csv',
        raw_teacher_responses=dict(path=str(raw),bytes=raw.stat().st_size,sha256=file_sha(raw),
            github_transport=archive_transport_parts(raw) if raw.stat().st_size>32 << 20 else None) if keys else None,
        large_models_and_predictions_not_in_git=True,teacher_or_GPU_called_by_packager=False)
    atomic_json(execution.public / 'logs.json',record)
    return record


if __name__=='__main__':
    from argparse import ArgumentParser
    parser=ArgumentParser(description='Reassemble the exact published log archive, without a model or GT')
    parser.add_argument('--reassemble',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    arguments=parser.parse_args()
    print(json.dumps(dict(archive=str(arguments.output),sha256=reassemble_archive(arguments.reassemble,arguments.output))))
