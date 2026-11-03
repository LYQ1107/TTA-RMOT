import gzip
import json
from types import SimpleNamespace
import sqlite3


def test_archive_exports_only_used_raw_responses_without_changing_cache(tmp_path,monkeypatch):
    from scripts import active_archive
    active=tmp_path / 'active';folder=active / 'C1/FULL/rho_0.3'
    video=folder / 'videos/v';video.mkdir(parents=True)
    (folder / 'predictions_seal.json').write_text(json.dumps(dict(GT_open_attempts=0,teacher=dict(GT_open_attempts=0))))
    (video / 'questions.json').write_text(json.dumps([dict(request_id='used')]))
    (video / 'receipt.json').write_text(json.dumps(dict(question_keys=['used'])))
    (folder / 'stream_000.pt').write_bytes(b'checkpoint-fixture-not-a-model')
    cache=tmp_path / 'teacher/raw.sqlite';cache.parent.mkdir()
    database=sqlite3.connect(cache)
    database.execute('CREATE TABLE outputs (key TEXT PRIMARY KEY,owner INTEGER,payload TEXT)')
    payload=dict(yes_probability=.7,yes_logit=1.,no_logit=.2)
    database.executemany('INSERT INTO outputs VALUES (?,?,?)',
        [('used',0,json.dumps(payload)),('unused',0,'{}')]);database.commit();database.close()
    public=tmp_path / 'public';public.mkdir();launchers=tmp_path / 'launchers';launchers.mkdir()
    (launchers / 'C1_FULL_0.3.log').write_text('closed worker log')
    (launchers / 'C1_FULL_0.3_completion.json').write_text(json.dumps(dict(exit_code=0)))
    live=launchers / 'C3_FULL_0.3.log';live.write_text('another worker is still producing this log')
    monkeypatch.setattr(active_archive,'RUN',tmp_path)
    from flex_tta.runtime.pf_receipts import file_sha
    before=file_sha(cache)
    result=active_archive.package(SimpleNamespace(active=active,public=public,launchers=launchers),[dict(path=str(folder))])
    with gzip.open(public / 'teacher_responses.jsonl.gz','rt') as stream:rows=[json.loads(x) for x in stream]
    assert rows==[dict(request_id='used',response=payload)]
    assert file_sha(cache)==before and result['raw_teacher_response_keys']==1
    assert len(result['checkpoints'])==1 and result['full108_matrix_completion_not_claimed']
    import tarfile
    with tarfile.open(public / 'experiment_logs.tar.gz') as archive:names=archive.getnames()
    assert 'launchers/C1_FULL_0.3.log' in names and 'launchers/C3_FULL_0.3.log' not in names
    assert result['unpackaged_launcher_logs']==[str(live)]
    (launchers / 'C3_FULL_0.3_completion.json').write_text(json.dumps(dict(
        exit_code=None,coordinator_exit_code_observed=143,all_task_pids_confirmed_absent=True)))
    closed=active_archive.package(SimpleNamespace(active=active,public=public,launchers=launchers),[dict(path=str(folder))])
    with tarfile.open(public / 'experiment_logs.tar.gz') as archive:closed_names=archive.getnames()
    assert 'launchers/C3_FULL_0.3.log' in closed_names
    assert closed['unpackaged_launcher_logs']==[]
    durable=launchers/'durable';durable.mkdir()
    (durable/'coordinator.log').write_text('closed detached coordinator log')
    (durable/'completion.json').write_text(json.dumps(dict(exit_code=0)))
    active_archive.package(SimpleNamespace(active=active,public=public,launchers=launchers),[dict(path=str(folder))])
    with tarfile.open(public / 'experiment_logs.tar.gz') as archive:durable_names=archive.getnames()
    assert 'launchers/durable/coordinator.log' in durable_names


def test_archive_transport_roundtrip_preserves_all_bytes(tmp_path):
    from scripts.active_archive import archive_transport_parts,reassemble_archive
    from flex_tta.runtime.pf_receipts import file_sha
    original=tmp_path/'experiment_logs.tar.gz'
    original.write_bytes(bytes(range(256))*23)
    transport=archive_transport_parts(original,chunk_bytes=1024)
    manifest=json.loads(open(transport['manifest']).read())
    assert len(manifest['parts'])==6
    assert sum(part['bytes'] for part in manifest['parts'])==original.stat().st_size
    output=tmp_path/'restored.tar.gz'
    assert reassemble_archive(transport['manifest'],output)==file_sha(original)
    assert output.read_bytes()==original.read_bytes()
