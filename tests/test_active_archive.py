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
    monkeypatch.setattr(active_archive,'RUN',tmp_path)
    from flex_tta.runtime.pf_receipts import file_sha
    before=file_sha(cache)
    result=active_archive.package(SimpleNamespace(active=active,public=public,launchers=launchers),[dict(path=str(folder))])
    with gzip.open(public / 'teacher_responses.jsonl.gz','rt') as stream:rows=[json.loads(x) for x in stream]
    assert rows==[dict(request_id='used',response=payload)]
    assert file_sha(cache)==before and result['raw_teacher_response_keys']==1
    assert len(result['checkpoints'])==1 and result['full108_matrix_completion_not_claimed']
