import json
import time
from types import SimpleNamespace


def test_live_teacher_journal_is_not_a_final_frozen_receipt(tmp_path):
    from flex_uda.active.teacher_client import TeacherClient
    client=TeacherClient.__new__(TeacherClient)
    client.session_root=tmp_path;client.session_id='session'
    client.started=time.perf_counter();client.child=SimpleNamespace(pid=123)
    client.stats=dict(logical_calls=2,new_calls=1,cache_hits=1,forward_seconds=.2)
    client.returned_keys=['a','b']
    client.progress('ANSWERS_RETURNED_NO_FINAL_WEIGHT_SEAL')
    record=json.loads((tmp_path/'progress/session.json').read_text())
    assert record['costs']==client.stats and record['returned_request_ids']==['a','b']
    assert not record['final_weight_seal_available']
    assert not list(tmp_path.glob('*.json'))
    client.progress('SEALED')
    assert json.loads((tmp_path/'progress/session.json').read_text())['final_weight_seal_available']
