"""Task contracts stay invariant under batch composition."""
import pytest
from flex_uda.active.v8p import classify_protocol, CHECKLIST, PRESUPPOSED


def test_declared_presence_does_not_flip_for_single_video():
    queries={'first':['white player'],'second':['red player']}
    for batch in (queries,{'first':queries['first']},{'second':queries['second']}):
        value=classify_protocol(batch,task_protocol=PRESUPPOSED)
        assert value['protocol']==PRESUPPOSED
        assert value['presence_inference'] is False
        assert value['protocol_source']=='EXPLICIT_TASK_CONTRACT'


def test_declared_checklist_does_not_require_identical_lists():
    queries={'first':['white player','red player'],'second':['white player']}
    assert classify_protocol(queries,task_protocol=CHECKLIST)['presence_inference'] is True


def test_legacy_behavior_and_invalid_contract():
    queries={'first':['white player'],'second':['red player']}
    assert classify_protocol(queries)['protocol']==PRESUPPOSED
    assert classify_protocol({'first':queries['first']})['protocol']==CHECKLIST
    with pytest.raises(ValueError,match='Unknown declared task protocol'):
        classify_protocol(queries,task_protocol='GUESS')
