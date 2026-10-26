import json
from flex_uda.numerics import reconcile_resume


def test_resume_preserves_failed_suffix_and_replays_checkpoint_boundary(tmp_path):
    records=[dict(step=i,wall_seconds=float(i)) for i in range(1,5)]
    (tmp_path/'metrics_train.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    (tmp_path/'pseudo_evidence.jsonl').write_text('e1\ne2\ne3\ne4\n')
    state=dict(step=2,tracking=dict(wall_seconds=2.,counters=dict(target_candidate_rows=2)))
    lost=reconcile_resume(tmp_path,state)
    assert lost==2 and [json.loads(s)['step'] for s in (tmp_path/'metrics_train.jsonl').read_text().splitlines()]==[1,2]
    assert (tmp_path/'pseudo_evidence.jsonl').read_text()=='e1\ne2\n'
    archived=list((tmp_path/'resume_discarded_traces').glob('*/metrics_train.jsonl'))
    assert len(archived)==1 and [json.loads(s)['step'] for s in archived[0].read_text().splitlines()]==[3,4]
    assert list((tmp_path/'resume_discarded_traces').glob('*/pseudo_evidence.jsonl'))[0].read_text()=='e3\ne4\n'
