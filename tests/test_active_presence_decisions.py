import numpy as np


def test_offline_presence_distinguishes_false_absence_and_unobservable_target(tmp_path):
    from scripts.report_active import saved_presence_decisions
    root=tmp_path / 'videos/v';root.mkdir(parents=True)
    np.savez(root / 'belief.npz',tracks=[1,2],queries=['visible','missed','empty'],
        expected_counts=[.5,.5,.5],low_presence=[True,True,True],
        confirmed_present=[True,False,False],absent=[True,True,True],
        posterior=np.zeros((3,2)))
    np.save(root / 'mask.npy',np.zeros((3,6),bool))
    meta=dict(videos=dict(v=dict(observations=[(t,i) for t in (1,2,3) for i in (1,2)])))
    zero=np.zeros(6,bool)
    truth={('v','visible'):dict(labels=np.array([1,0,1,0,1,0],bool),counts=np.ones(3)),
        ('v','missed'):dict(labels=zero,counts=np.ones(3)),
        ('v','empty'):dict(labels=zero,counts=np.zeros(3))}
    rows=saved_presence_decisions(tmp_path,meta,truth,'C1','FULL',.3)
    assert rows[0]['false_absence_with_retrievable_GT'] and rows[0]['teacher_contradiction']
    assert rows[0]['matched_candidate_positive_tracks_majority']==1
    assert rows[1]['query_present_GT'] and not rows[1]['false_absence_with_retrievable_GT']
    assert not rows[2]['query_present_GT'] and rows[2]['selected_prediction_rows']==0
    assert all(row['ORACLE_GT_DIAGNOSTIC_ONLY'] and not row['GT_feedback_to_runtime'] for row in rows)
